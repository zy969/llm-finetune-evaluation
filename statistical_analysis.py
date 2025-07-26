"""
Statistical analysis script for LLM fine-tuning evaluation metrics.

Performs normality testing, ANOVA or non-parametric tests (Friedman, Wilcoxon),
and computes effect sizes (Cohen's d) across model versions on various tasks.
"""

import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import shapiro, friedmanchisquare, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import re


metrics_per_task = {
    'qa': ['exact_match', 'f1'],
    'summarization': ['rouge1', 'rougeL', 'rougeLsum'],
    'instruction': ['gpt4_win_rate']
}


def load_all_runs(log_dir="."):
    all_csv = glob.glob(os.path.join(log_dir, "*.csv"))
    all_csv = [f for f in all_csv if "summary" not in f]
    
    print(f"[INFO] Found {len(all_csv)} CSV files for analysis.")
    
    records = []
    for path in all_csv:
        fname = os.path.basename(path)
        match = re.match(r"(instruction|qa|summarization)_[^_]+_(llama2-7b)_(base|full|lora)_run(\d)\.csv", fname)
        if not match:
            print(f"[WARN] Skipping {fname}: filename format not recognized.")
            continue
        task, model_family, version, run = match.groups()
        run = int(run)
      
        try:
            df = pd.read_csv(path)
            df["task"] = task.split("_")[0]
            df["model_family"] = model_family
            df["version"] = version
            df["run"] = run
            records.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
    
    if not records:
        raise ValueError("No valid CSV data found. Please check file existence and naming.")
    
    return pd.concat(records, ignore_index=True)

df = load_all_runs()



def analyze_metric(df_task, metric):
    pivot_df = df_task.pivot_table(index='run', columns='version', values=metric)

    print(f"\n== Analysis for metric: {metric} ==")
    print("Pivot table (rows: runs, columns: models):")
    print(pivot_df)

    pivot_df = pivot_df.dropna(axis=1, how='all')
    if pivot_df.shape[1] < 2:
        print(f"Skipping metric {metric} — only {list(pivot_df.columns)} available, insufficient models for comparison.")
        return

    normality = {}
    for model in pivot_df.columns:
        data = pivot_df[model].dropna()
        if len(data) < 3:
            print(f"Model {model} has too few samples ({len(data)}), skipping normality test — assuming non-normal.")
            normality[model] = 0
        else:
            stat, p = shapiro(data)
            normality[model] = p
            print(f'Shapiro-Wilk test for {model}: p={p:.4f}')

    all_normal = all(p > 0.05 for p in normality.values())

    if all_normal:
        print("All model data passed normality check.")
        df_melt = pivot_df.reset_index().melt(id_vars='run', var_name='model', value_name=metric)
        anova = AnovaRM(df_melt, depvar=metric, subject='run', within=['model']).fit()
        print(anova)

        tukey = pairwise_tukeyhsd(endog=df_melt[metric], groups=df_melt['model'], alpha=0.05)
        print(tukey)
    else:
        print("At least one model failed the normality test.")
        stat, p = friedmanchisquare(*[pivot_df[model].dropna() for model in pivot_df.columns])
        print(f"Friedman test p={p:.4f}")

        models = pivot_df.columns
        comparisons, p_values = [], []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                s1, s2 = pivot_df[models[i]], pivot_df[models[j]]
                s1, s2 = s1.align(s2, join='inner')
                stat, pval = wilcoxon(s1, s2)
                comparisons.append((models[i], models[j]))
                p_values.append(pval)

        reject, pvals_corrected, _, _ = multipletests(p_values, method='holm')
        for idx, (comp, p_corr) in enumerate(zip(comparisons, pvals_corrected)):
            print(f'Wilcoxon between {comp[0]} and {comp[1]}: corrected p = {p_corr:.4f}')

  

    print("\nEffect sizes (Cohen's d, pooled standard deviation):")
    models = pivot_df.columns
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            try:
                x = pivot_df[models[i]].dropna()
                y = pivot_df[models[j]].dropna()
                x, y = x.align(y, join='inner')

                if len(x) < 2:
                    print(f"Not enough paired samples for {models[i]} vs {models[j]}, skipping.")
                    continue

                diff_mean = np.mean(x - y)

                std_x = np.std(x, ddof=1)
                std_y = np.std(y, ddof=1)
                pooled_std = np.sqrt((std_x**2 + std_y**2) / 2)

                if pooled_std < 1e-6:
                    print(f"Warning: pooled std too small between {models[i]} and {models[j]}, skipping.")
                    continue

                d = diff_mean / pooled_std
                print(f"Pooled Cohen's d between {models[i]} and {models[j]}: {d:.3f}")
            except Exception as e:
                print(f"Error computing effect size for {models[i]} vs {models[j]}: {e}")


for task, metrics in metrics_per_task.items():
    df_task = df[df["task"] == task]
    for metric in metrics:
        if metric in df_task.columns:
            analyze_metric(df_task, metric)
