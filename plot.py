"""
Visualization script for plotting LLM fine-tuning performance 
across tasks.

Generates bar charts with error bars.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('summary1.csv')


task_metrics = {
    'qa': ['exact_match', 'f1'],
    'summarization': ['rouge1', 'rougeL', 'rougeLsum'],
    'instruction': ['gpt4_win_rate']
}


model_order = ['base', 'lora', 'full']


colors = ['#4C72B0', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 3, figsize=(18,5), constrained_layout=True)

for i, (task, metrics) in enumerate(task_metrics.items()):
    ax = axes[i]
    width = 0.2
    x = np.arange(len(metrics))
    
    for j, model in enumerate(model_order):
        subset = df[(df['task']==task) & (df['version']==model)]
        means = []
        stds = []
        for m in metrics:
            means.append(subset[m].values[0])
            stds.append(subset[m + '_std'].values[0])
        pos = x + (j - 1) * width  # -1,0,1分布
        ax.bar(pos, means, width=width, label=model.capitalize(), yerr=stds, capsize=5, color=colors[j])
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_title(f'{task.capitalize()} Performance')
    ax.set_ylabel('Score')
    ax.legend()

plt.savefig('task_performance_combined.png')
plt.show()
