"""
Visualization script for plotting LLM fine-tuning performance 
across tasks.

Generates bar charts with error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('summary1.csv')


efficiency_metrics = {
    'latency': 'Latency (s)',
    'gpu_memory': 'GPU Memory (MB)',
    'cpu_usage': 'CPU Usage (%)'
}


model_order = ['base', 'lora', 'full']
model_labels = {'base': 'Base', 'lora': 'LoRA', 'full': 'Full Fine-tuned'}
colors = ['#4C72B0', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 3, figsize=(18,5), constrained_layout=True)

for i, (metric, ylabel) in enumerate(efficiency_metrics.items()):
    ax = axes[i]
    width = 0.2
    tasks = df['task'].unique()
    x = np.arange(len(tasks))
    
    for j, model in enumerate(model_order):
        means = []
        stds = []
        for task in tasks:
            subset = df[(df['task']==task) & (df['version']==model)]
            if subset.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(subset[metric].values[0])
                stds.append(subset[metric + '_std'].values[0])
        
        pos = x + (j - 1) * width  
        ax.bar(pos, means, width=width, label=model_labels[model], yerr=stds, capsize=5, color=colors[j])
    
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    if i == 0:
        ax.legend()

plt.savefig('inference_efficiency.png')
plt.show()
