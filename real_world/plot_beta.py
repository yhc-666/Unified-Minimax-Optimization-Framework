# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font to support better rendering
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

datasets = {
    'coat': {
        'csv_file': '../coat_beta_evaluation_results.csv',
        'metric': 'NDCG@5',
        'f1_metric': 'F1@5'
    },
    'yahoo': {
        'csv_file': '../yahoo_beta_evaluation_results.csv', 
        'metric': 'NDCG@5',
        'f1_metric': 'F1@5'
    },
    'kuai': {
        'csv_file': '../kuai_beta_evaluation_results.csv',
        'metric': 'NDCG@20',
        'f1_metric': 'F1@20'
    }
}

# Count available datasets
available_datasets = []
for dataset_name, dataset_info in datasets.items():
    if os.path.exists(dataset_info['csv_file']):
        available_datasets.append((dataset_name, dataset_info))
    else:
        print(f"Skipping {dataset_name}: file not found")

# Create subplots based on available datasets
n_datasets = len(available_datasets)
if n_datasets == 0:
    print("No datasets found!")
    exit()

fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6))
if n_datasets == 1:
    axes = [axes]

fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)

for idx, (dataset_name, dataset_info) in enumerate(available_datasets):
    df = pd.read_csv(dataset_info['csv_file'])
    
    beta_values = df['beta'].values
    ndcg_values = df[dataset_info['metric']].values
    f1_values = df[dataset_info['f1_metric']].values
    
    # Create first y-axis for NDCG
    color1 = 'tab:blue'
    ax1 = axes[idx]
    ax1.semilogx(beta_values, ndcg_values, 'o-', color=color1, linewidth=2, markersize=8, label=dataset_info['metric'])
    ax1.set_xlabel('beta', fontsize=12)
    ax1.set_ylabel(dataset_info['metric'], fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{dataset_name.upper()} Dataset', fontsize=14)
    
    # Create second y-axis for F1
    color2 = 'tab:red'
    ax2 = ax1.twinx()
    ax2.semilogx(beta_values, f1_values, 's-', color=color2, linewidth=2, markersize=8, label=dataset_info['f1_metric'])
    ax2.set_ylabel(dataset_info['f1_metric'], fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add value annotations
    for i, (x, y1, y2) in enumerate(zip(beta_values, ndcg_values, f1_values)):
        ax1.annotate(f'{y1:.4f}', (x, y1), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8, color=color1)
        ax2.annotate(f'{y2:.4f}', (x, y2), textcoords="offset points", 
                     xytext=(0,-15), ha='center', fontsize=8, color=color2)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig('beta_sensitivity_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('beta_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()