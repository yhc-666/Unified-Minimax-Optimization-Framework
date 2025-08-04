#!/usr/bin/env python
"""
Script to plot normalized metric values across different models for multiple p values.
Creates a combined figure with three subplots side by side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_data(csv_path):
    """Load evaluation results from CSV."""
    df = pd.read_csv(csv_path)
    return df

def minmax_normalize(df, metrics):
    """Apply min-max normalization to each metric across all models."""
    normalized_data = {}
    
    for metric in metrics:
        values = df[metric].values
        min_val = values.min()
        max_val = values.max()
        
        # Handle edge case where all values are the same
        if max_val - min_val < 1e-10:
            normalized_values = np.ones_like(values) * 0.5  # Set to middle value
        else:
            normalized_values = (values - min_val) / (max_val - min_val)
        
        normalized_data[metric] = normalized_values
    
    # Add model names
    normalized_data['model'] = df['model'].values
    
    return pd.DataFrame(normalized_data)

def create_combined_chart(p_values=[0.4, 0.5, 0.6], metrics=['ECE', 'BMSE', 'DR_Bias', 'DR_Variance'], 
                         output_path='metric_comparison_combined.png'):
    """Create a combined figure with three subplots for different p values."""
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(28, 6), sharey=True)
    
    # Define colors for each model - using colors similar to reference
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Metric labels
    metric_labels = {
        'ECE': 'Calibration Error',
        'BMSE': 'Balancing Error', 
        'DR_Bias': 'DR Bias',
        'DR_Variance': 'DR Variance'
    }
    
    # Process each p value
    for idx, (ax, p_val) in enumerate(zip(axes, p_values)):
        # Load data for this p value
        csv_path = f'semi-synthetic/outputs/evaluation_results_p{p_val}.csv'
        
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_path} not found!")
            continue
            
        df = load_data(csv_path)
        df_normalized = minmax_normalize(df, metrics)
        
        # Prepare data for plotting
        models = df_normalized['model'].values
        n_models = len(models)
        n_metrics = len(metrics)
        
        # Set up bar positions
        bar_width = 0.13  # Slightly narrower bars for less crowding
        indices = np.arange(n_metrics)
        
        # Store bar containers for legend
        bar_containers = []
        
        # Plot bars for each model
        for i, model in enumerate(models):
            positions = indices + i * bar_width - (n_models - 1) * bar_width / 2
            values = [df_normalized.loc[df_normalized['model'] == model, metric].values[0] 
                     for metric in metrics]
            
            # Add minimum bar height for visibility (2% of y-axis range)
            min_bar_height = 0.02
            values_with_min = [max(v, min_bar_height) for v in values]
            
            bars = ax.bar(positions, values_with_min, bar_width, label=model, color=color_palette[i])
            
            # Add value labels on top of bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                # Only show label if value is meaningful (> 0.01)
                if value > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
            
            # Store bars from first subplot for legend
            if idx == 0:
                bar_containers.append(bars)
        
        # Customize each subplot
        ax.set_xlabel('Metric', fontsize=16, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Normalized Metric Value', fontsize=16, fontweight='bold')
        
        # Set subplot title with p value
        ax.set_title(f'p = {p_val}', fontsize=18, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(indices)
        ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], fontsize=14, fontweight='bold')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        
        # Customize y-axis
        ax.tick_params(axis='y', labelsize=12)
        # Make y-axis labels bold
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add main title
    fig.suptitle('Controlling One Metric Does Not Control Others', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Create unified legend at the bottom
    if bar_containers:
        models = df_normalized['model'].values
        # Create legend at the very bottom of the figure
        legend = fig.legend(bar_containers, models, loc='upper center', ncol=len(models), 
                  bbox_to_anchor=(0.5, 0.08), fontsize=14, prop={'weight': 'bold'},
                  frameon=True, title='Models', fancybox=True, shadow=True)
        legend.get_title().set_fontsize(16)
        legend.get_title().set_fontweight('bold')
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.2, top=0.88, wspace=0.15)
    
    # Save the figure without tight bbox to preserve legend
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot combined normalized metrics from evaluation results')
    parser.add_argument('--p-values', nargs='+', type=float, default=[0.4, 0.5, 0.6],
                       help='List of p values to include')
    parser.add_argument('--output', type=str, default='semi-synthetic/outputs/metric_comparison_combined.png',
                       help='Output path for the plot')
    parser.add_argument('--metrics', nargs='+', 
                       default=['ECE', 'BMSE', 'DR_Bias', 'DR_Variance'],
                       help='Metrics to include in the plot')
    
    args = parser.parse_args()
    
    # Create the combined plot
    print(f"Creating combined bar chart for p values: {args.p_values}")
    fig = create_combined_chart(p_values=args.p_values, metrics=args.metrics, output_path=args.output)
    
    print(f"Plot saved to {args.output}")
    print(f"PDF version saved to {args.output.replace('.png', '.pdf')}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()