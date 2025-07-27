#!/usr/bin/env python
"""
Script to plot normalized metric values across different models.
Reads evaluation_results.csv and creates a grouped bar chart.
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

def create_bar_chart(df_normalized, metrics, output_path='metric_comparison.png'):
    """Create a grouped bar chart similar to the reference image."""
    
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for plotting
    models = df_normalized['model'].values
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Set up bar positions
    bar_width = 0.15  # Fixed bar width for better spacing
    indices = np.arange(n_metrics)
    
    # Define colors for each model - using colors similar to reference
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = color_palette[:n_models]
    
    # Plot bars for each model
    for i, model in enumerate(models):
        positions = indices + i * bar_width - (n_models - 1) * bar_width / 2
        values = [df_normalized.loc[df_normalized['model'] == model, metric].values[0] 
                 for metric in metrics]
        
        ax.bar(positions, values, bar_width, label=model, color=colors[i])
    
    # Customize the plot
    ax.set_xlabel('Metric', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized Metric Value', fontsize=16, fontweight='bold')
    ax.set_title('Controlling One Metric Does Not Control Others (Under Unmeasured Confounding)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis labels
    metric_labels = {
        'ECE': 'Calibration Error (ECE)',
        'BMSE': 'Balancing Error (BMSE)', 
        'DR_Bias': 'DR Bias',
        'DR_Variance': 'DR Variance'
    }
    ax.set_xticks(indices)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Customize y-axis
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend with better styling
    legend = ax.legend(title='Penalty', bbox_to_anchor=(1.02, 1), loc='upper left', 
                      frameon=True, fontsize=12)
    legend.get_title().set_fontsize(14)
    legend.get_title().set_fontweight('bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='Plot normalized metrics from evaluation results')
    parser.add_argument('--input', type=str, default='evaluation_results.csv',
                       help='Path to evaluation results CSV file')
    parser.add_argument('--output', type=str, default='metric_comparison.png',
                       help='Output path for the plot')
    parser.add_argument('--metrics', nargs='+', 
                       default=['ECE', 'BMSE', 'DR_Bias', 'DR_Variance'],
                       help='Metrics to include in the plot')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Found {len(df)} models: {', '.join(df['model'].values)}")
    
    # Check if all requested metrics exist
    missing_metrics = [m for m in args.metrics if m not in df.columns]
    if missing_metrics:
        print(f"Error: Metrics {missing_metrics} not found in the data!")
        print(f"Available metrics: {list(df.columns)}")
        return
    
    # Normalize the data
    print("Applying min-max normalization...")
    df_normalized = minmax_normalize(df, args.metrics)
    
    # Print normalization results
    print("\nNormalized values:")
    for metric in args.metrics:
        print(f"\n{metric}:")
        for idx, model in enumerate(df['model']):
            original = df.loc[idx, metric]
            normalized = df_normalized.loc[idx, metric]
            print(f"  {model}: {original:.6f} -> {normalized:.3f}")
    
    # Create the plot
    print(f"\nCreating bar chart...")
    fig, ax = create_bar_chart(df_normalized, args.metrics, args.output)
    
    print(f"Plot saved to {args.output}")
    print(f"PDF version saved to {args.output.replace('.png', '.pdf')}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()