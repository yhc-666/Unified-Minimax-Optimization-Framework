import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('minimax_evaluation_results.csv')

# Sort by num_bins to ensure proper line plotting
df = df.sort_values('num_bins')

# Manual control for x-axis range (set to None to use all data)
# Example: x_min = 5, x_max = 30 will only show bins from 5 to 30
x_min = 1  # Minimum bin number to display
x_max = 45  # Maximum bin number to display

# Filter data based on x-axis range if specified
if x_min is not None or x_max is not None:
    if x_min is not None:
        df = df[df['num_bins'] >= x_min]
    if x_max is not None:
        df = df[df['num_bins'] <= x_max]

# Create figure with 5 subplots (3x2 grid, with last subplot empty)
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
fig.suptitle('Evaluation Metrics vs Number of Bins', fontsize=16)

# Flatten axes for easier iteration
axes = axes.flatten()

# Function to calculate appropriate y-axis limits
def calculate_ylim(data, padding_percent=10):
    """Calculate y-axis limits with padding to show variations clearly"""
    min_val = data.min()
    max_val = data.max()
    range_val = max_val - min_val
    
    # If range is very small, use a minimum padding
    if range_val < 1e-10:
        padding = max(abs(min_val) * 0.01, 1e-10)
    else:
        padding = range_val * (padding_percent / 100)
    
    return (min_val - padding, max_val + padding)

# Define metrics and calculate their y-axis limits dynamically
metrics = [
    {'name': 'ECE', 'column': 'ECE', 'ylabel': 'ECE'},
    {'name': 'BMSE', 'column': 'BMSE', 'ylabel': 'BMSE'},
    {'name': 'DR_Bias', 'column': 'DR_Bias', 'ylabel': 'DR Bias'},
    {'name': 'DR_Variance', 'column': 'DR_Variance', 'ylabel': 'DR Variance'},
    {'name': 'AUC', 'column': 'AUC', 'ylabel': 'AUC'}
]

# Calculate y-axis limits for each metric
for metric in metrics:
    metric['ylim'] = calculate_ylim(df[metric['column']], padding_percent=15)

# Plot each metric
for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Plot the line
    ax.plot(df['num_bins'], df[metric['column']], 
            marker='o', linewidth=2, markersize=8, color='blue')
    
    # Set labels and title
    ax.set_xlabel('Number of Bins', fontsize=12)
    ax.set_ylabel(metric['ylabel'], fontsize=12)
    ax.set_title(f'{metric["name"]} vs Number of Bins', fontsize=14)
    
    # Set y-axis limits for better visibility of changes
    ax.set_ylim(metric['ylim'])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis for scientific notation if needed
    if metric['name'] == 'DR_Variance':
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Rotate x-axis labels if needed
    ax.set_xticks(df['num_bins'])
    ax.set_xticklabels(df['num_bins'].astype(int))

# Hide the empty subplot (6th subplot)
axes[-1].set_visible(False)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the figure
plt.savefig('bin_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig('bin_evaluation_metrics.pdf', bbox_inches='tight')

# Show the plot
plt.show()

# Print summary statistics
print("Summary Statistics:")
print("-" * 50)
for metric in metrics:
    col = metric['column']
    print(f"\n{metric['name']}:")
    print(f"  Min: {df[col].min():.6f} (at {df.loc[df[col].idxmin(), 'num_bins']} bins)")
    print(f"  Max: {df[col].max():.6f} (at {df.loc[df[col].idxmax(), 'num_bins']} bins)")
    print(f"  Range: {df[col].max() - df[col].min():.6f}")
    print(f"  Std: {df[col].std():.6f}")