import matplotlib.pyplot as plt
import numpy as np
import math

def plot_observed_against_expected(
    results,
    title="Histograms of Observed - Expected Values",
    xlabel="Observed - Expected"
):
    # Calculate the number of subplots needed
    num_plots = len(results)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_plots)  # Maximum of 5 columns
    num_rows = math.ceil(num_plots / num_cols)
    
    # Set a minimum width for the figure
    min_width = 8  # minimum width in inches
    width = max(min_width, 4 * num_cols)
    height = 4 * num_rows
    
    # Create the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    # Flatten the axes array
    axes = axes.flatten()
    
    for i, (_prediction_time, values) in enumerate(results.items()):
        observed = np.array(values['observed'])
        expected = np.array(values['expected'])
        difference = observed - expected
        
        ax = axes[i]
        
        # Use integer bins
        bins = np.arange(math.floor(min(difference)), math.ceil(max(difference)) + 2) - 0.5
        
        ax.hist(difference, bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax.set_title(_prediction_time)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        
        # Ensure x-axis ticks are at integer values
        # ax.set_xticks(np.arange(math.floor(min(difference)), math.ceil(max(difference)) + 1))
    
    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()