import matplotlib.pyplot as plt
import numpy as np

def plot_observed_against_expected(results):

    # First, calculate the overall min and max for the differences
    all_differences = []
    for values in results.values():
        observed = np.array(values['observed'])
        expected = np.array(values['expected'])
        difference = observed - expected
        all_differences.extend(difference)

    overall_min = min(all_differences)
    overall_max = max(all_differences)

    # Create the plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Histograms of Observed - Expected Values', fontsize=16)

    for i, (_prediction_time, values) in enumerate(results.items()):
        observed = np.array(values['observed'])
        expected = np.array(values['expected'])
        difference = observed - expected
        
        axes[i].hist(difference, bins=20, edgecolor='black', alpha=0.7)  # Added alpha for translucency
        axes[i].axvline(x=0, color='r', linestyle='--', linewidth=1)  # Added vertical line at x=0
        axes[i].set_title(_prediction_time)
        axes[i].set_xlabel('Observed - Expected')
        axes[i].set_ylabel('Frequency')
        
        # Set consistent x-axis limits
        axes[i].set_xlim(overall_min, overall_max)

    plt.tight_layout()
    plt.show()
