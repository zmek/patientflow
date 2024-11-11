import matplotlib.pyplot as plt
import numpy as np
import math

def plot_observed_against_expected(
    results1,
    results2=None,
    title1=None,
    title2=None,
    main_title="Histograms of Observed - Expected Values",
    xlabel="Observed minus expected number of admissions"
):
    # Calculate the number of subplots needed
    num_plots = len(results1)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(5, num_plots)  # Maximum of 5 columns
    num_rows = math.ceil(num_plots / num_cols)
    
    if results2:
        num_rows *= 2  # Double the number of rows if we have two result sets
    
    # Set a minimum width for the figure
    min_width = 8  # minimum width in inches
    width = max(min_width, 4 * num_cols)
    height = 4 * num_rows
    
    # Create the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height), squeeze=False)
    fig.suptitle(main_title, fontsize=14)
    
    # Flatten the axes array
    axes = axes.flatten()
    
    def format_prediction_time(prediction_time):
        # Split the string by underscores and take the last element
        last_part = prediction_time.split('_')[-1]
        # Add a colon in the middle
        return f"{last_part[:2]}:{last_part[2:]}"

    def plot_results(results, start_index, result_title):
        for i, (_prediction_time, values) in enumerate(results.items()):
            observed = np.array(values['observed'])
            expected = np.array(values['expected'])
            difference = observed - expected
            
            ax = axes[start_index + i]
            
            # Use integer bins
            bins = np.arange(math.floor(min(difference)), math.ceil(max(difference)) + 2) - 0.5
            
            ax.hist(difference, bins=bins, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
            
            # Format the prediction time
            formatted_time = format_prediction_time(_prediction_time)
            
            # Combine the result_title and formatted_time
            if result_title:
                ax.set_title(f"{result_title} {formatted_time}")
            else:
                ax.set_title(formatted_time)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Frequency')
            
            # Store the x-limits for later synchronization
            ax.x_limits = ax.get_xlim()
    
    # Plot the first results set
    plot_results(results1, 0, title1)
    
    # Plot the second results set if provided
    if results2:
        plot_results(results2, num_plots, title2)
    
    # Synchronize x-axis limits for corresponding plots
    for i in range(num_plots):
        if results2:
            x_min = min(axes[i].x_limits[0], axes[i + num_plots].x_limits[0])
            x_max = max(axes[i].x_limits[1], axes[i + num_plots].x_limits[1])
            axes[i].set_xlim(x_min, x_max)
            axes[i + num_plots].set_xlim(x_min, x_max)
        else:
            axes[i].set_xlim(axes[i].x_limits)
    
    # Hide any unused subplots
    for j in range(num_plots * (2 if results2 else 1), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()