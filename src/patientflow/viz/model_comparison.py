import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from patientflow.load import get_model_name

def load_model_results(model_file_path, prediction_times, models):
    """
    Load model results from metadata files.
    
    Parameters:
    model_file_path (Path): Path to model directory
    prediction_times (list): List of prediction times as tuples (hour, minute)
    models (dict): Dictionary mapping model names to their metadata filenames
                  e.g. {'admissions_minimal': 'minimal_model_metadata.json'}
    
    Returns:
    dict: Dictionary containing results for each model
    """
    # Sort prediction times chronologically
    prediction_times = sorted(prediction_times, key=lambda x: x[0] * 60 + x[1])
    
    results = {}
    for model_name, metadata_file in models.items():
        # Load metadata
        full_path = model_file_path / "model-output" / metadata_file
        with open(full_path, 'r') as f:
            model_metadata = json.load(f)
            
        # Extract results for each time
        auc_values = []
        auprc_values = []
        logloss_values = []
        
        for time in prediction_times:
            model_key = get_model_name(model_name, time)
            test_results = model_metadata[model_key]['test_set_results']
            
            auc_values.append(test_results['test_auc'])
            auprc_values.append(test_results['test_auprc'])
            logloss_values.append(test_results['test_logloss'])
        
        # Store results
        results[model_name] = {
            'test_auc': auc_values,
            'test_auprc': auprc_values,
            'test_logloss': logloss_values
        }
    
    return results, prediction_times

def plot_model_comparisons(model_file_path, prediction_times, models, figsize=(8, 6)):
    """
    Load data and plot model comparison metrics.
    
    Parameters:
    model_file_path (Path): Path to model directory
    prediction_times (list): List of prediction times as tuples (hour, minute)
    models (dict): Dictionary mapping model names to their metadata filenames
                  e.g. {'admissions_minimal': 'minimal_model_metadata.json'}
    figsize (tuple): Figure size in inches (width, height)
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    # Load results with sorted times
    results, sorted_times = load_model_results(model_file_path, prediction_times, models)
    
    # Format times for display
    display_times = [f"{hour:02d}:{minute:02d}" for hour, minute in sorted_times]
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Set width of bars and positions of the bars
    width = 0.25
    x = np.arange(len(display_times))
    
    # Calculate bar positions
    positions = np.linspace(-width * (len(models)-1)/2, 
                          width * (len(models)-1)/2, 
                          len(models))
    
    # Plot metrics for each model
    metrics = ['test_auc', 'test_auprc', 'test_logloss']
    axes = [ax1, ax2, ax3]
    titles = ['AUC', 'AUPRC', 'Log Loss']
    
    bars = []
    labels = []
    for ax, metric, title in zip(axes, metrics, titles):
        for model_name, pos in zip(models.keys(), positions):
            label = ' '.join(model_name.split('_')).title()
            bar = ax.bar(x + pos, results[model_name][metric], width)
            if ax == ax1:  # Only store first set of bars/labels for legend
                bars.append(bar)
                labels.append(label)
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(display_times)
    
    # Add single legend outside plots
    fig.legend(bars, labels, loc='center', bbox_to_anchor=(0.5, 0), ncol=len(models))

    fig.suptitle('Model Performance Comparison')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig