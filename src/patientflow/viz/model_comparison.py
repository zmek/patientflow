import matplotlib.pyplot as plt
import numpy as np
import json
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
    dict: Nested dictionary containing results for each model and time
    """
    # Sort prediction times chronologically
    sorted_times = sorted(prediction_times, key=lambda x: x[0] * 60 + x[1])

    results = {}
    for model_name, metadata_file in models.items():
        # Load metadata
        full_path = model_file_path / "model-output" / metadata_file
        with open(full_path, "r") as f:
            model_metadata = json.load(f)

        # Initialize model results
        results[model_name] = {}

        # Extract results for each time
        for time in sorted_times:
            model_key = get_model_name(model_name, time)
            test_results = model_metadata[model_key]["test_set_results"]

            # Store results using model_key directly
            results[model_name][model_key] = {
                "test_auc": test_results["test_auc"],
                "test_auprc": test_results["test_auprc"],
                "test_logloss": test_results["test_logloss"],
            }

    return results


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
    # Load results
    results = load_model_results(model_file_path, prediction_times, models)

    # Get sorted time tuples
    sorted_times = sorted(prediction_times, key=lambda x: x[0] * 60 + x[1])

    # Create consistent keys for all models
    display_keys = [
        get_model_name(next(iter(models.keys())), time) for time in sorted_times
    ]

    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    # Set width of bars and positions of the bars
    width = 0.25
    x = np.arange(len(display_keys))

    # Calculate bar positions
    positions = np.linspace(
        -width * (len(models) - 1) / 2, width * (len(models) - 1) / 2, len(models)
    )

    # Plot metrics for each model
    metrics = ["test_auc", "test_auprc", "test_logloss"]
    axes = [ax1, ax2, ax3]
    titles = ["AUC", "AUPRC", "Log Loss"]

    bars = []
    labels = []
    for ax, metric, title in zip(axes, metrics, titles):
        for model_name, pos in zip(models.keys(), positions):
            # Generate the correct key for this model and time
            model_keys = [get_model_name(model_name, time) for time in sorted_times]
            metric_values = [results[model_name][key][metric] for key in model_keys]

            label = " ".join(model_name.split("_")).title()
            bar = ax.bar(x + pos, metric_values, width)
            if ax == ax1:  # Only store first set of bars/labels for legend
                bars.append(bar)
                labels.append(label)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([key.split("_")[-1] for key in display_keys])

    # Add single legend outside plots
    fig.legend(bars, labels, loc="center", bbox_to_anchor=(0.5, 0), ncol=len(models))

    fig.suptitle("Model Performance Comparison")

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig


def print_model_results(results):
    """
    Print model results in a specific format.

    Parameters:
    results (dict): Nested dictionary containing results for each model and time
    """
    for model_name, model_results in results.items():
        print(f"\nResults for {model_name} model")
        for model_key, metrics in model_results.items():
            print(
                f"\nModel: {model_key}; "
                f"AUC: {metrics['test_auc']:.3f}; "
                f"AUPRC: {metrics['test_auprc']:.3f}; "
                f"log loss {metrics['test_logloss']:.3f}",
                end=" ",
            )
        print("\n")
