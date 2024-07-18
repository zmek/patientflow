from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_madcap(predict_proba, label, dataset, media_path=None):
    """
    Save a MADCAP plot comparing predicted probabilities and actual outcomes.

    Parameters
    predict_proba (array-like): Array of predicted probabilities.
    label (array-like): Array of actual labels.
    dataset (str): Name of the dataset, used in the plot title and file name.
    media_path (Path): Media path to save the plot.

    """
    # Ensure inputs are numpy arrays
    predict_proba = np.array(predict_proba)
    label = np.array(label)

    # Sort by predict_proba
    sorted_indices = np.argsort(predict_proba)
    sorted_proba = predict_proba[sorted_indices]
    sorted_label = label[sorted_indices]

    # Compute unique probabilities and their mean labels
    unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
    mean_labels = np.zeros_like(unique_probs)

    np.add.at(mean_labels, inverse_indices, sorted_label)
    counts = np.bincount(inverse_indices)
    mean_labels = mean_labels / counts

    # Cumulative sums for model and observed
    model = np.cumsum(sorted_proba)
    observed = np.cumsum(mean_labels[inverse_indices])

    x = np.arange(len(sorted_proba))

    fig, ax = plt.subplots(2, figsize=(4, 8))

    # Plot perfectly calibrated
    ax[0].plot(x, model, label="model")
    ax[0].plot(x, observed, label="observed")
    ax[0].legend(loc="upper left")
    ax[0].set_xlabel("Test set visits ordered by increasing predicted probability")
    ax[0].set_ylabel("Number of admissions")
    ax[0].set_title("Expected number of admissions compared with MADCAP")

    # Plot difference
    ax[1].plot(x, model - observed)
    ax[1].legend(loc="upper left")
    ax[1].set_xlabel("Test set visits ordered by increasing predicted probability")
    ax[1].set_ylabel("Expected number of admissions - observed")
    ax[1].set_title("Difference between expected and observed")

    fig.tight_layout(pad=1.08, rect=[0, 0, 1, 0.97])

    fig.suptitle("MADCAP plot: " + dataset)

    if media_path:
        plot_name = "madcap_plot_" + dataset + ".png"
        madcap_plot_path = Path(media_path) / plot_name
        plt.savefig(madcap_plot_path)

    plt.show()
    plt.close(fig)


def plot_madcap_by_group(
    predict_proba, label, group, dataset, group_name, media_path=None
):
    """
    Save MADCAP plots subdivided by a specified grouping variable.

    Parameters
    predict_proba (array-like): Array of predicted probabilities.
    label (array-like): Array of actual labels.
    group (array-like): Array of grouping variable values.
    dataset (str): Name of the dataset, used in the plot title and file name.
    group_name (str): Name of the grouping variable
    media_path (Path): Media path to save the plot.

    """
    # Ensure inputs are numpy arrays
    predict_proba = np.array(predict_proba)
    label = np.array(label)
    group = np.array(group)

    unique_groups = np.unique(group)
    fig, ax = plt.subplots(2, len(unique_groups), figsize=(12, 8))

    for i, grp in enumerate(unique_groups):
        mask = group == grp
        sorted_indices = np.argsort(predict_proba[mask])
        sorted_proba = predict_proba[mask][sorted_indices]
        sorted_label = label[mask][sorted_indices]

        # Compute unique probabilities and their mean labels
        unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
        mean_labels = np.zeros_like(unique_probs)

        np.add.at(mean_labels, inverse_indices, sorted_label)
        counts = np.bincount(inverse_indices)
        mean_labels = mean_labels / counts

        # Cumulative sums for model and observed
        model = np.cumsum(sorted_proba)
        observed = np.cumsum(mean_labels[inverse_indices])

        x = np.arange(len(sorted_proba))

        # Plot perfectly calibrated
        ax[0, i].plot(x, model, label="model")
        ax[0, i].plot(x, observed, label="observed")
        ax[0, i].legend(loc="upper left")
        ax[0, i].set_xlabel("Test set visits ordered by predicted probability")
        ax[0, i].set_ylabel("Number of admissions")
        ax[0, i].set_title(f"{group_name}: {grp!s}")

        # Plot difference
        ax[1, i].plot(x, model - observed)
        ax[1, i].set_xlabel("Test set visits ordered by predicted probability")
        ax[1, i].set_ylabel("Expected number of admissions - observed")
        ax[1, i].set_title(f"{group_name}: {grp!s}")

    fig.suptitle(f"MADCAP plots by {group_name}: {dataset}")
    fig.tight_layout(pad=1.08, rect=[0, 0, 1, 0.97])

    if media_path:
        plot_name = f"madcap_plot_by_{group_name}_{dataset}.png"
        madcap_plot_path = Path(media_path) / plot_name
        plt.savefig(madcap_plot_path)
    plt.show()

    # Close the plot to free memory
    plt.close(fig)
