"""
Module for generating MADCAP (Model Accuracy and Discriminative Calibration Plots) visualizations.

This module provides functions for creating MADCAP plots, which compare model-predicted probabilities to observed outcomes.
MADCAP plots can be generated for individual prediction times or for specific groups (e.g., age groups).

Functions
---------
classify_age_group(age_group)
    Classifies patients into age categories based on predefined age group ranges.

classify_age(age_on_arrival)
    Classifies patients based on their age on arrival, categorizing them as 'children', 'adults', or '65 or over'.

generate_madcap_plots(prediction_times, model_file_path, media_file_path, visits_csv_path)
    Generates MADCAP plots for a series of prediction times, plotting the model vs. observed cumulative admissions.

plot_madcap_subplot(predict_proba, label, _prediction_time, ax)
    Helper function to plot a single MADCAP subplot for a given prediction time.

plot_madcap_by_group(predict_proba, label, group, _prediction_time, group_name, media_path=None, plot_difference=True)
    Generates MADCAP plots for subgroups (e.g., age groups) at a specific prediction time.

generate_madcap_plots_by_group(prediction_times, model_file_path, media_file_path, visits_csv_path, grouping_var, grouping_var_name)
    Generates MADCAP plots for groups (e.g., age groups) across a series of prediction times.
"""

from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import math
import numpy as np
from patientflow.prepare import prepare_for_inference

exclude_from_training_data = [
    "visit_number",
    "snapshot_date",
    "prediction_time",
    "specialty",
    "consultation_sequence",
    "final_sequence",
]


def classify_age(age):
    """
    Classifies age into categories, either based on a direct age value or an age group string.

    Parameters
    ----------
    age : int, float, or str
        Age value (e.g., 30) or age group string (e.g., '18-24').

    Returns
    -------
    str
        'children' if the age is less than 18 or the age group is '0-17',
        'adults' if the age is between 18 and 64 or the age group is between '18-64',
        '65 or over' if the age is 65 or above or the age group is '65-102',
        'unknown' for unexpected or invalid values.
    """
    if isinstance(age, (int, float)):
        if age < 18:
            return "children"
        elif age < 65:
            return "adults"
        elif age >= 65:
            return "65 or over"
        else:
            return "unknown"
    elif isinstance(age, str):
        if age in ["0-17"]:
            return "children"
        elif age in ["18-24", "25-34", "35-44", "45-54", "55-64"]:
            return "adults"
        elif age in ["65-74", "75-102"]:
            return "65 or over"
        else:
            return "unknown"
    else:
        return "unknown"


def generate_madcap_plots(
    prediction_times: List[Tuple[int, int]],
    model_file_path: Union[str, Path],
    media_file_path: Union[str, Path, None],
    visits_csv_path: Union[str, Path],
) -> None:
    """
    Generates MADCAP plots for a list of prediction times, comparing predicted probabilities
    to actual admissions.

    Parameters
    ----------
    prediction_times : list of tuple
        List of prediction times as (hour, minute) tuples.
    model_file_path : str or Path
        Path to the trained model file.
    media_file_path : str or Path
        Directory path where the generated plots will be saved.
    visits_csv_path : str or Path
        Path to the CSV file containing visit data.
    """

    num_plots = len(prediction_times)

    # Calculate the number of rows and columns for the subplots
    num_cols = min(num_plots, 5)  # Maximum 5 columns
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_plots * 5, 4))

    # Ensure axes is always a 2D array
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, _prediction_time in enumerate(prediction_times):
        X_test, y_test, pipeline = prepare_for_inference(
            model_file_path,
            "admissions",
            prediction_time=_prediction_time,
            data_path=visits_csv_path,
            single_snapshot_per_visit=False,
        )

        predict_proba = pipeline.predict_proba(X_test)[:, 1]

        row = i // num_cols
        col = i % num_cols
        plot_madcap_subplot(predict_proba, y_test, _prediction_time, axes[row, col])

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if media_file_path:
        plot_name = "madcap_plot"
        madcap_plot_path = Path(media_file_path) / plot_name
        plt.savefig(madcap_plot_path)

    plt.show()
    plt.close(fig)


def plot_madcap_subplot(predict_proba, label, _prediction_time, ax):
    """
    Plots a single MADCAP subplot showing cumulative predicted and observed admissions.

    Parameters
    ----------
    predict_proba : array-like
        Array of predicted probabilities.
    label : array-like
        Array of true labels (admissions).
    _prediction_time : tuple
        Prediction time as (hour, minute).
    ax : matplotlib.axes.Axes
        The axis on which the subplot will be drawn.
    """
    hour, minutes = _prediction_time
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

    # Plot
    ax.plot(x, model, label="model")
    ax.plot(x, observed, label="observed")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_xlabel("Test set visits ordered by predicted probability", fontsize=12)
    ax.set_ylabel("Number of admissions", fontsize=12)
    ax.set_title(f"MADCAP Plot for {hour}:{minutes:02}", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize="x-small")


def plot_madcap_by_group(
    predict_proba,
    label,
    group,
    _prediction_time,
    group_name,
    media_path=None,
    plot_difference=True,
):
    """
    Generates MADCAP plots for specific groups (e.g., age groups) at a given prediction time.

    Parameters
    ----------
    predict_proba : array-like
        Array of predicted probabilities.
    label : array-like
        Array of true labels (admissions).
    group : array-like
        Array of group labels for each visit (e.g., age group).
    _prediction_time : tuple
        Prediction time as (hour, minute).
    group_name : str
        Name of the group variable being plotted (e.g., 'Age Group').
    media_path : str or Path, optional
        Path to save the generated plot, if specified.
    plot_difference : bool, optional
        If True, includes an additional plot showing the difference between predicted and observed admissions.
    """
    # Remove those with unknown age
    mask_known = group != "unknown"
    predict_proba = predict_proba[mask_known]
    label = label[mask_known]
    group = group[mask_known]

    hour, minutes = _prediction_time

    predict_proba, label, group = map(np.array, (predict_proba, label, group))
    unique_groups = [grp for grp in np.unique(group) if grp != "unknown"]

    fig_size = (10, 8) if plot_difference else (9, 3)
    fig, ax = plt.subplots(
        2 if plot_difference else 1, len(unique_groups), figsize=fig_size
    )
    ax = ax.reshape(-1, len(unique_groups)) if plot_difference else ax.reshape(1, -1)

    for i, grp in enumerate(unique_groups):
        mask = group == grp
        sorted_indices = np.argsort(predict_proba[mask])
        sorted_proba = predict_proba[mask][sorted_indices]
        sorted_label = label[mask][sorted_indices]

        unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
        mean_labels = np.bincount(inverse_indices, weights=sorted_label) / np.bincount(
            inverse_indices
        )

        model = np.cumsum(sorted_proba)
        observed = np.cumsum(mean_labels[inverse_indices])
        x = np.arange(len(sorted_proba))

        ax[0, i].plot(x, model, label="model")
        ax[0, i].plot(x, observed, label="observed")
        ax[0, i].legend(loc="upper left", fontsize=8)
        ax[0, i].set_xlabel(
            "Test set visits ordered by predicted probability", fontsize=8
        )
        ax[0, i].set_ylabel("Number of admissions", fontsize=8)
        ax[0, i].set_title(f"{group_name}: {grp!s}", fontsize=8)
        ax[0, i].tick_params(axis="both", which="major", labelsize=8)

        if plot_difference:
            ax[1, i].plot(x, model - observed)
            ax[1, i].set_xlabel(
                "Test set visits ordered by predicted probability", fontsize=8
            )
            ax[1, i].set_ylabel("Expected number of admissions - observed", fontsize=8)
            ax[1, i].set_title(f"{group_name}: {grp!s}", fontsize=8)
            ax[1, i].tick_params(axis="both", which="major", labelsize=8)

        # Adjust layout first
    fig.tight_layout(pad=1.08)

    # Then add super title
    fig.suptitle(
        f"MADCAP Plots by {group_name} for {hour}:{minutes:02}", fontsize=10, y=1.04
    )

    # Fine-tune the layout
    fig.subplots_adjust(top=0.90)

    # fig.tight_layout(pad=1.08, rect=[0, 0.03, 1, 0.95])

    if media_path:
        plot_name = (
            f"madcap_plot_by_{group_name.replace(' ', '_')}_{hour}{minutes:02}.png"
        )
        madcap_plot_path = Path(media_path) / plot_name
        plt.savefig(madcap_plot_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_madcap_plots_by_group(
    prediction_times: List[Tuple[int, int]],
    model_file_path: Union[str, Path],
    media_file_path: Union[str, Path, None],
    visits_csv_path: Union[str, Path],
    grouping_var: str,
    grouping_var_name: str,
    plot_difference: bool = False,
) -> None:
    """
    Generates MADCAP plots for different groups (e.g., age groups) across multiple prediction times.

    This function creates MADCAP (Model Accuracy and Discriminative Calibration Plot) visualizations,
    comparing predicted probabilities from a trained model to observed outcomes. The plots are generated
    for specific groups (such as age groups) over a series of prediction times. The grouping variable
    (e.g., 'age_group', 'age_on_arrival', or other) is specified and must exist in the dataset.

    Parameters
    ----------
    prediction_times : list of tuple
        A list of prediction times, each specified as a tuple of (hour, minute).
    model_file_path : str or Path
        Path to the trained machine learning model file that will be used for inference.
    media_file_path : str or Path or None
        Directory path where the generated plots will be saved. If None, the plots are not saved.
    visits_csv_path : str or Path
        Path to the CSV file containing visit data used to prepare data for inference.
    grouping_var : str
        The column name in the dataset that defines the grouping variable (e.g., 'age_group', 'age_on_arrival').
        This variable must exist in the `X_test` columns.
    grouping_var_name : str
        A descriptive name for the grouping variable, used in plot titles (e.g., 'Age Group').
    plot_difference : bool, optional
        If True, includes an additional plot showing the difference between predicted and observed admissions.
        Default is False.

    Raises
    ------
    ValueError
        If `grouping_var` is not found in the columns of the test data (`X_test`).

    Notes
    -----
    The function first prepares the test data (X_test) and true labels (y_test) for each prediction time using
    the `prepare_for_inference` function. It then uses the trained model pipeline to compute predicted
    probabilities. Patients are classified into groups based on the specified `grouping_var`, such as 'age_group'
    or 'age_on_arrival'. If the `grouping_var` is not found in the dataset, a `ValueError` is raised.

    Finally, the function generates MADCAP plots for each group using the `plot_madcap_by_group` function, and
    optionally saves the plots to the specified `media_file_path`.

    Examples
    --------
    >>> generate_madcap_plots_by_group(
            prediction_times=[(12, 0), (14, 30)],
            model_file_path="path/to/model.pkl",
            media_file_path="path/to/media",
            visits_csv_path="path/to/visits.csv",
            grouping_var='age_group',
            grouping_var_name='Age Group'
        )
    """

    for i, _prediction_time in enumerate(prediction_times):
        X_test, y_test, pipeline = prepare_for_inference(
            model_file_path,
            "admissions",
            prediction_time=_prediction_time,
            data_path=visits_csv_path,
            single_snapshot_per_visit=False,
        )

        # Check if the grouping variable exists in X_test columns
        if grouping_var not in X_test.columns:
            raise ValueError(f"'{grouping_var}' not found in the dataset columns.")

        predict_proba = pipeline.predict_proba(X_test)[:, 1]

        # Apply classification based on the grouping variable
        if grouping_var == "age_group":
            group = X_test["age_group"].apply(classify_age)
        elif grouping_var == "age_on_arrival":
            group = X_test["age_on_arrival"].apply(classify_age)
        else:
            group = X_test[grouping_var]  # If it's another grouping variable

        plot_madcap_by_group(
            predict_proba,
            y_test,
            group,
            _prediction_time,
            grouping_var_name,
            media_file_path,
            plot_difference,
        )
