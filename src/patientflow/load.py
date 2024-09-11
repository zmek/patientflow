"""
This module provides functionality for loading configuration files, data from CSV files, and trained machine learning models.

It includes the following features:

- **Loading Configurations**: Parse YAML configuration files and extract necessary parameters for data processing and modeling.
- **Data Handling**: Load and preprocess data from CSV files, including optional operations like setting an index, sorting, and applying literal evaluation on columns.
- **Model Management**: Load saved machine learning models, customize model filenames based on time, and categorize DataFrame columns into predefined groups for analysis.

The module handles common file and parsing errors, returning appropriate error messages or exceptions.

Functions
---------
- `load_config_file`: Load a YAML configuration file and extract key parameters.
- `set_file_locations`: Set file locations based on UCLH-specific or default data sources.
- `safe_literal_eval`: Safely evaluate string literals into Python objects.
- `data_from_csv`: Load and preprocess data from a CSV file.
- `get_model_name`: Generate a model name based on the time of day.
- `load_saved_model`: Load a machine learning model saved in a joblib file.
- `get_dict_cols`: Categorize columns from a DataFrame into predefined groups for analysis.
"""


import ast  # to convert tuples to strings
import os
from pathlib import Path

import pandas as pd
from joblib import load
from errors import ModelLoadError

import yaml
from typing import Any, Dict, Tuple, Union, List, Optional


def load_config_file(
    config_file_path: str, return_start_end_dates: bool = False
) -> Optional[
    Union[
        Dict[str, Any],
        Tuple[str, str],
        Tuple[
            List[Tuple[int, int]],
            str,
            str,
            str,
            str,
            float,
            float,
            float,
            float,
            int,
            float,
            float,
        ],
    ]
]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file_path : str
        The path to the configuration file.
    return_start_end_dates : bool, optional
        If True, return the start and end dates from the file (default is False).

    Returns
    -------
    dict or tuple or None
        If `return_start_end_dates` is True, returns a tuple of start and end dates (str).
        Otherwise, returns a tuple containing prediction times, modelling dates, and other configuration values.
        Returns None if an error occurs during file reading or parsing.
    """
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

    try:
        if return_start_end_dates:
            if "file_dates" in config and config["file_dates"]:
                start_date, end_date = [str(item) for item in config["file_dates"]]
            else:
                print(
                    "Error: 'file_dates' key not found or empty in the configuration file."
                )
                return None

        if "prediction_times" in config:
            prediction_times = [tuple(item) for item in config["prediction_times"]]
        else:
            print("Error: 'prediction_times' key not found in the configuration file.")
            return None

        if "modelling_dates" in config and len(config["modelling_dates"]) == 4:
            start_training_set, start_validation_set, start_test_set, end_test_set = [
                item for item in config["modelling_dates"]
            ]
        else:
            print(
                "Error: expecting 4 modelling dates and only got "
                + str(len(config["modelling_dates"]))
            )
            return None

        x1 = float(config.get("x1", 4))
        y1 = float(config.get("y1", 0.76))
        x2 = float(config.get("x2", 12))
        y2 = float(config.get("y2", 0.99))
        prediction_window = config.get("prediction_window", 480)
        epsilon = config.get("epsilon", 10**-7)
        yta_time_interval = config.get("yta_time_interval", 15)

        if return_start_end_dates:
            return (start_date, end_date)
        else:
            return (
                prediction_times,
                start_training_set,
                start_validation_set,
                start_test_set,
                end_test_set,
                x1,
                y1,
                x2,
                y2,
                prediction_window,
                epsilon,
                yta_time_interval,
            )
    except KeyError as e:
        print(f"Error: Missing key in the configuration file: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid value found in the configuration file: {e}")
        return None


def set_file_locations(uclh: bool, data_path: Path, config_file_path: Optional[str] = None):
    """
    Set file locations based on UCLH or default data source.

    Parameters
    ----------
    uclh : bool
        If True, use UCLH-specific file locations. If False, use default file locations.
    data_path : Path
        The base path to the data directory.
    config_file_path : str, optional
        The path to the configuration file, required if `uclh` is True.

    Returns
    -------
    tuple
        Paths to the required files (visits, arrivals) based on the configuration.
    """
    if not uclh:
        csv_filename = "ed_visits.csv"
        yta_csv_filename = "arrivals.csv"

        visits_csv_path = data_path / csv_filename
        yta_csv_path = data_path / yta_csv_filename

        return visits_csv_path, yta_csv_path

    else:
        start_date, end_date = load_config_file(
            config_file_path, return_start_end_dates=True
        )
        data_filename = (
            "uclh_visits_exc_beds_inc_minority_"
            + str(start_date)
            + "_"
            + str(end_date)
            + ".pickle"
        )
        csv_filename = "uclh_visits.csv"
        yta_filename = (
            "uclh_yet_to_arrive_" + str(start_date) + "_" + str(end_date) + ".pickle"
        )
        yta_csv_filename = "uclh_arrivals.csv"

        visits_path = data_path / data_filename
        yta_path = data_path / yta_filename
        visits_csv_path = data_path / csv_filename
        yta_csv_path = data_path / yta_csv_filename

    return visits_path, visits_csv_path, yta_path, yta_csv_path


def safe_literal_eval(s: str) -> Optional[Any]:
    """
    Safely evaluate a string literal into a Python object.

    Parameters
    ----------
    s : str
        The string to evaluate.

    Returns
    -------
    Any or None
        The evaluated Python object if successful, otherwise None.
    """
    try:
        if pd.isna(s) or str(s).strip().lower() in ["nan", "none", ""]:
            return None
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def data_from_csv(
    csv_path: str, index_column: Optional[str] = None, sort_columns: Optional[List[str]] = None, eval_columns: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file and perform optional operations on it.

    Parameters
    ----------
    csv_path : str
        The path to the CSV data file.
    index_column : str, optional
        The column to set as the DataFrame index.
    sort_columns : list of str, optional
        Columns to sort the DataFrame by.
    eval_columns : list of str, optional
        Columns to apply `safe_literal_eval` to.

    Returns
    -------
    pd.DataFrame or None
        The DataFrame containing the data, or None if the file couldn't be loaded.
    """
    path = os.path.join(Path().home(), csv_path)

    try:
        df = pd.read_csv(path, parse_dates=True)
    except FileNotFoundError:
        print(f"Data file not found at path: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    if index_column:
        try:
            if df.index.name != index_column:
                df = df.set_index(index_column)
        except KeyError:
            print(f"Index column '{index_column}' not found in dataframe")

    if sort_columns:
        try:
            df.sort_values(sort_columns, inplace=True)
        except KeyError:
            print("One or more sort columns not found in dataframe")

    if eval_columns:
        for column in eval_columns:
            if column in df.columns:
                try:
                    df[column] = df[column].apply(safe_literal_eval)
                except Exception as e:
                    print(f"Error applying safe_literal_eval to column '{column}': {e}")

    return df


def get_model_name(model_name: str, prediction_time_: Tuple[int, int]) -> str:
    """
    Create a model name based on the time of day.

    Parameters
    ----------
    model_name : str
        The base name of the model.
    prediction_time_ : tuple of int
        A tuple representing the time of day (hour, minute).

    Returns
    -------
    str
        A string representing the model name based on the time of day.
    """
    hour_, min_ = prediction_time_
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + "_" + f"{hour_:02}" + min_
    return model_name


def load_saved_model(model_file_path: Path, model_name: str, prediction_time: Optional[Tuple[int, int]] = None) -> Any:
    """
    Load a saved model from a file.

    Parameters
    ----------
    model_file_path : Path
        The path to the directory where the model is saved.
    model_name : str
        The base name of the model.
    prediction_time : tuple of int, optional
        The time of day the model was trained for.

    Returns
    -------
    Any
        The loaded model.

    Raises
    ------
    ModelLoadError
        If the model file cannot be found or loaded.
    """
    if prediction_time:
        model_name = get_model_name(model_name, prediction_time)

    full_path = model_file_path / model_name
    full_path = full_path.with_suffix(".joblib")

    try:
        model = load(full_path)
        return model
    except FileNotFoundError:
        raise ModelLoadError(
            f"Model named {model_name} not found at path: {model_file_path}"
        )
    except Exception as e:
        raise ModelLoadError(f"Error loading model called {model_name}: {e}")


def get_dict_cols(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize DataFrame columns into predefined groups.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to categorize.

    Returns
    -------
    dict
        A dictionary where keys are column group names and values are lists of column names in each group.
    """
    not_used_in_training_vars = [
        "snapshot_id",
        "snapshot_date",
        "prediction_time",
        "visit_number",
        "training_validation_test",
        "random_number",
    ]
    arrival_and_demographic_vars = [
        "elapsed_los",
        "sex",
        "age_group",
        "age_on_arrival",
        "arrival_method",
    ]
    summary_vars = [
        "num_obs",
        "num_obs_events",
        "num_obs_types",
        "num_lab_batteries_ordered",
    ]

    location_vars = []
    observations_vars = []
    labs_vars = []
    consults_vars = [
        "has_consultation",
        "consultation_sequence",
        "final_sequence",
        "specialty",
    ]
    outcome_vars = ["is_admitted"]

    for col in df.columns:
        if (
            col in not_used_in_training_vars
            or col in arrival_and_demographic_vars
            or col in summary_vars
        ):
            continue
        elif "visited" in col or "location" in col:
            location_vars.append(col)
        elif "num_obs" in col or "latest_obs" in col:
            observations_vars.append(col)
        elif "lab_orders" in col or "latest_lab_results" in col:
            labs_vars.append(col)
        elif col in consults_vars or col in outcome_vars:
            continue  # Already categorized
        else:
            print(f"Column '{col}' did not match any predefined group")

    col_group_names = [
        "not used in training",
        "arrival and demographic",
        "summary",
        "location",
        "observations",
        "lab orders and results",
        "consults",
        "outcome",
    ]

    col_groups = [
        not_used_in_training_vars,
        arrival_and_demographic_vars,
        summary_vars,
        location_vars,
        observations_vars,
        labs_vars,
        consults_vars,
        outcome_vars,
    ]

    dict_col_groups = {
        category: var_list for category, var_list in zip(col_group_names, col_groups)
    }

    return dict_col_groups
