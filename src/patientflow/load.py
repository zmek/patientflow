"""
This module provides functionality for loading configuration files, data from CSV files, and trained machine learning models.

It includes the following features:

- **Loading Configurations**: Parse YAML configuration files and extract necessary parameters for data processing and modeling.
- **Data Handling**: Load and preprocess data from CSV files, including optional operations like setting an index, sorting, and applying literal evaluation on columns.
- **Model Management**: Load saved machine learning models, customize model filenames based on time, and categorize DataFrame columns into predefined groups for analysis.

The module handles common file and parsing errors, returning appropriate error messages or exceptions.

Functions
---------
parse_args:
    Parses command-line arguments for training models.
set_project_root:
    Validates project root path from specified environment variable.
load_config_file:
    Load a YAML configuration file and extract key parameters.
set_file_paths:
    Sets up the file paths based on UCLH-specific or default parameters.
set_data_file_names:
    Set file locations based on UCLH-specific or default data sources.
safe_literal_eval:
    Safely evaluate string literals into Python objects when loading from csv.
data_from_csv:
    Load and preprocess data from a CSV file.
get_model_name:
    Generate a model name based on the time of day.
load_saved_model:
    Load a machine learning model saved in a joblib file.
get_dict_cols:
    Categorize columns from a DataFrame into predefined groups for analysis.
"""

import ast  # to convert tuples to strings
import os
from pathlib import Path
import sys

import pandas as pd
from joblib import load
from patientflow.errors import ModelLoadError

import yaml
from typing import Any, Dict, Tuple, Union, Optional
import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: The parsed arguments containing 'data_folder_name' and 'uclh' keys.
    """
    parser = argparse.ArgumentParser(description="Train emergency demand models")
    parser.add_argument(
        "--data_folder_name",
        type=str,
        default="data-synthetic",
        help="Location of data for training",
    )
    parser.add_argument(
        "--uclh",
        type=lambda x: x.lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Train using UCLH data (True) or Public data (False)",
    )
    args = parser.parse_args()
    return args

def set_project_root(env_var="PROJECT_ROOT"):
    """
    Validates project root path from specified environment variable.
    
    Args:
        env_var (str): Name of environment variable containing project root path
        
    Returns:
        pathlib.Path: Validated project root path
        
    Raises:
        ValueError: If environment variable is not set
        NotADirectoryError: If path doesn't exist
    """
    from pathlib import Path
    import os
    
    try:
        project_root = Path(os.getenv(env_var))
        if project_root is None:
            raise ValueError(f"{env_var} environment variable not set")
        # if not project_root.exists():
        #     raise NotADirectoryError(f"Path {project_root} does not exist")
        print(f"Project root is {project_root}")
        return project_root
        
    except Exception as e:
        print(f"Error setting project root: {e}")
        print(f"\nCurrent directory: {Path().absolute()}")
        print(f"\nRun one of these commands in a new cell to set {env_var}:")
        print("# Linux/Mac:")
        print(f"%env {env_var}=/path/to/project")
        print("\n# Windows:")
        print(f"%env {env_var}=C:\\path\\to\\project")
        raise

def load_config_file(
    config_file_path: str, return_start_end_dates: bool = False
) -> Optional[Union[Dict[str, Any], Tuple[str, str]]]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file_path : str
        The path to the configuration file.
    return_start_end_dates : bool, optional
        If True, return only the start and end dates from the file (default is False).

    Returns
    -------
    dict or tuple or None
        If `return_start_end_dates` is True, returns a tuple of start and end dates (str).
        Otherwise, returns a dictionary containing the configuration parameters.
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
            # load the dates used in saved data for uclh versions
            if "file_dates" in config and config["file_dates"]:
                start_date, end_date = [str(item) for item in config["file_dates"]]
                return (start_date, end_date)
            else:
                print(
                    "Error: 'file_dates' key not found or empty in the configuration file."
                )
                return None

        params: Dict[str, Any] = {}

        if "prediction_times" in config:
            params["prediction_times"] = [
                tuple(item) for item in config["prediction_times"]
            ]
        else:
            print("Error: 'prediction_times' key not found in the configuration file.")
            sys.exit(1)

        if "modelling_dates" in config and len(config["modelling_dates"]) == 4:
            (
                params["start_training_set"],
                params["start_validation_set"],
                params["start_test_set"],
                params["end_test_set"],
            ) = [item for item in config["modelling_dates"]]
        else:
            print(
                f"Error: expecting 4 modelling dates and only got {len(config.get('modelling_dates', []))}"
            )
            return None

        params["x1"] = float(config.get("x1", 4))
        params["y1"] = float(config.get("y1", 0.76))
        params["x2"] = float(config.get("x2", 12))
        params["y2"] = float(config.get("y2", 0.99))
        params["prediction_window"] = config.get("prediction_window", 480)
        params["epsilon"] = config.get("epsilon", 10**-7)
        params["yta_time_interval"] = config.get("yta_time_interval", 15)

        return params

    except KeyError as e:
        print(f"Error: Missing key in the configuration file: {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid value found in the configuration file: {e}")
        return None


def set_file_paths(
    project_root: Path,
    data_folder_name: str,
    train_dttm: str = None,
    inference_time: bool = False,
    config_file: str = "config.yaml",
    prefix: str = None,
    verbose: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    """
    Sets up the file paths

    Args:
        project_root (Path): Root path of the project
        data_folder_name (str): Name of the folder where data files are located
        train_dttm (str, optional): A string representation of the datetime at which training commenced. Defaults to None
        inference_time (bool, optional): A flag indicating whether it is inference time or not. Defaults to False
        config_file (str, optional): Name of config file. Defaults to "config.yaml"
        prefix (str, optional): String to prefix model folder names. Defaults to None
        verbose (bool, optional): Whether to print path information. Defaults to True

    Returns:
        tuple: Contains (data_file_path, media_file_path, model_file_path, config_path)
    """
    config_path = Path(project_root) / config_file
    if verbose:
        print(f"Configuration will be loaded from: {config_path}")

    data_file_path = Path(project_root) / data_folder_name
    if verbose:
        print(f"Data files will be loaded from: {data_file_path}")

    model_id = data_folder_name.lstrip("data-")
    if prefix:
        model_id = f"{prefix}_{model_id}"
    if train_dttm:
        model_id = f"{model_id}_{train_dttm}"

    model_file_path = Path(project_root) / "trained-models" / model_id
    media_file_path = model_file_path / "media"

    if not inference_time:
        if verbose:
            print(f"Trained models will be saved to: {model_file_path}")
        model_file_path.mkdir(parents=True, exist_ok=True)
        (model_file_path / "model-output").mkdir(parents=False, exist_ok=True)
        media_file_path.mkdir(parents=False, exist_ok=True)
        if verbose:
            print(f"Images will be saved to: {media_file_path}")

    return data_file_path, media_file_path, model_file_path, config_path

def safe_literal_eval(s):
    """
    Safely evaluate a string literal into a Python object.
    Handles list-like strings by converting them to lists.

    Parameters
    ----------
    s : str
        The string to evaluate.

    Returns
    -------
    Any, list, or None
        The evaluated Python object if successful, a list if the input is list-like,
        or None for empty/null values.
    """
    if pd.isna(s) or str(s).strip().lower() in ["nan", "none", ""]:
        return None

    if isinstance(s, str):
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                # Remove square brackets and split by comma
                items = s[1:-1].split(",")
                # Strip whitespace from each item and remove empty strings
                return [item.strip() for item in items if item.strip()]
            except Exception:
                # If the above fails, fall back to ast.literal_eval
                pass

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If ast.literal_eval fails, return the original string
        return s

def load_data(data_file_path, file_name, index_column=None, sort_columns=None, eval_columns=None):
    """
    Loads data from CSV or pickle file with optional transformations.
    
    Parameters
    ----------
    data_file_path : str
        Directory path containing the data file
    file_name : str
        Name of the CSV or pickle file to load
    index_column : str, optional
        Column to set as DataFrame index
    sort_columns : list of str, optional
        Columns to sort DataFrame by
    eval_columns : list of str, optional
        Columns to apply safe_literal_eval to
        
    Returns
    -------
    pd.DataFrame
        Loaded and transformed DataFrame
        
    Raises
    ------
    SystemExit
        If file not found or error occurs during processing
    """
    path = os.path.join(Path().home(), data_file_path, file_name)
    
    if not os.path.exists(path):
        print(f"Data file not found at path: {path}")
        sys.exit(1)
        
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(path, parse_dates=True)
        elif file_name.endswith('.pkl'):
            df = pd.read_pickle(path)
        else:
            print(f"Unsupported file format. Must be CSV or pickle")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if index_column and df.index.name != index_column:
        try:
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


def get_model_name(model_name, prediction_time):
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

    hour_, min_ = prediction_time
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + "_" + f"{hour_:02}" + min_
    return model_name


def load_saved_model(model_file_path, model_name, prediction_time=None):
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
        # retrieve model based on the time of day it is trained for
        model_name = get_model_name(model_name, prediction_time)

    full_path = model_file_path / model_name
    full_path = full_path.with_suffix(".joblib")

    try:
        model = load(full_path)
        return model
    except FileNotFoundError:
        # print(f"Model named {model_name} not found at path: {model_file_path}")
        raise ModelLoadError(
            f"Model named {model_name} not found at path: {model_file_path}"
        )
    except Exception as e:
        # print(f"Error loading model: {e}")
        raise ModelLoadError(f"Error loading model called {model_name}: {e}")


def get_dict_cols(df):
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

    # Create a list of column groups
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

    # Create a list of the column names within those groups
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

    # Use dictionary to combine them
    dict_col_groups = {
        category: var_list for category, var_list in zip(col_group_names, col_groups)
    }

    return dict_col_groups
