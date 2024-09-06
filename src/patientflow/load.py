"""
Contains functions for loading config files, data from csv and saved models
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
            else:
                print(
                    "Error: 'file_dates' key not found or empty in the configuration file."
                )
                return None

        # Convert list of times of day at which predictions will be made (currently stored as lists) to list of tuples
        if "prediction_times" in config:
            prediction_times = [tuple(item) for item in config["prediction_times"]]
        else:
            print("Error: 'prediction_times' key not found in the configuration file.")
            return None

        # Load the dates defining the beginning and end of training, validation and test sets
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

        # desired error for Poisson distribution (1 - sum of each approximated Poisson)
        epsilon = config.get("epsilon", 10**-7)

        # time interval for the calculation of aspiration yet-to-arrive in minutes
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


def safe_literal_eval(s):
    try:
        if pd.isna(s) or str(s).strip().lower() in ["nan", "none", ""]:
            return None
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def data_from_csv(csv_path, index_column=None, sort_columns=None, eval_columns=None):
    """
    Loads data from csv file

    Args:
    csv_path (str): The path to the ED data file
    index_column (str): The column to set as index
    sort_columns (list): The columns to sort the dataframe by
    eval_columns (list): The columns to apply safe_literal_eval to

    Returns:
    pd.DataFrame: A dataframe with the ED visits. See data dictionary

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


def get_model_name(model_name, prediction_time_):
    """
    Create a model name based on the time of day.

    Parameters
    prediction_time_ (tuple): A tuple representing the time of day (hour, minute).

    Returns
    str: A string representing the model name based on the time of day.

    """
    hour_, min_ = prediction_time_
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + "_" + f"{hour_:02}" + min_
    return model_name


def load_saved_model(model_file_path, model_name, prediction_time=None):

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


def set_file_locations(uclh, data_path, config_file_path=None):
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


def get_dict_cols(df):
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
