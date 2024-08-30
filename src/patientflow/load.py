"""
Contains functions for loading data from csv and for loading saved models
"""

import ast  # to convert tuples to strings
import os
from pathlib import Path

import pandas as pd
from joblib import load
from errors import ModelLoadError
from prepare import get_model_name

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
        raise ModelLoadError(f"Error loading model: {e}")
