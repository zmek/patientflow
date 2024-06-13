"""
Contains functions for loading built-in datasets
"""

import pandas as pd
import os
import ast  # to convert tuples to strings

from pathlib import Path


def safe_literal_eval(s):
    try:
        # Check if the string is "nan" or empty and return None
        if pd.isna(s) or str(s).strip().lower() == "nan":
            return None
        # Otherwise, evaluate it as a tuple
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Return None if there's any other error during evaluation
        return None


def ed_admissions_get_data(path_ed_data):
    """
    Loads XXX ED visits

    Returns:
    pd.DataFrame: A dataframe with the ED visits. See data dictionary
    """
    path = os.path.join(Path().home(), path_ed_data)

    # read dataframe
    df = pd.read_csv(path, parse_dates=True)

    # sort by visit and date if in dataset
    sort_columns = [
        col for col in ["visit_number", "snapshot_datetime"] if col in df.columns
    ]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)

    # convert strings to tuples
    if "prediction_time" in df.columns:
        df["prediction_time"] = df["prediction_time"].apply(
            lambda x: ast.literal_eval(x)
        )

    if "consultation_sequence" in df.columns:
        df["consultation_sequence"] = df["consultation_sequence"].apply(
            safe_literal_eval
        )

    if "final_sequence" in df.columns:
        df["final_sequence"] = df["final_sequence"].apply(safe_literal_eval)

    return df
