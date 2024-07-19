"""
Contains functions for loading built-in datasets
"""

import ast  # to convert tuples to strings
import os
from pathlib import Path

import pandas as pd

def safe_literal_eval(s):
    try:
        if pd.isna(s) or str(s).strip().lower() in ["nan", "none", ""]:
            return None
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def ed_admissions_get_data(path_ed_data):
    """
    Loads XXX ED visits

    Returns
    pd.DataFrame: A dataframe with the ED visits. See data dictionary

    """
    path = os.path.join(Path().home(), path_ed_data)
    df = pd.read_csv(path, parse_dates=True)

    sort_columns = [col for col in ["visit_number", "snapshot_datetime"] if col in df.columns]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)

    for column in ["prediction_time", "consultation_sequence", "final_sequence"]:
        if column in df.columns:
            df[column] = df[column].apply(safe_literal_eval)

    return df
