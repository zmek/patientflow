"""
Module for preparing data, loading models, and organizing snapshots for inference.

This module provides functionality to load a trained model, prepare data for
making predictions, calculate arrival rates, and organize snapshot data. It allows for selecting one
snapshot per visit, filtering snapshots by prediction time, and mapping
snapshot dates to corresponding indices.

Functions
---------
prepare_for_inference(model_file_path, model_name, prediction_time=None,
                      model_only=False, df=None, data_path=None,
                      single_snapshot_per_visit=True, index_column='snapshot_id',
                      sort_columns=None, eval_columns=None,
                      exclude_from_training_data=None)
    Loads a model and prepares data for inference.

select_one_snapshot_per_visit(df, visit_col, seed=42)
    Selects one snapshot per visit based on a random number and returns the filtered DataFrame.

get_snapshots_at_prediction_time(df, prediction_time_, exclude_columns, single_snapshot_per_visit=True)
    Filters the DataFrame by prediction time and optionally selects one snapshot per visit.

prepare_snapshots_dict(df, start_dt=None, end_dt=None)
    Prepares a dictionary mapping snapshot dates to their corresponding snapshot indices.

calculate_time_varying_arrival_rates(df, yta_time_interval)
    Calculates the time-varying arrival rates for a dataset indexed by datetime.
"""

import pandas as pd
import numpy as np
import random
from patientflow.load import data_from_csv, load_saved_model, get_dict_cols
from datetime import datetime, timedelta
from functools import reduce


from typing import Dict, Any
from patientflow.errors import MissingKeysError


def convert_set_to_dummies(df, column, prefix):
    # Explode the set into rows
    exploded_df = df[column].explode().dropna().to_frame()

    # Create dummy variables for each unique item with a specified prefix
    dummies = pd.get_dummies(exploded_df[column], prefix=prefix)

    # # Sum the dummies back to the original DataFrame's index
    dummies = dummies.groupby(dummies.index).sum()

    # Convert dummy variables to boolean
    dummies = dummies.astype(bool)

    return dummies


def convert_dict_to_values(df, column, prefix):
    def extract_relevant_value(d):
        if isinstance(d, dict):
            if "value_as_real" in d or "value_as_text" in d:
                return (
                    d.get("value_as_real")
                    if d.get("value_as_real") is not None
                    else d.get("value_as_text")
                )
            else:
                return d  # Return the dictionary as is if it does not contain 'value_as_real' or 'value_as_text'
        return d  # Return the value as is if it is not a dictionary

    # Apply the extraction function to each entry in the dictionary column
    extracted_values = df[column].apply(
        lambda x: {k: extract_relevant_value(v) for k, v in x.items()}
    )

    # Create a DataFrame from the processed dictionary column
    dict_df = extracted_values.apply(pd.Series)

    # Add a prefix to the column names
    dict_df.columns = [f"{prefix}_{col}" for col in dict_df.columns]

    return dict_df


# function that will assign each mrn to one of training, validation, making a random choice that is weighted by the proportion of visits occuring in each set
def apply_set(row):
    return random.choices(
        ["train", "valid", "test"],
        weights=[row.training_set, row.validation_set, row.test_set],
    )[0]


def assign_mrns(
    df,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    col_name="arrival_datetime",
    grouping_cols=["mrn", "encounter"],
):

    # assign each mrn to only one of the three sets to ensure no visit appears in more than one set
    mrns = df.groupby(grouping_cols)[col_name].max().reset_index()
    mrns["training_set"] = mrns[col_name].dt.date < start_validation_set
    mrns["validation_set"] = (mrns[col_name].dt.date >= start_validation_set) & (
        mrns[col_name].dt.date < start_test_set
    )
    mrns["test_set"] = mrns[col_name].dt.date >= start_test_set
    mrns = mrns.groupby("mrn")[["training_set", "validation_set", "test_set"]].sum()
    mrns["training_validation_test"] = mrns.apply(apply_set, axis=1)
    print(
        "\n" + f"{mrns[mrns.training_set * mrns.validation_set != 0].shape[0]} "
        f"mrns are in both training and validation sets, of a total of "
        f"{mrns[mrns.training_set + mrns.validation_set > 0].shape[0]} "
        "mrns in one or other set"
    )
    print(
        f"{mrns[mrns.validation_set * mrns.test_set != 0].shape[0]} "
        f"mrns are in both validation and test sets, of a total of "
        f"{mrns[mrns.validation_set + mrns.test_set > 0].shape[0]} "
        "mrns in one or other set"
    )
    print(
        f"{mrns[mrns.training_set * mrns.test_set != 0].shape[0]} "
        f"mrns are in both training and test sets, of a total of "
        f"{mrns[mrns.training_set + mrns.test_set > 0].shape[0]} "
        "mrns in one or other set"
    )
    print(
        f"{mrns[mrns.training_set * mrns.validation_set * mrns.test_set != 0].shape[0]} "
        f"mrns are in all three sets, of a total of "
        f"{mrns.shape[0]} mrns"
    )
    return mrns


def assign_mrn_to_training_validation_test_set(
    df,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    yta=None,
    col_name="arrival_datetime",
    grouping_cols=["mrn", "encounter"]
):
    
    if "snapshot_date" not in df.columns:
        df["snapshot_date"] = df["snapshot_datetime"].dt.date
        remove_snapshot_date = True
    else:
        remove_snapshot_date = False
        
    set_assignment = assign_mrns(
        df,
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        col_name,
        grouping_cols
    ).reset_index()

    print(
        "\nNumber of rows before assigning mrn to a single set - training, validation or test"
    )
    print(df.shape)

    df.loc["training_validation_test"] = None

    # Define the criteria for each set

    training_mrns = set_assignment[set_assignment.training_validation_test == "train"][
        "mrn"
    ]
    validation_mrns = set_assignment[
        set_assignment.training_validation_test == "valid"
    ]["mrn"]
    test_mrns = set_assignment[set_assignment.training_validation_test == "test"]["mrn"]

    # Assign relevant set
    df.loc[
        (df[col_name].dt.date < start_validation_set) & (df.mrn.isin(training_mrns)),
        "training_validation_test",
    ] = "train"
    df.loc[
        (df[col_name].dt.date >= start_validation_set)
        & (df[col_name].dt.date < start_test_set)
        & (df.mrn.isin(validation_mrns)),
        "training_validation_test",
    ] = "valid"
    df.loc[
        (df[col_name].dt.date >= start_test_set)
        & (df[col_name].dt.date < end_test_set)
        & (df.mrn.isin(test_mrns)),
        "training_validation_test",
    ] = "test"

    # Filter to include only the rows that were assigned to a set
    df = df[df["training_validation_test"].notnull()]

    # Remove any snapshots that fall outside the start and end dates for the relevant set
    df = df[
        (
            (df.training_validation_test == "train")
            & (df.snapshot_date < start_validation_set)
        )
        | (
            (df.training_validation_test == "valid")
            & (df.snapshot_date >= start_validation_set)
            & (df.snapshot_date < start_test_set)
        )
        | (
            (df.training_validation_test == "test")
            & (df.snapshot_date >= start_test_set)
            & (df.snapshot_date < end_test_set)
        )
    ]

    print("Number of rows after assigning mrn to a set")
    print(df.shape)

    if yta is not None:
        yta.loc[:, "training_validation_test"] = None
        # Assign relevant set to yta
        yta.loc[
            (yta[col_name].dt.date < start_validation_set), "training_validation_test"
        ] = "train"
        yta.loc[
            (yta[col_name].dt.date >= start_validation_set)
            & (yta[col_name].dt.date < start_test_set),
            "training_validation_test",
        ] = "valid"
        yta.loc[
            (yta[col_name].dt.date >= start_test_set)
            & (yta[col_name].dt.date < end_test_set),
            "training_validation_test",
        ] = "test"

        # Remove any snapshots that fall outside the start and end dates for the relevant set
        yta = yta[
            (
                (yta.training_validation_test == "train")
                & (yta[col_name].dt.date < start_validation_set)
            )
            | (
                (yta.training_validation_test == "valid")
                & (yta[col_name].dt.date >= start_validation_set)
                & (yta[col_name].dt.date < start_test_set)
            )
            | (
                (yta.training_validation_test == "test")
                & (yta[col_name].dt.date >= start_test_set)
                & (yta[col_name].dt.date < end_test_set)
            )
        ]

        return (df, yta)
    
    if remove_snapshot_date is not None:
        df.drop(columns = 'snapshot_date', inplace=True)

    return df


def prep_uclh_dataset_for_inference(
    df,
    uclh,
    remove_bed_requests=False,
    exclude_minority_categories=False,
    inference_time=True,
):
    pd.set_option("future.no_silent_downcasting", True)

    if exclude_minority_categories:
        df = df[~df.sex.isin(["U", "I"])].copy()  # Ensure it's a copy

    if remove_bed_requests:
        df["has_bed_request"] = df["has_bed_request"].fillna(False)
        df["has_bed_request"] = df["has_bed_request"].astype(bool)
        df = df[~df.has_bed_request].copy()  # Ensure it's a copy

    # Convert locations from set to dummy variables
    visited = convert_set_to_dummies(df, "visited_location_types", "visited")

    # Convert number of observations dictionary to values
    num_obs = convert_dict_to_values(df, "observation_counts", "num_obs")

    # Convert obs and set missing values
    latest_obs = convert_dict_to_values(df, "latest_observation_values", "latest_obs")
    latest_obs.loc[
        latest_obs.latest_obs_RESPIRATIONS == 0, "latest_obs_RESPIRATIONS"
    ] = pd.NA
    latest_obs.loc[
        latest_obs.latest_obs_TEMPERATURE > 110, "latest_obs_TEMPERATURE"
    ] = pd.NA
    latest_obs["latest_obs_R NEWS SCORE RESULT - DISPLAYED"] = latest_obs[
        "latest_obs_R NEWS SCORE RESULT - DISPLAYED"
    ].astype("float")
    latest_obs.loc[
        latest_obs["latest_obs_R UCLH ED MANCHESTER TRIAGE OBJECTIVE PAIN SCORE"].isin(
            [r"Severe\E\Very Severe", r"Severe\Very Severe"]
        ),
        "latest_obs_R UCLH ED MANCHESTER TRIAGE OBJECTIVE PAIN SCORE",
    ] = "Severe_Very Severe"

    # Convert lab orders from set to dummies
    lab_orders = convert_set_to_dummies(df, "requested_lab_batteries", "lab_orders")

    # Convert lab results from dict to values
    lab_results = convert_dict_to_values(df, "latest_lab_results", "latest_lab_results")

    # Create dummy variable for consultations (used in prob admission model, in which consultations data is otherwise not used)
    df["has_consultation"] = df.consultations.map(len) > 0

    if inference_time:
        df["visit_number"] = df.index

    dfs_to_join = (
        [
            df[
                [
                    "snapshot_date",
                    "prediction_time",
                    "visit_number",
                    "elapsed_los",
                    "sex",
                    "age_on_arrival",
                    "arrival_method",
                    "current_location_type",
                    "total_locations_visited",
                    "num_obs",
                    "num_obs_events",
                    "num_obs_types",
                    "num_lab_batteries_ordered",
                    "has_consultation",
                    "has_bed_request",
                    "consultations",
                ]
            ],
            visited,
            num_obs,
            latest_obs,
            lab_orders,
            lab_results,
        ]
        if uclh
        else [
            df[
                [
                    "snapshot_date",
                    "prediction_time",
                    "visit_number",
                    "elapsed_los",
                    "sex",
                    "age_group",
                    "arrival_method",
                    "current_location_type",
                    "total_locations_visited",
                    "num_obs",
                    "num_obs_events",
                    "num_obs_types",
                    "num_lab_batteries_ordered",
                    "has_consultation",
                    "has_bed_request",
                    "consultations",
                ]
            ],
            visited,
            num_obs,
            latest_obs,
            lab_orders,
            lab_results,
        ]
    )

    if not inference_time:
        dfs_to_join.append(
            df[
                [
                    "training_validation_test",
                    "all_consultations",
                    "specialty",
                    "destination",
                ]
            ]
        )

    new = reduce(lambda left, right: left.join(right, how="left"), dfs_to_join)

    for col in new.select_dtypes(include="object").columns:
        if new[col].dropna().isin([True, False]).all():
            new[col] = new[col].fillna(False)
            new[col] = new[col].astype("bool")

    bool_cols = new.select_dtypes(include="bool").columns
    new[bool_cols] = new[bool_cols].fillna(False)

    if exclude_minority_categories:
        new = new[~(new.current_location_type == "taf")].copy()
        new = new[~(new.visited_taf)].copy()
        new.drop(columns="visited_taf", inplace=True)

    float_cols = [
        col
        for col in new.select_dtypes(include="float").columns
        if col.startswith("num_obs")
    ]
    new[float_cols] = new[float_cols].fillna(0)

    new.columns = (
        new.columns.str.lower()
        .str.replace(" - displayed", "")
        .str.replace(" ", "_")
        .str.replace("_r_", "_")
    )
    new = new.drop(columns="latest_lab_results_hco3", errors="ignore")

    new = new.rename(
        columns={
            "num_obs_uclh_ed_manchester_triage_subjective_pain_score": "num_obs_subjective_pain_score",
            "num_obs_uclh_ed_manchester_triage_objective_pain_score": "num_obs_objective_pain_score",
            "num_obs_uclh_ed_manchester_triage_calculated_acuity": "num_obs_manchester_triage_acuity",
            "latest_obs_uclh_ed_manchester_triage_objective_pain_score": "latest_obs_objective_pain_score",
            "latest_obs_uclh_ed_manchester_triage_calculated_acuity": "latest_obs_manchester_triage_acuity",
        }
    )

    if inference_time:
        new = new.rename(columns={"consultations": "consultation_sequence"})
        new["consultation_sequence"] = new["consultation_sequence"].apply(
            lambda x: tuple(x) if x else ()
        )

    if not inference_time:
        new["is_admitted"] = df.destination == 2
        new.drop(columns="destination", inplace=True)
        np.random.seed(seed=42)
        new["random_number"] = np.random.randint(0, len(new), new.shape[0])

    if remove_bed_requests:
        new = new.drop(columns="has_bed_request")

    new = new.reset_index(drop=True)
    new.index.name = "snapshot_id"

    return new


def create_special_category_objects(uclh):
    special_category_dict = {
        "medical": 0.0,
        "surgical": 0.0,
        "haem/onc": 0.0,
        "paediatric": 1.0,
    }

    # Function to determine if the patient is a child
    def is_paediatric_uclh(row):
        return row["age_on_arrival"] < 18

    def is_paediatric_non_uclh(row):
        return row["age_group"] == "0-17"

    if uclh:
        special_category_func = is_paediatric_uclh
    else:
        special_category_func = is_paediatric_non_uclh

    # Function to return the opposite of special_category_func
    def opposite_special_category_func(row):
        return not special_category_func(row)

    special_func_map = {
        "paediatric": special_category_func,
        "default": opposite_special_category_func,
    }

    special_params = {
        "special_category_func": special_category_func,
        "special_category_dict": special_category_dict,
        "special_func_map": special_func_map,
    }

    return special_params


def validate_special_category_objects(special_params: Dict[str, Any]) -> None:
    required_keys = [
        "special_category_func",
        "special_category_dict",
        "special_func_map",
    ]
    missing_keys = [key for key in required_keys if key not in special_params]

    if missing_keys:
        raise MissingKeysError(missing_keys)


def create_yta_filters(uclh):
    # Get the special category parameters
    special_params = create_special_category_objects(uclh)

    # Extract necessary functions and data from the special_params
    special_category_dict = special_params["special_category_dict"]

    # Create the specialty_filters dictionary
    specialty_filters = {}

    for specialty, is_paediatric_flag in special_category_dict.items():
        if is_paediatric_flag == 1.0:
            # For the paediatric specialty, set `is_child` to True
            specialty_filters[specialty] = {"is_child": True}
        else:
            # For other specialties, set `is_child` to False
            specialty_filters[specialty] = {"specialty": specialty, "is_child": False}

    return specialty_filters


def select_one_snapshot_per_visit(df, visit_col, seed=42):
    # Generate random numbers if not present
    if "random_number" not in df.columns:
        if seed is not None:
            np.random.seed(seed)
        df["random_number"] = np.random.random(size=len(df))

    # Select the row with the maximum random_number for each visit
    max_indices = df.groupby(visit_col)["random_number"].idxmax()
    return df.loc[max_indices].drop(columns=["random_number"])


def get_snapshots_at_prediction_time(
    df, prediction_time_, exclude_columns, single_snapshot_per_visit=True, visit_col="visit_number", label_col="is_admitted"
):
    """
    Get snapshots of data at a specific prediction time with configurable visit and label columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    prediction_time_ : str or datetime
        The specific prediction time to filter for
    exclude_columns : list
        List of columns to exclude from the final DataFrame
    single_snapshot_per_visit : bool, default=True
        Whether to select only one snapshot per visit
    visit_col : str, default="visit_number"
        Name of the column containing visit identifiers
    label_col : str, default="is_admitted"
        Name of the column containing the target labels
    
    Returns:
    --------
    tuple(pandas.DataFrame, pandas.Series)
        Processed DataFrame and corresponding labels
    """
    # Filter by the time of day while keeping the original index
    df_tod = df[df["prediction_time"] == prediction_time_].copy()
    
    if single_snapshot_per_visit:
        # Group by visit_col and get the row with the maximum 'random_number'
        df_single = select_one_snapshot_per_visit(df_tod, visit_col)
        # Create label array with the same index
        y = df_single.pop(label_col).astype(int)
        # Drop specified columns and ensure we do not reset the index
        df_single.drop(columns=exclude_columns, inplace=True)
        return df_single, y
    else:
        # Directly modify df_tod without resetting the index
        df_tod.drop(columns=["random_number"] + exclude_columns, inplace=True)
        y = df_tod.pop(label_col).astype(int)
        return df_tod, y


def prepare_for_inference(
    model_file_path,
    model_name,
    prediction_time=None,
    model_only=False,
    df=None,
    data_path=None,
    single_snapshot_per_visit=True,
    index_column="snapshot_id",
    sort_columns=["visit_number", "snapshot_date", "prediction_time"],
    eval_columns=["prediction_time", "consultation_sequence", "final_sequence"],
    exclude_from_training_data=["visit_number", "snapshot_date", "prediction_time"],
):
    """
    Load a trained model and prepare data for making predictions.

    This function retrieves a trained model from a specified file path and,
    if requested, prepares the data required for inference. The data can be
    provided either as a DataFrame or as a file path to a CSV file. The function
    allows filtering and processing of the data to match the model's requirements.

    Parameters
    ----------
    model_file_path : str
        The file path where the trained model is saved.
    model_name : str
        The name of the model to be loaded.
    prediction_time : str, optional
        The time at which predictions are to be made. This is used to filter
        the data for the relevant time snapshot.
    model_only : bool, optional
        If True, only the model is returned. If False, both the prepared data
        and the model are returned. Default is False.
    df : pandas.DataFrame, optional
        The DataFrame containing the data to be used for inference. If not
        provided, data_path must be specified.
    data_path : str, optional
        The file path to a CSV file containing the data to be used for inference.
        Ignored if `df` is provided.
    single_snapshot_per_visit : bool, optional
        If True, only a single snapshot per visit is considered. Default is True.
    index_column : str, optional
        The name of the index column in the data. Default is 'snapshot_id'.
    sort_columns : list of str, optional
        The columns to sort the data by. Default is ["visit_number", "snapshot_date", "prediction_time"].
    eval_columns : list of str, optional
        The columns that require literal evaluation of their content when loading from csv.
        Default is ["prediction_time", "consultation_sequence", "final_sequence"].
    exclude_from_training_data : list of str, optional
        The columns to be excluded from the training data. Default is ["visit_number", "snapshot_date", "prediction_time"].

    Returns
    -------
    model : object
        The loaded model.
    X_test : pandas.DataFrame, optional
        The features prepared for testing, returned only if model_only is False.
    y_test : pandas.Series, optional
        The labels corresponding to X_test, returned only if model_only is False.

    Raises
    ------
    KeyError
        If the 'training_validation_test' column is not found in the provided DataFrame.

    Notes
    -----
    Either `df` or `data_path` must be provided. If neither is provided or if `df`
    is empty, the function will print an error message and return None.

    """

    # retrieve model trained for this time of day
    model = load_saved_model(model_file_path, model_name, prediction_time)

    if model_only:
        return model

    if data_path:
        df = data_from_csv(data_path, index_column, sort_columns, eval_columns)
    elif df is None or df.empty:
        print("Please supply a dataset if not passing a data path")
        return None

    try:
        test_df = (
            df[df.training_validation_test == "test"]
            .drop(columns="training_validation_test")
            .copy()
        )
    except KeyError:
        print("Column training_validation_test not found in dataframe")
        return None

    X_test, y_test = get_snapshots_at_prediction_time(
        test_df,
        prediction_time,
        exclude_from_training_data,
        single_snapshot_per_visit,
    )

    return X_test, y_test, model


def prepare_snapshots_dict(df, start_dt=None, end_dt=None):
    """
    Prepares a dictionary mapping snapshot dates to their corresponding snapshot indices.

    Args:
    df (pd.DataFrame): DataFrame containing at least a 'snapshot_date' column which represents the dates.
    start_dt (datetime.date): Start date (optional)
    end_dt (datetime.date): End date (optional)

    Returns:
    dict: A dictionary where keys are dates and values are arrays of indices corresponding to each date's snapshots.
    A array can be empty if there are no snapshots associated with a date

    """
    # Ensure 'snapshot_date' is in the DataFrame
    if "snapshot_date" not in df.columns:
        raise ValueError("DataFrame must include a 'snapshot_date' column")

    # Group the DataFrame by 'snapshot_date' and collect the indices for each group
    snapshots_dict = {
        date: group.index.tolist() for date, group in df.groupby("snapshot_date")
    }

    # If start_dt and end_dt are specified, add any missing keys from prediction_dates
    if start_dt:
        prediction_dates = pd.date_range(
            start=start_dt, end=end_dt, freq="D"
        ).date.tolist()[:-1]
        for dt in prediction_dates:
            if dt not in snapshots_dict:
                snapshots_dict[dt] = []

    return snapshots_dict


def calculate_time_varying_arrival_rates(df, yta_time_interval):
    """
    Calculate the time-varying arrival rates for a dataset indexed by datetime.

    This function computes the arrival rates for each time interval specified, across the entire date range present in the dataframe. The arrival rate is calculated as the number of entries in the dataframe for each time interval, divided by the number of days in the dataset's timespan.

    Parameters
    df (pandas.DataFrame): A DataFrame indexed by datetime, representing the data for which arrival rates are to be calculated. The index of the DataFrame should be of datetime type.
    yta_time_interval (int): The time interval, in minutes, for which the arrival rates are to be calculated. For example, if `yta_time_interval=60`, the function will calculate hourly arrival rates.

    Returns
    dict: A dictionary where the keys are the start times of each interval (as `datetime.time` objects), and the values are the corresponding arrival rates (as floats).

    Raises
    TypeError: If the index of the DataFrame is not a datetime index.

    """
    # Validate that the DataFrame index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a DatetimeIndex.")

    # Determine the start and end date of the data
    start_dt = df.index.min()
    end_dt = df.index.max()

    # Convert start and end times to datetime if they are not already
    if not isinstance(start_dt, datetime):
        start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S%z")

    if not isinstance(end_dt, datetime):
        end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S%z")

    # Calculate the total number of days covered by the dataset
    num_days = pd.Series(df.index.date).nunique()
    print(
        f"Calculating time-varying arrival rates for data provided, which spans {num_days} unique dates"
    )

    arrival_rates_dict = {}

    # Initialize a time object to iterate through one day in the specified intervals
    _start_datetime = datetime(1970, 1, 1, 0, 0, 0, 0)
    _stop_datetime = _start_datetime + timedelta(days=1)

    # Iterate over each interval in a single day to calculate the arrival rate
    while _start_datetime != _stop_datetime:
        _start_time = _start_datetime.time()
        _end_time = (_start_datetime + timedelta(minutes=yta_time_interval)).time()

        # Filter the dataframe for entries within the current time interval
        _df = df.between_time(_start_time, _end_time, inclusive="left")

        # Calculate and store the arrival rate for the interval
        arrival_rates_dict[_start_time] = _df.shape[0] / num_days

        # Move to the next interval
        _start_datetime = _start_datetime + timedelta(minutes=yta_time_interval)

    return arrival_rates_dict


# Function to generate description based on column name
def generate_description(col_name):
    manual_descriptions = get_manual_descriptions()

    # Check if manual description is provided
    if col_name in manual_descriptions:
        return manual_descriptions[col_name]

    if (
        col_name.startswith("num")
        and not col_name.startswith("num_obs")
        and not col_name.startswith("num_orders")
    ):
        return "Number of times " + col_name[4:] + " has been recorded"
    if col_name.startswith("num_obs"):
        return "Number of observations of " + col_name[8:]
    if col_name.startswith("latest_obs"):
        return "Latest result for " + col_name[11:]
    if col_name.startswith("latest_lab"):
        return "Latest result for " + col_name[19:]
    if col_name.startswith("lab_orders"):
        return "Request for lab battery " + col_name[11:] + " has been placed"
    if col_name.startswith("visited"):
        return "Patient visited " + col_name[8:] + " previously or is there now"
    else:
        return col_name


def additional_details(column, col_name):
    def is_date(string):
        try:
            # Try to parse the string using the strptime method
            datetime.strptime(
                string, "%Y-%m-%d"
            )  # You can adjust the format to match your date format
            return True
        except (ValueError, TypeError):
            return False

    # Convert to datetime if it's an object but formatted as a date
    if column.dtype == "object" and all(
        is_date(str(x)) for x in column.dropna().unique()
    ):
        column = pd.to_datetime(column)
        return f"Date Range: {column.min().strftime('%Y-%m-%d')} - {column.max().strftime('%Y-%m-%d')}"

    if column.dtype in ["object", "category", "bool"]:
        # Categorical data: Frequency of unique values

        if len(column.value_counts()) <= 12:
            value_counts = column.value_counts(dropna=False).to_dict()
            value_counts = dict(sorted(value_counts.items(), key=lambda x: str(x[0])))
            value_counts_formatted = {k: f"{v:,}" for k, v in value_counts.items()}
            return f"Frequencies: {value_counts_formatted}"
        value_counts = column.value_counts(dropna=False)[0:12].to_dict()
        value_counts = dict(sorted(value_counts.items(), key=lambda x: str(x[0])))
        value_counts_formatted = {k: f"{v:,}" for k, v in value_counts.items()}
        return f"Frequencies (highest 12): {value_counts_formatted}"

    if pd.api.types.is_float_dtype(column):
        # Float data: Range with rounding
        na_count = column.isna().sum()
        column = column.dropna()
        return f"Range: {column.min():.2f} - {column.max():.2f},  Mean: {column.mean():.2f}, Std Dev: {column.std():.2f}, NA: {na_count}"
    if pd.api.types.is_integer_dtype(column):
        # Float data: Range without rounding
        na_count = column.isna().sum()
        column = column.dropna()
        return f"Range: {column.min()} - {column.max()}, Mean: {column.mean():.2f}, Std Dev: {column.std():.2f}, NA: {na_count}"
    if pd.api.types.is_datetime64_any_dtype(column):
        # Datetime data: Minimum and Maximum dates
        return f"Date Range: {column.min().strftime('%Y-%m-%d %H:%M')} - {column.max().strftime('%Y-%m-%d %H:%M')}"
    else:
        return "N/A"


def find_group_for_colname(column, dict_col_groups):
    for key, values_list in dict_col_groups.items():
        if column in values_list:
            return key
    return None


def get_manual_descriptions():
    manual_descriptions = {
        "snapshot_id": "Unique identifier for the visit snapshot (an internal reference field only)",
        "snapshot_date": "Date of visit, shifted by a random number of days",
        "visit_number": "Hospital visit number (replaced with fictional number, but consistent across visit snapshots is retained)",
        "arrival_method": "How the patient arrived at the ED",
        "current_location_type": "Location in ED currently",
        "sex": "Sex of patient",
        "age_on_arrival": "Age in years on arrival at ED",
        "elapsed_los": "Elapsed time since patient arrived in ED (seconds)",
        "num_obs": "Number of observations recorded",
        "num_obs_events": "Number of unique events when one or more observations have been recorded",
        "num_obs_types": "Number of types of observations recorded",
        "num_lab_batteries_ordered": "Number of lab batteries ordered (each many contain multiple tests)",
        "has_consult": "One or more consult request has been made",
        "total_locations_visited": "Number of ED locations visited",
        "is_admitted": "Patient was admitted after ED",
        "hour_of_day": "Hour of day at which visit was sampled",
        "consultation_sequence": "Consultation sequence at time of snapshot",
        "has_consultation": "Consultation request made before time of snapshot",
        "final_sequence": "Consultation sequence at end of visit",
        "observed_specialty": "Specialty of admission at end of visit",
        "random_number": "A random number that will be used during model training to sample one visit snapshot per visit",
        "prediction_time": "The time of day at which the visit was observed",
        "training_validation_test": "Whether visit snapshot is assigned to training, validation or test set",
        "age_group": "Age group",
        "is_child": "Is under age of 18 on day of arrival",
        "ed_visit_start_dttm": "Timestamp of visit start",
    }
    return manual_descriptions


def write_data_dict(df, dict_name, dict_path):
    cols_to_exclude = ["snapshot_id", "visit_number"]

    df = df.copy(deep=True)

    if "visits" in dict_name:
        df.consultation_sequence = df.consultation_sequence.apply(
            lambda x: str(x)
        ).to_frame()
        df.final_sequence = df.final_sequence.apply(lambda x: str(x)).to_frame()
        df_admitted = df[df.is_admitted]
        df_not_admitted = df[~df.is_admitted]
        dict_col_groups = get_dict_cols(df)

        data_dict = pd.DataFrame(
            {
                "Variable type": [
                    find_group_for_colname(col, dict_col_groups) for col in df.columns
                ],
                "Column Name": df.columns,
                "Data Type": df.dtypes,
                "Description": [generate_description(col) for col in df.columns],
                "Whole dataset": [
                    additional_details(df[col], col)
                    if col not in cols_to_exclude
                    else ""
                    for col in df.columns
                ],
                "Admitted": [
                    additional_details(df_admitted[col], col)
                    if col not in cols_to_exclude
                    else ""
                    for col in df_admitted.columns
                ],
                "Not admitted": [
                    additional_details(df_not_admitted[col], col)
                    if col not in cols_to_exclude
                    else ""
                    for col in df_not_admitted.columns
                ],
            }
        )
        data_dict["Whole dataset"] = data_dict["Whole dataset"].str.replace("'", "")
        data_dict["Admitted"] = data_dict["Admitted"].str.replace("'", "")
        data_dict["Not admitted"] = data_dict["Not admitted"].str.replace("'", "")

    else:
        data_dict = pd.DataFrame(
            {
                "Column Name": df.columns,
                "Data Type": df.dtypes,
                "Description": [generate_description(col) for col in df.columns],
                "Additional Details": [
                    additional_details(df[col], col)
                    if col not in cols_to_exclude
                    else ""
                    for col in df.columns
                ],
            }
        )
        data_dict["Additional Details"] = data_dict["Additional Details"].str.replace(
            "'", ""
        )

    # Export to Markdown and csv for data dictionary
    data_dict.to_markdown(str(dict_path) + "/" + dict_name + ".md", index=False)
    data_dict.to_csv(str(dict_path) + "/" + dict_name + ".csv", index=False)

    return data_dict
