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

calculate_time_varying_arrival_rates(df, time_interval)
    Calculates the time-varying arrival rates for a dataset indexed by datetime.
"""


import pandas as pd
import numpy as np
from load import data_from_csv, load_saved_model
from datetime import datetime, timedelta



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
    df, prediction_time_, exclude_columns, single_snapshot_per_visit=True
):
    # Filter by the time of day while keeping the original index
    df_tod = df[df["prediction_time"] == prediction_time_].copy()

    if single_snapshot_per_visit:
        # Group by 'visit_number' and get the row with the maximum 'random_number'
        df_single = select_one_snapshot_per_visit(df_tod, visit_col="visit_number")

        # Create label array with the same index
        y = df_single.pop("is_admitted").astype(int)

        # Drop specified columns and ensure we do not reset the index
        df_single.drop(columns=exclude_columns, inplace=True)

        return df_single, y

    else:
        # Directly modify df_tod without resetting the index
        df_tod.drop(columns=["random_number"] + exclude_columns, inplace=True)
        y = df_tod.pop("is_admitted").astype(int)

        return df_tod, y

    # include one one snapshot per visit and drop the random number


def prepare_for_inference(
    model_file_path,
    model_name,
    prediction_time=None,
    model_only=False,
    df=None,
    data_path=None,
    single_snapshot_per_visit=True,
    index_column="snapshot_id",
    sort_columns=None,
    eval_columns=None,
    exclude_from_training_data=None,
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
                print(dt)
                snapshots_dict[dt] = []

    return snapshots_dict




def calculate_time_varying_arrival_rates(df, time_interval):
    """
    Calculate the time-varying arrival rates for a dataset indexed by datetime.

    This function computes the arrival rates for each time interval specified, across the entire date range present in the dataframe. The arrival rate is calculated as the number of entries in the dataframe for each time interval, divided by the number of days in the dataset's timespan.

    Parameters
    df (pandas.DataFrame): A DataFrame indexed by datetime, representing the data for which arrival rates are to be calculated. The index of the DataFrame should be of datetime type.
    time_interval (int): The time interval, in minutes, for which the arrival rates are to be calculated. For example, if `time_interval=60`, the function will calculate hourly arrival rates.

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
        _end_time = (_start_datetime + timedelta(minutes=time_interval)).time()

        # Filter the dataframe for entries within the current time interval
        _df = df.between_time(_start_time, _end_time, inclusive="left")

        # Calculate and store the arrival rate for the interval
        arrival_rates_dict[_start_time] = _df.shape[0] / num_days

        # Move to the next interval
        _start_datetime = _start_datetime + timedelta(minutes=time_interval)

    return arrival_rates_dict


def get_specialty_probs(
    model_file_path,
    snapshots_df,
    special_category_func=None,
    special_category_dict=None,
):
    """
    Calculate specialty probability distributions for patient visits based on their data.

    This function applies a predictive model to each row of the input DataFrame to compute
    specialty probability distributions. Optionally, it can classify certain rows as
    belonging to a special category (like pediatric cases) based on a user-defined function,
    applying a fixed probability distribution for these cases.

    Parameters
    ----------
    model_file_path : str
        Path to the predictive model file.
    snapshots_df : pandas.DataFrame
        DataFrame containing the data on which predictions are to be made. Must include
        a 'consultation_sequence' column if no special_category_func is applied.
    special_category_func : callable, optional
        A function that takes a DataFrame row (Series) as input and returns True if the row
        belongs to a special category that requires a fixed probability distribution.
        If not provided, no special categorization is applied.
    special_category_dict : dict, optional
        A dictionary containing the fixed probability distribution for special category cases.
        This dictionary is applied to rows identified by `special_category_func`. If
        `special_category_func` is provided, this parameter must also be provided.

    Returns
    -------
    pandas.Series
        A Series containing dictionaries as values. Each dictionary represents the probability
        distribution of specialties for each patient visit.

    Raises
    ------
    ValueError
        If `special_category_func` is provided but `special_category_dict` is None.


    """
    if special_category_func and not special_category_dict:
        raise ValueError(
            "special_category_dict must be provided if special_category_func is specified."
        )

    # Load model for specialty predictions
    specialty_model = prepare_for_inference(
        model_file_path, "ed_specialty", model_only=True
    )

    # Function to determine the specialty probabilities
    def determine_specialty(row):
        if special_category_func and special_category_func(row):
            return special_category_dict
        else:
            return specialty_model.predict(row["consultation_sequence"])

    # Apply the determine_specialty function to each row
    specialty_prob_series = snapshots_df.apply(determine_specialty, axis=1)

    # Find all unique keys used in any dictionary within the series
    all_keys = set().union(
        *(d.keys() for d in specialty_prob_series if isinstance(d, dict))
    )

    # Ensure each dictionary contains all keys found, with default values of 0 for missing keys
    specialty_prob_series = specialty_prob_series.apply(
        lambda d: (
            {key: d.get(key, 0) for key in all_keys} if isinstance(d, dict) else d
        )
    )

    return specialty_prob_series
