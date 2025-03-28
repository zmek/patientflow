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

get_snapshots_at_prediction_time(df, prediction_time, exclude_columns, single_snapshot_per_visit=True)
    Filters the DataFrame by prediction time and optionally selects one snapshot per visit.

prepare_group_snapshot_dict(df, start_dt=None, end_dt=None)
    Prepares a dictionary mapping snapshot dates to their corresponding snapshot indices.

calculate_time_varying_arrival_rates(df, yta_time_interval)
    Calculates the time-varying arrival rates for a dataset indexed by datetime.
"""

import pandas as pd
import numpy as np
import random
from patientflow.load import load_saved_model, get_dict_cols, data_from_csv
from datetime import datetime, date


from typing import Tuple, List, Set, Dict, Any

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


def apply_set(row: pd.Series) -> str:
    return random.choices(
        ["train", "valid", "test"],
        weights=[row.training_set, row.validation_set, row.test_set],
    )[0]


def assign_patient_ids(
    df: pd.DataFrame,
    start_training_set: date,
    start_validation_set: date,
    start_test_set: date,
    end_test_set: date,
    date_col: str = "arrival_datetime",
    patient_id: str = "mrn",
    visit_col: str = "encounter",
) -> pd.DataFrame:
    """Probabilistically assign patient IDs to train/validation/test sets.

    Args:
        df: DataFrame with patient_id, encounter, and temporal columns
        start_training_set: Start date for training period
        start_validation_set: Start date for validation period
        start_test_set: Start date for test period
        end_test_set: End date for test period
        date_col: Column name for temporal splitting
        patient_id: Column name for patient identifier (default: 'mrn')
        visit_col: Column name for visit identifier (default: 'encounter')

    Returns:
        DataFrame with patient ID assignments based on weighted random sampling

    Notes:
        - Counts encounters in each time period per patient ID
        - Randomly assigns each patient ID to one set, weighted by their temporal distribution
        - Patient with 70% encounters in training, 30% in validation has 70% chance of training assignment
    """
    patients: pd.DataFrame = (
        df.groupby([patient_id, visit_col])[date_col].max().reset_index()
    )

    # Handle date_col as string, datetime, or date type
    if pd.api.types.is_datetime64_any_dtype(patients[date_col]):
        # Already datetime, extract date if needed
        if hasattr(patients[date_col].iloc[0], "date"):
            date_series = patients[date_col].dt.date
        else:
            # Already date type
            date_series = patients[date_col]
    else:
        # Try to convert string to datetime
        try:
            patients[date_col] = pd.to_datetime(patients[date_col])
            date_series = patients[date_col].dt.date
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not convert column '{date_col}' to datetime format: {str(e)}"
            )

    # Filter out patient IDs outside temporal bounds
    pre_training_patients = patients[date_series < start_training_set]
    post_test_patients = patients[date_series >= end_test_set]

    if len(pre_training_patients) > 0:
        print(
            f"Filtered out {len(pre_training_patients)} patients with only pre-training visits"
        )
    if len(post_test_patients) > 0:
        print(
            f"Filtered out {len(post_test_patients)} patients with only post-test visits"
        )

    valid_patients = patients[
        (date_series >= start_training_set) & (date_series < end_test_set)
    ]
    patients = valid_patients

    # Use the date_series for set assignment
    patients["training_set"] = (date_series >= start_training_set) & (
        date_series < start_validation_set
    )
    patients["validation_set"] = (date_series >= start_validation_set) & (
        date_series < start_test_set
    )
    patients["test_set"] = (date_series >= start_test_set) & (
        date_series < end_test_set
    )

    patients = patients.groupby(patient_id)[
        ["training_set", "validation_set", "test_set"]
    ].sum()
    patients["training_validation_test"] = patients.apply(apply_set, axis=1)

    print(
        f"\nPatient Set Overlaps (before random assignment):"
        f"\nTrain-Valid: {patients[patients.training_set * patients.validation_set != 0].shape[0]} of {patients[patients.training_set + patients.validation_set > 0].shape[0]}"
        f"\nValid-Test: {patients[patients.validation_set * patients.test_set != 0].shape[0]} of {patients[patients.validation_set + patients.test_set > 0].shape[0]}"
        f"\nTrain-Test: {patients[patients.training_set * patients.test_set != 0].shape[0]} of {patients[patients.training_set + patients.test_set > 0].shape[0]}"
        f"\nAll Sets: {patients[patients.training_set * patients.validation_set * patients.test_set != 0].shape[0]} of {patients.shape[0]} total patients"
    )

    return patients


def create_temporal_splits(
    df: pd.DataFrame,
    start_train: date,
    start_valid: date,
    start_test: date,
    end_test: date,
    col_name: str = "arrival_datetime",
    patient_id: str = "mrn",
    visit_col: str = "encounter",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into temporal train/validation/test sets.

    Creates temporal data splits using primary datetime column and optional snapshot dates.
    Handles patient ID grouping if present to prevent data leakage.

    Args:
        df: Input dataframe
        start_train: Training start (inclusive)
        start_valid: Validation start (inclusive)
        start_test: Test start (inclusive)
        end_test: Test end (exclusive)
        col_name: Primary datetime column for splitting
        patient_id: Column name for patient identifier (default: 'mrn')
        visit_col: Column name for visit identifier (default: 'encounter')

    Returns:
        tuple: (train_df, valid_df, test_df) Split dataframes
    """

    def get_date_value(series: pd.Series) -> pd.Series:
        """Convert timestamp or date column to date, handling both types"""
        try:
            return pd.to_datetime(series).dt.date
        except (AttributeError, TypeError):
            return series

    if patient_id in df.columns:
        set_assignment: pd.DataFrame = assign_patient_ids(
            df,
            start_train,
            start_valid,
            start_test,
            end_test,
            col_name,
            patient_id,
            visit_col,
        )
        patient_sets: Dict[str, Set] = {
            k: set(set_assignment[set_assignment.training_validation_test == v].index)
            for k, v in {"train": "train", "valid": "valid", "test": "test"}.items()
        }

    splits: List[pd.DataFrame] = []
    for start, end, set_key in [
        (start_train, start_valid, "train"),
        (start_valid, start_test, "valid"),
        (start_test, end_test, "test"),
    ]:
        mask = (get_date_value(df[col_name]) >= start) & (
            get_date_value(df[col_name]) < end
        )

        if "snapshot_date" in df.columns:
            mask &= (get_date_value(df.snapshot_date) >= start) & (
                get_date_value(df.snapshot_date) < end
            )

        if patient_id in df.columns:
            mask &= df[patient_id].isin(patient_sets[set_key])

        splits.append(df[mask].copy())

    print(f"Split sizes: {[len(split) for split in splits]}")
    return tuple(splits)


class SpecialCategoryParams:
    """
    A picklable implementation of special category parameters for patient classification.

    This class identifies pediatric patients based on available age-related columns
    in the dataset and provides functions to categorize patients accordingly.
    It's designed to be serializable with pickle by implementing the __reduce__ method.

    Attributes:
        columns (list): List of column names from the dataset
        method_type (str): The method used for age detection ('age_on_arrival' or 'age_group')
        special_category_dict (dict): Default category values mapping
    """

    def __init__(self, columns):
        """
        Initialize the SpecialCategoryParams object.

        Parameters:
            columns (list or pandas.Index): Column names from the dataset
                used to determine the appropriate age identification method

        Raises:
            ValueError: If neither 'age_on_arrival' nor 'age_group' columns are found
        """
        self.columns = columns
        self.special_category_dict = {
            "medical": 0.0,
            "surgical": 0.0,
            "haem/onc": 0.0,
            "paediatric": 1.0,
        }

        if "age_on_arrival" in columns:
            self.method_type = "age_on_arrival"
        elif "age_group" in columns:
            self.method_type = "age_group"
        else:
            raise ValueError("Unknown data format: could not find expected age columns")

    def special_category_func(self, row):
        """
        Identify if a patient is pediatric based on age data.

        Parameters:
            row (dict or pandas.Series): A row of patient data containing either
                'age_on_arrival' or 'age_group'

        Returns:
            bool: True if the patient is pediatric (age < 18 or age_group is '0-17'),
                 False otherwise
        """
        if self.method_type == "age_on_arrival":
            return row["age_on_arrival"] < 18
        else:  # age_group
            return row["age_group"] == "0-17"

    def opposite_special_category_func(self, row):
        """
        Identify if a patient is NOT pediatric.

        Parameters:
            row (dict or pandas.Series): A row of patient data

        Returns:
            bool: True if the patient is NOT pediatric, False if they are pediatric
        """
        return not self.special_category_func(row)

    def get_params_dict(self):
        """
        Get the special parameter dictionary in the format expected by the application.

        Returns:
            dict: A dictionary containing:
                - 'special_category_func': Function to identify pediatric patients
                - 'special_category_dict': Default category values
                - 'special_func_map': Mapping of category names to detection functions
        """
        return {
            "special_category_func": self.special_category_func,
            "special_category_dict": self.special_category_dict,
            "special_func_map": {
                "paediatric": self.special_category_func,
                "default": self.opposite_special_category_func,
            },
        }

    def __reduce__(self):
        """
        Support for pickle serialization.

        Returns:
            tuple: A tuple containing:
                - The class itself (to be called as a function)
                - A tuple of arguments to pass to the class constructor
        """
        return (self.__class__, (self.columns,))


def create_special_category_objects(columns):
    """
    Creates a configuration for categorizing patients with special handling for pediatric cases.

    Parameters:
    -----------
    columns : list or pandas.Index
        The column names available in the dataset. Used to determine which age format is present.

    Returns:
    --------
    dict
        A dictionary containing special category configuration.
    """
    # Create the class instance and return its parameter dictionary
    params_obj = SpecialCategoryParams(columns)
    return params_obj.get_params_dict()


def validate_special_category_objects(special_params: Dict[str, Any]) -> None:
    required_keys = [
        "special_category_func",
        "special_category_dict",
        "special_func_map",
    ]
    missing_keys = [key for key in required_keys if key not in special_params]

    if missing_keys:
        raise MissingKeysError(missing_keys)


def create_yta_filters(df):
    """
    Create specialty filters for categorizing patients by specialty and age group.

    This function generates a dictionary of filters based on specialty categories,
    with special handling for pediatric patients. It uses the SpecialCategoryParams
    class to determine which specialties correspond to pediatric care.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing patient data with columns that include either
        'age_on_arrival' or 'age_group' for pediatric classification

    Returns:
    --------
    dict
        A dictionary mapping specialty names to filter configurations.
        Each configuration contains:
        - For pediatric specialty: {"is_child": True}
        - For other specialties: {"specialty": specialty_name, "is_child": False}

    Examples:
    ---------
    >>> df = pd.DataFrame({'patient_id': [1, 2], 'age_on_arrival': [10, 40]})
    >>> filters = create_yta_filters(df)
    >>> print(filters['paediatric'])
    {'is_child': True}
    >>> print(filters['medical'])
    {'specialty': 'medical', 'is_child': False}
    """
    # Get the special category parameters using the picklable implementation
    special_params = create_special_category_objects(df.columns)

    # Extract necessary data from the special_params
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
    df,
    prediction_time,
    exclude_columns=[],
    single_snapshot_per_visit=True,
    visit_col=None,
    label_col="is_admitted",
):
    """
    Get snapshots of data at a specific prediction time with configurable visit and label columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    prediction_time : str or datetime
        The specific prediction time to filter for
    exclude_columns : list
        List of columns to exclude from the final DataFrame
    single_snapshot_per_visit : bool, default=True
        Whether to select only one snapshot per visit. If True, visit_col must be provided.
    visit_col : str, optional
        Name of the column containing visit identifiers. Required if single_snapshot_per_visit is True.
    label_col : str, default="is_admitted"
        Name of the column containing the target labels

    Returns:
    --------
    tuple(pandas.DataFrame, pandas.Series)
        Processed DataFrame and corresponding labels

    Raises:
    -------
    ValueError
        If single_snapshot_per_visit is True but visit_col is not provided
    """
    if single_snapshot_per_visit and visit_col is None:
        raise ValueError(
            "visit_col must be provided when single_snapshot_per_visit is True"
        )

    # Filter by the time of day while keeping the original index
    df_tod = df[df["prediction_time"] == prediction_time].copy()

    if single_snapshot_per_visit:
        # Select one row for each visit
        df_single = select_one_snapshot_per_visit(df_tod, visit_col)
        # Create label array with the same index
        y = df_single.pop(label_col).astype(int)
        # Drop specified columns and ensure we do not reset the index
        df_single.drop(columns=exclude_columns, inplace=True)
        return df_single, y
    else:
        # Directly modify df_tod without resetting the index
        df_tod.drop(
            columns=["random_number"] + exclude_columns, inplace=True, errors="ignore"
        )
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
    If available, it will use the calibrated pipeline instead of the regular pipeline.

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
        The loaded model (calibrated pipeline if available, otherwise regular pipeline).
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
    - Either `df` or `data_path` must be provided. If neither is provided or if `df`
      is empty, the function will print an error message and return None.
    - The function will automatically use a calibrated pipeline if one is available
      in the model, otherwise it will fall back to the regular pipeline.
    """

    # retrieve model trained for this time of day
    model = load_saved_model(model_file_path, model_name, prediction_time)

    # Use calibrated pipeline if available, otherwise use regular pipeline
    if hasattr(model, "calibrated_pipeline") and model.calibrated_pipeline is not None:
        pipeline = model.calibrated_pipeline
    else:
        pipeline = model.pipeline

    if model_only:
        return pipeline

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

    return X_test, y_test, pipeline


def prepare_group_snapshot_dict(df, start_dt=None, end_dt=None):
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

    # Filter DataFrame to date range if provided
    filtered_df = df.copy()
    if start_dt and end_dt:
        filtered_df = df[
            (df["snapshot_date"] >= start_dt) & (df["snapshot_date"] < end_dt)
        ]

    # Group the DataFrame by 'snapshot_date' and collect the indices for each group
    snapshots_dict = {
        date: group.index.tolist()
        for date, group in filtered_df.groupby("snapshot_date")
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
