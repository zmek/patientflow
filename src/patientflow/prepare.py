import pandas as pd
from load import data_from_csv, load_saved_model



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
):
    # retrieve model trained for this time of day
    model = load_saved_model(model_file_path, model_name, prediction_time)

    if model_only:
        return model

    if data_path:
        df = data_from_csv(data_path, index_column = 'snapshot_id')
    elif df is None or df.empty:
        print("Please supply a dataset if not passing a data path")
        return None

    test_df = (
        df[df.training_validation_test == "test"]
        .drop(columns="training_validation_test")
        .copy()
    )

    # print("Prep for inference - test_df")
    # print(test_df.index)

    exclude_from_training_data = [
        "visit_number",
        "snapshot_date",
        "prediction_time",
    ]

    X_test, y_test = get_snapshots_at_prediction_time(
        test_df,
        prediction_time,
        exclude_from_training_data,
        single_snapshot_per_visit,
    )

    # print("Prep for inference - X_test")
    # print(X_test.index)
    # print("Prep for inference - y_test")
    # print(y_test.index)

    return X_test, y_test, model


def prepare_snapshots_dict(df, start_dt=None, end_dt=None):
    """
    Prepares a dictionary mapping horizon dates to their corresponding snapshot indices.

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
