from ed_admissions_utils import preprocess_data, load_saved_model
from ed_admissions_data_retrieval import ed_admissions_get_data


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
        df = ed_admissions_get_data(data_path)
    elif df is None or df.empty:
        print("Please supply a dataset if not passing a data path")
        return

    # print("Prep for inference - df")
    # print(df[(df.training_validation_test == 'test')].index)

    if df.index.name != "snapshot_id":
        df = df.set_index("snapshot_id")

    test_df = (
        df[df.training_validation_test == "test"]
        .drop(columns="training_validation_test")
        .copy()
    )

    # print("Prep for inference - test_df")
    # print(test_df.index)

    exclude_from_training_data = [
        "visit_number",
        "snapshot_datetime",
        "prediction_time",
    ]

    X_test, y_test = preprocess_data(
        test_df, prediction_time, exclude_from_training_data, single_snapshot_per_visit
    )

    # print("Prep for inference - X_test")
    # print(X_test.index)
    # print("Prep for inference - y_test")
    # print(y_test.index)

    return X_test, y_test, model


def prepare_snapshots_dict(df):
    """
    Prepares a dictionary mapping horizon dates to their corresponding snapshot indices.

    Args:
    df (pd.DataFrame): DataFrame containing at least a 'snapshot_datetime' column which represents the dates.

    Returns:
    dict: A dictionary where keys are dates and values are arrays of indices corresponding to each date's snapshots.
    """
    # Ensure 'snapshot_datetime' is in the DataFrame
    if "snapshot_datetime" not in df.columns:
        raise ValueError("DataFrame must include a 'snapshot_datetime' column")

    # Group the DataFrame by 'snapshot_datetime' and collect the indices for each group
    snapshots_dict = {
        date: group.index.tolist() for date, group in df.groupby("snapshot_datetime")
    }

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

    Examples
    --------
    >>> model_file_path = 'path/to/model'
    >>> data = {'age_group': ['0-17', '18-65', '66+'], 'age': [16, 40, 70], 'consultation_sequence': [1, 2, 3]}
    >>> snapshots_df = pd.DataFrame(data)
    >>> special_category_func = lambda row: row['age'] <= 17
    >>> special_category_dict = {'medical': 0.0, 'surgical': 0.0, 'haem_onc': 0.0, 'paediatric': 1.0}
    >>> probs = get_specialty_probs(model_file_path, snapshots_df, special_category_func, special_category_dict)
    >>> print(probs)
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
        lambda d: {key: d.get(key, 0) for key in all_keys} if isinstance(d, dict) else d
    )

    return specialty_prob_series
