"""
Module for calculating specialty probability distributions for patient visits.

This module provides a function `get_specialty_probs` that leverages a predictive model
to compute specialty probability distributions for patient visits based on their data.
It supports custom classification of certain visits into special categories that can
have predefined probability distributions.

Functions
---------
get_specialty_probs(model_file_path, snapshots_df, special_category_func=None, special_category_dict=None)
    Calculate specialty probability distributions for patient visits based on their data.
"""

from prepare import prepare_for_inference


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
    >>> snapshots_df = pd.DataFrame({
    ...     'consultation_sequence': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    ...     'age': [5, 40, 70]
    ... })
    >>> def pediatric_case(row):
    ...     return row['age'] < 18
    >>> special_dist = {'pediatrics': 0.9, 'general': 0.1}
    >>> get_specialty_probs('model.pkl', snapshots_df, pediatric_case, special_dist)
    0    {'pediatrics': 0.9, 'general': 0.1}
    1    {'cardiology': 0.7, 'general': 0.3}
    2    {'neurology': 0.8, 'general': 0.2}
    dtype: object
    """

    # Convert consultation_sequence to tuple if not already a tuple
    if not isinstance(snapshots_df["consultation_sequence"].iloc[0], tuple):
        snapshots_df.loc[:, "consultation_sequence"] = snapshots_df["consultation_sequence"].apply(lambda x: tuple(x) if x else ())


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
