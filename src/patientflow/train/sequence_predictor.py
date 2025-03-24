from pandas import DataFrame

from patientflow.prepare import (
    select_one_snapshot_per_visit,
    create_special_category_objects,
)
from patientflow.predictors.sequence_predictor import SequencePredictor


def get_default_visits(admitted: DataFrame) -> DataFrame:
    """
    Filters a dataframe of patient visits to include only non-pediatric patients.

    This function identifies and removes pediatric patients from the dataset based on
    both age criteria and specialty assignment. It automatically detects the appropriate
    age column format from the provided dataframe.

    Parameters:
    -----------
    admitted : DataFrame
        A pandas DataFrame containing patient visit information. Must include either
        'age_on_arrival' or 'age_group' columns, and a 'specialty' column.

    Returns:
    --------
    DataFrame
        A filtered DataFrame containing only non-pediatric patients (adults).

    Notes:
    ------
    The function automatically detects which age-related columns are present in the
    dataframe and configures the appropriate filtering logic. It removes patients who
    are either:
    1. Identified as pediatric based on age criteria, or
    2. Assigned to a pediatric specialty

    Examples:
    ---------
    >>> adult_visits = get_default_visits(all_patient_visits)
    >>> print(f"Reduced from {len(all_patient_visits)} to {len(adult_visits)} adult visits")
    """
    # Get configuration for categorizing patients based on age columns
    special_params = create_special_category_objects(admitted.columns)

    # Extract function that identifies non-pediatric patients
    opposite_special_category_func = special_params["special_func_map"]["default"]

    # Determine which category is the special category (should be "paediatric")
    special_category_key = next(
        key
        for key, value in special_params["special_category_dict"].items()
        if value == 1.0
    )

    # Filter out pediatric patients based on both age criteria and specialty
    filtered_admitted = admitted[
        admitted.apply(opposite_special_category_func, axis=1)
        & (admitted["specialty"] != special_category_key)
    ]

    return filtered_admitted


def train_sequence_predictor(
    train_visits: DataFrame,
    model_name: str,
    visit_col: str,
    input_var: str,
    grouping_var: str,
    outcome_var: str,
) -> SequencePredictor:
    """Train a specialty prediction model.

    Args:
        train_visits: Training data containing visit information
        model_name: Name identifier for the model
        visit_col: Column name containing visit identifiers
        input_var: Column name for input sequence
        grouping_var: Column name for grouping sequence
        outcome_var: Column name for target variable

    Returns:
        Trained SequencePredictor model
    """
    visits_single = select_one_snapshot_per_visit(train_visits, visit_col)
    admitted = visits_single[
        (visits_single.is_admitted) & ~(visits_single.specialty.isnull())
    ]
    filtered_admitted = get_default_visits(admitted)

    filtered_admitted.loc[:, input_var] = filtered_admitted[input_var].apply(
        lambda x: tuple(x) if x else ()
    )
    filtered_admitted.loc[:, grouping_var] = filtered_admitted[grouping_var].apply(
        lambda x: tuple(x) if x else ()
    )

    spec_model = SequencePredictor(
        input_var=input_var,
        grouping_var=grouping_var,
        outcome_var=outcome_var,
    )
    spec_model.fit(filtered_admitted)

    return spec_model
