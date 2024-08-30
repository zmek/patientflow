from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from prepare import (
    get_specialty_probs,
    prepare_for_inference,
)
from predict.emergency_demand.from_individual_probs import (
    model_input_to_pred_proba,
    pred_proba_to_pred_demand,
)
from predict.emergency_demand.admission_in_prediction_window_using_aspirational_curve import (
    calculate_probability,
)

from ed_specialty_paediatric_functions import validate_special_category_objects


def add_missing_columns(pipeline, df):
    # check input data for missing columns
    column_transformer = pipeline.named_steps["feature_transformer"]

    # Function to get feature names before one-hot encoding
    def get_feature_names_before_encoding(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers:
            if isinstance(transformer, OneHotEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, OrdinalEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, StandardScaler):
                feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        return feature_names

    feature_names_before_encoding = get_feature_names_before_encoding(
        column_transformer
    )

    added_columns = []
    for missing_col in set(feature_names_before_encoding).difference(set(df.columns)):
        if missing_col.startswith(("lab_orders_", "visited_", "has_")):
            df[missing_col] = False
        elif missing_col.startswith(("num_", "total_")):
            df[missing_col] = 0
        elif missing_col.startswith("latest_"):
            df[missing_col] = pd.NA
        elif missing_col == "arrival_method":
            df[missing_col] = "None"
        else:
            df[missing_col] = pd.NA
        added_columns.append(missing_col)

    if added_columns:
        print(
            f"Warning: The following columns were used in training, but not found in the real-time data. These have been added to the dataframe: {', '.join(added_columns)}"
        )

    return df


def index_of_sum(sequence: List[float], max_sum: float) -> int:
    """Returns the index where the cumulative sum of a sequence of probabilities exceeds max_sum."""
    cumulative_sum = 0.0
    for i, value in enumerate(sequence):
        cumulative_sum += value
        if cumulative_sum >= 1 - max_sum:
            return i
    return len(sequence) - 1  # Return the last index if the sum doesn't exceed max_sum


def create_predictions(
    model_file_path: str,
    prediction_time: Tuple,
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window_hrs: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: List[float],
    special_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Create predictions for emergency demand for a single prediction moment.

    Parameters:
    - model_file_path (str): Path to the model files.
    - prediction_moment (Tuple): Hour and minute of time for which model to be used for inference was trained
    - prediction_snapshots (pd.DataFrame): DataFrame containing prediction snapshots.
    - specialties (List[str]): List of specialty names for which predictions are required.
    - prediction_window_hrs (float): Prediction window in hours.
    - x1, y1, x2, y2 (float): Parameters for calculating probability of admission within prediction window.
    - cdf_cut_points (List[float]): List of cumulative distribution function cut points.
    - special_params (Optional[Dict[str, Any]]): Dictionary containing 'special_category_func', 'special_category_dict', and 'special_func_map'.
      - special_category_func (Callable[[Any], Any]): Function identifying patients whose specialty predictions are handled outside the get_specialty_probs() function.
      - special_category_dict (Dict[str, Any]): Dictionary of probabilities applied to those patients.
      - special_func_map (Dict[str, Callable[[pd.Series], bool]]): A dictionary mapping specialties to specific functions that are applied to each row of the prediction snapshots to filter indices.

    Returns:
    - Dict[str, Dict[str, List[int]]]: Predictions for each specialty.

    Example:
    ```python
    from datetime import datetime
    import pandas as pd

    special_category_dict = {
        'medical': 0.0,
        'surgical': 0.0,
        'haem/onc': 0.0,
        'paediatric': 1.0
    }

    # Function to determine if the patient is a child
    special_category_func = lambda row: row['age_on_arrival'] < 18

    special_func_map = {
        'paediatric': special_category_func,
        'default': lambda row: True  # Default function for other specialties
    }

    prediction_time = (15,30)
    prediction_snapshots = pd.DataFrame([
        {'age_on_arrival': 15, 'elapsed_los': 3600},
        {'age_on_arrival': 45, 'elapsed_los': 7200}
    ])
    specialties = ['paediatric', 'medical']
    prediction_window_hrs = 4.0
    cdf_cut_points = [0.7, 0.9]
    x1, y1, x2, y2 = 4.0, 0.76, 12.0, 0.99

    predictions = create_predictions(
        model_file_path='path/to/model/file',
        prediction_time=prediction_time,
        prediction_snapshots=prediction_snapshots,
        specialties=specialties,
        prediction_window_hrs=prediction_window_hrs,
        cdf_cut_points=cdf_cut_points,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        special_func_map=special_func_map,
        special_category_dict=special_category_dict,
        special_category_func=special_category_func


    )
    ```
    """

    if special_params:
        validate_special_category_objects(special_params)
        special_category_func = special_params["special_category_func"]
        special_category_dict = special_params["special_category_dict"]
        special_func_map = special_params["special_func_map"]
    else:
        special_category_func = special_category_dict = special_func_map = None

    predictions: Dict[str, Dict[str, List[int]]] = {
        specialty: {"in_ed": [], "yet_to_arrive": []} for specialty in specialties
    }

    # Load models
    admissions_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name="ed_admission",
        prediction_time=prediction_time,
        model_only=True,
    )

    # add missing columns to predictions_snapshots that are expected by the model
    prediction_snapshots = add_missing_columns(admissions_model, prediction_snapshots)

    yet_to_arrive_model_name = (
        f"ed_yet_to_arrive_by_spec_{int(prediction_window_hrs)}_hours"
    )
    yet_to_arrive_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name=yet_to_arrive_model_name,
        model_only=True,
    )

    # Get predictions of admissions for ED patients
    prob_admission_after_ed = model_input_to_pred_proba(
        prediction_snapshots, admissions_model
    )

    # Get predictions of admission to specialty
    prediction_snapshots["specialty_prob"] = get_specialty_probs(
        model_file_path,
        prediction_snapshots,
        special_category_func=special_category_func,
        special_category_dict=special_category_dict,
    )

    prediction_snapshots["elapsed_los_hrs"] = prediction_snapshots["elapsed_los"] / 3600

    # Get probability of admission within prediction window
    prob_admission_in_window = prediction_snapshots.apply(
        lambda row: calculate_probability(
            row["elapsed_los_hrs"], prediction_window_hrs, x1, y1, x2, y2
        ),
        axis=1,
    )

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}

    for specialty in specialties:
        func = special_func_map.get(specialty, special_func_map["default"])
        non_zero_indices = prediction_snapshots[
            prediction_snapshots.apply(func, axis=1)
        ].index

        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]
        prob_admission_to_specialty = prediction_snapshots["specialty_prob"].apply(
            lambda x: x[specialty]
        )
        filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[
            non_zero_indices
        ]
        filtered_prob_admission_in_window = prob_admission_in_window.loc[
            non_zero_indices
        ]

        filtered_weights = (
            filtered_prob_admission_to_specialty * filtered_prob_admission_in_window
        )

        pred_demand_in_ed = pred_proba_to_pred_demand(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )
        prediction_context = {specialty: {"prediction_time": prediction_time}}
        pred_demand_yta = yet_to_arrive_model.predict(
            prediction_context, x1, y1, x2, y2
        )

        predictions[specialty]["in_ed"] = [
            index_of_sum(pred_demand_in_ed["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
        predictions[specialty]["yet_to_arrive"] = [
            index_of_sum(
                pred_demand_yta[specialty]["agg_proba"].values.cumsum(), cut_point
            )
            for cut_point in cdf_cut_points
        ]

    return predictions
