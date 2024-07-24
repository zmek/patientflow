from typing import List, Callable, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from ed_admissions_helper_functions import (
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
    prediction_moment: datetime,
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window_hrs: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: List[float],
    special_category_func: Optional[Callable[[Any], Any]] = None,
    special_category_dict: Optional[Dict[str, Any]] = None,
    special_func_map: Optional[Dict[str, Callable[[pd.Series], bool]]] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Create predictions for emergency demand for a single prediction moment.

    Parameters:
    - model_file_path (str): Path to the model files.
    - prediction_moment (datetime): Datetime of predition 
    - prediction_snapshots (pd.DataFrame): DataFrame containing prediction snapshots.
    - specialties (List[str]): List of specialty names for which predictions are required.
    - prediction_window_hrs (float): Prediction window in hours.
    - x1, y1, x2, y2 (float): Parameters for calculating probability of admission within prediction window.
    - cdf_cut_points (List[float]): List of cumulative distribution function cut points.
    - special_category_func (Optional[Callable[[Any], Any]]): Function identifying patients whose specialty predictions are handled outside the get_specialty_probs() function.
    - special_category_dict (Optional[Dict[str, Any]]): Dictionary of probabilities applied to those patients
    - special_func_map (Optional[Dict[str, Callable[[pd.Series], bool]]]): A dictionary mapping specialties to specific functions that are applied to each row of the prediction snapshots to filter indices

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

    prediction_moment = datetime.now()
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
        prediction_moment=prediction_moment,
        prediction_snapshots=prediction_snapshots,
        specialties=specialties,
        prediction_window_hrs=prediction_window_hrs,
        cdf_cut_points=cdf_cut_points,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        special_func_map=special_func_map,
    )
    ```
    """
    prediction_time = (prediction_moment.hour, prediction_moment.minute)
    predictions = {specialty: {"in_ed": [], "yet_to_arrive": []} for specialty in specialties}

    # Load models
    admissions_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name="ed_admission",
        prediction_time=prediction_time,
        model_only=True,
    )

    yet_to_arrive_model_name = f"ed_yet_to_arrive_by_spec_{int(prediction_window_hrs)}_hours"
    yet_to_arrive_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name=yet_to_arrive_model_name,
        model_only=True,
    )

    # Get predictions of admissions for ED patients
    prob_admission_after_ed = model_input_to_pred_proba(prediction_snapshots, admissions_model)

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
        lambda row: calculate_probability(row["elapsed_los_hrs"], prediction_window_hrs, x1, y1, x2, y2),
        axis=1,
    )

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}

    for specialty in specialties:
        func = special_func_map.get(specialty, special_func_map["default"])
        non_zero_indices = prediction_snapshots[prediction_snapshots.apply(func, axis=1)].index

        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]
        prob_admission_to_specialty = prediction_snapshots["specialty_prob"].apply(lambda x: x[specialty])
        filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[non_zero_indices]
        filtered_prob_admission_in_window = prob_admission_in_window.loc[non_zero_indices]

        filtered_weights = filtered_prob_admission_to_specialty * filtered_prob_admission_in_window

        pred_demand_in_ed = pred_proba_to_pred_demand(filtered_prob_admission_after_ed, weights=filtered_weights)
        prediction_context = {specialty: {"prediction_time": prediction_time}}
        pred_demand_yta = yet_to_arrive_model.predict(prediction_context, x1, y1, x2, y2)

        predictions[specialty]["in_ed"] = [
            index_of_sum(pred_demand_in_ed["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
        predictions[specialty]["yet_to_arrive"] = [
            index_of_sum(pred_demand_yta[specialty]["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]

    return predictions
