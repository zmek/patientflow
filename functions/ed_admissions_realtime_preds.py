from typing import List, Callable, Dict, Any, Tuple, Optional
from ed_admissions_helper_functions import (
    get_specialty_probs,
    prepare_for_inference,
)
from datetime import datetime
import pandas as pd

from predict.emergency_demand.from_individual_probs import (
    model_input_to_pred_proba,
    pred_proba_to_pred_demand,
)
from predict.emergency_demand.admission_in_prediction_window_using_aspirational_curve import (
    calculate_probability,
)


def index_of_sum(sequence: List[float], max_sum: float) -> int:
    s = 0.0
    for i, p in enumerate(sequence):
        s += p
        if s >= 1 - max_sum:  ## only this line has changed
            return i
    return i


def create_predictions(
    model_file_path: str,
    snapshot_datetime: datetime,
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window_hrs: float,
    cdf_cut_points: List[float],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    special_category_func: Optional[Callable[[Any], Any]] = None,
    special_category_dict: Optional[Dict[str, Any]] = None,
    special_func_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, List[int]]]:  # [SpecialtyType, DemandPredictions]:
    # initialisation
    hour = snapshot_datetime.hour
    minute = snapshot_datetime.minute
    prediction_time: Tuple[int, int] = (hour, minute)

    # initialise predictions dict
    predictions: Dict[str, Dict[str, List[int]]] = {
        key: {subkey: [] for subkey in ["in_ed", "yet_to_arrive"]}
        for key in specialties
    }

    # load models
    admissions_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name="ed_admission",
        prediction_time=prediction_time,
        model_only=True,
    )

    yet_to_arrive_model_name = (
        "ed_yet_to_arrive_by_spec_" + str(int(prediction_window_hrs)) + "_hours"
    )

    yet_to_arrive_model = prepare_for_inference(
        model_file_path=model_file_path,
        model_name=yet_to_arrive_model_name,
        model_only=True,
    )

    # get predictions of admissions for ED patients
    prob_admission_after_ed = model_input_to_pred_proba(
        prediction_snapshots, admissions_model
    )

    # get predictions of admission to specialty
    prediction_snapshots["specialty_probs"] = get_specialty_probs(
        model_file_path,
        prediction_snapshots,
        special_category_func=special_category_func,
        special_category_dict=special_category_dict,
    )

    # get probability of admission in prediction window
    prediction_snapshots["elapsed_los_hrs"] = prediction_snapshots["elapsed_los"] / 3600

    # get probability of admission within prediction window
    prob_admission_in_window = prediction_snapshots.apply(
        lambda row: calculate_probability(
            row["elapsed_los_hrs"], prediction_window_hrs, x1, y1, x2, y2
        ),
        axis=1,
    )

    for _spec in specialties:
        func = special_func_map.get(_spec, special_func_map["default"])

        # Apply the function to filter indices
        non_zero_indices = prediction_snapshots[
            prediction_snapshots.apply(func, axis=1)
        ].index.values

        # Filter the weights and the patients to exclude any with zero probability,
        # before calling the function to generate a distribution over bed numbers
        # Where prob_admission_to_specialty are equal to zero, this is either because
        # a child has zero probability of being admitted to an adult ward, or vice versa
        # These non-standard cases should not be included in the overall bed counts in such cases

        # non_zero_indices = prob_admission_to_specialty != 0
        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]

        prob_admission_to_specialty = prediction_snapshots["specialty_prob"].apply(
            lambda x: x["medical"]
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

        # Call the function to predict demand for patients in ED, with the filtered data
        pred_demand_in_ED = pred_proba_to_pred_demand(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )

        # Process patients yet-to-arrive
        prediction_context = {_spec: {"prediction_time": prediction_time}}
        pred_demand_yta = yet_to_arrive_model.predict(
            prediction_context, x1, y1, x2, y2
        )

        # Return the distributions at the desired cut points
        predictions[_spec]["in_ed"] = [
            index_of_sum(pred_demand_in_ED["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
        predictions[_spec]["yet_to_arrive"] = [
            index_of_sum(pred_demand_yta[_spec]["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
    return predictions
