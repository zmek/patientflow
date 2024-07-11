from predict.emergency_demand.from_individual_probs import (
    model_input_to_pred_proba,
    pred_proba_to_pred_demand,
)
from ed_admissions_helper_functions import prepare_for_inference
from ed_admissions_utils import load_saved_model

from ed_admissions_helper_functions import get_specialty_probs


def index_of_sum(sequence: list[float], max_sum: float) -> int:
    s = 0.0
    for i, p in enumerate(sequence):
        s += p
        if s >= 1 - max_sum:  ## only this line has changed
            return i
    return i


def create_predictions(
    model_dir,
    snapshot_datetime,
    snapshots_df,
    specialties,
    prediction_window,
    cdf_cut_points,
) -> dict:  # [SpecialtyType, DemandPredictions]:
    # initialisation
    hour = snapshot_datetime.hour
    minute = snapshot_datetime.minute
    prediction_time = (hour, minute)

    # initiase predictions dict
    predictions: dict = {key: {} for key in specialties}

    # Prepare data and model for patients in ED
    X_test, y_test, admissions_model = prepare_for_inference(
        model_dir, "ed_admission", prediction_time=prediction_time, df=snapshots_df
    )
    ### NOTE - probably need to drop consult sequence ######

    # Run inference on the patients in ED (admission probabilities)
    prob_admission_after_ed = model_input_to_pred_proba(X_test, admissions_model)

    # Run inference on the patients in ED (admission probabilities)
    snapshots_df = get_specialty_probs(model_dir, snapshots_df)

    # Prepare data and model for yet-to-arrive
    yet_to_arrive_model_name = (
        "ed_yet_to_arrive_by_spec_" + prediction_window + "_hours"
    )
    yet_to_arrive_model = load_saved_model(model_dir, yet_to_arrive_model_name)

    # Run inference on the yet-to-arrive
    prediction_context = {
        key: {"prediction_time": prediction_time} for key in specialties
    }
    pred_demand_yta = yet_to_arrive_model.predict(prediction_context)

    for spec_ in specialties:
        # Process patients in ED
        prob_admission_to_specialty = (
            snapshots_df["specialty_prob"].apply(lambda x: x[spec_]).values
        )

        # Filter the weights and the patients to exclude any with zero probability,
        # before calling the function to generate a distribution over bed numbers
        # Where prob_admission_to_specialty are equal to zero, this is either because
        # a child has zero probability of being admitted to an adult ward, or vice versa
        # These non-standard cases should not be included in the overall bed counts in such cases

        non_zero_indices = prob_admission_to_specialty != 0
        filtered_prob_admission_after_ed = prob_admission_after_ed[non_zero_indices]
        filtered_prob_admission_to_specialty = prob_admission_to_specialty[
            non_zero_indices
        ]

        # Call the function to predict demand for patients in ED, with the filtered data
        pred_demand_in_ED = pred_proba_to_pred_demand(
            filtered_prob_admission_after_ed, filtered_prob_admission_to_specialty
        )

        # Process patients yet-to-arrive
        prediction_context = {spec_: {"prediction_time": prediction_time}}

        # Return the distributions at the desired cut points
        predictions[spec_]["in_ed"] = [
            index_of_sum(pred_demand_in_ED["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
        predictions[spec_]["yet_to_arrive"] = [
            index_of_sum(pred_demand_yta[spec_]["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]

    return predictions
