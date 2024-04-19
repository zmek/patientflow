from pathlib import Path
from datetime import datetime
import pandas as pd

from predict.emergency_demand.from_individual_probs import model_input_to_pred_proba, pred_proba_to_pred_demand
from ed_admissions_helper_functions import prepare_for_inference
from ed_admissions_utils import load_saved_model 

def index_of_sum(sequence: list[float], max_sum: float) -> int:
    s = 0.0
    for i, p in enumerate(sequence):
        s += p
        if s >= 1 - max_sum:  ## only this line has changed
            return i
    return i

def create_predictions(model_dir, slice_datetime, episode_slices_df, specialties, prediction_window, cdf_cut_points) -> dict:#[SpecialtyType, DemandPredictions]:

    # initialisation
    hour = slice_datetime.hour
    minute = slice_datetime.minute
    time_of_day = (hour, minute)


    predictions = {key: {subkey: None for subkey in ['in_ed', 'yet_to_arrive']} for key in specialties}

    # set prediction context for the yet to arrive model

    # Prepare data and model for patients in ED
    X_test, y_test, admissions_model = prepare_for_inference(model_dir, 'ed_admission', time_of_day = time_of_day, df = episode_slices_df) 

    # Run inference on the patients in ED (admission probabilities)
    pred_proba = model_input_to_pred_proba(X_test, admissions_model)

    # temporary code until specialty model is ready--------
    episode_slices_df['is_child'] = episode_slices_df['age_group'] == '0-17'

    # Temp dictionaries until spec model ready
    adult_dict = {
        'medical': 0.71,
        'surgical': 0.20,
        'haem_onc': 0.09,
        'paediatric': 0.00
    }
    child_dict = {
        'medical': 0.0,
        'surgical': 0.0,
        'haem_onc': 0.0,
        'paediatric': 1.0
    }

    # Assign dictionary based on the condition in 'is_child' column
    episode_slices_df['specialty_prob'] = episode_slices_df['is_child'].apply(lambda x: child_dict if x else adult_dict)
    # end of temporary code --------


    # Prepare data and model for yet-to-arrive
    yet_to_arrive_model_name = 'ed_yet_to_arrive_by_spec_' + prediction_window + '_hours'
    yet_to_arrive_model = load_saved_model(model_dir, yet_to_arrive_model_name)

    # Run inference on the yet-to-arrive 
    prediction_context = {key: {'time_of_day': time_of_day} for key in specialties}
    pred_demand_yta = yet_to_arrive_model.predict(prediction_context)


    for spec_ in specialties:

        # Process patients in ED
        spec_weights = episode_slices_df['specialty_prob'].apply(lambda x: x[spec_]).values


        # Where spec_weights are equal to zero, this is either because - following the formulation chosen here - a child has zero probability of being admitted to an adult ward, or vice versa
        # These non-standard cases should not be included in the overall bed counts in such cases
        # Therefore filter the weights and the patients to exclude any with zero probability,
        # before calling the function to generate a distribution over bed numbers

        non_zero_indices = spec_weights != 0
        filtered_pred_proba = pred_proba[non_zero_indices]
        filtered_spec_weights = spec_weights[non_zero_indices]

        # Call the function with the filtered data
        pred_demand_in_ED = pred_proba_to_pred_demand(filtered_pred_proba, filtered_spec_weights)

        # Process patients yet-to-arrive
        prediction_context = {
        spec_: {
            'time_of_day': time_of_day  
            }
        }

        # Return the distributions at the desired cut points
        predictions[spec_]['in_ed'] = [index_of_sum(pred_demand_in_ED['agg_proba'].values.cumsum(), cut_point) for cut_point in cdf_cut_points]
        predictions[spec_]['yet_to_arrive'] = [index_of_sum(pred_demand_yta[spec_]['agg_proba'].values.cumsum(), cut_point) for cut_point in cdf_cut_points]
        
    return(predictions)
    
        
    

    
    