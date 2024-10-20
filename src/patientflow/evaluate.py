import numpy as np
from typing import Dict, List, Any, Union
import pandas as pd
from datetime import datetime, timedelta
from patientflow.predict.admission_in_prediction_window import get_y_from_aspirational_curve
from patientflow.load import get_model_name
from scipy import signal


def calc_mae_mpe(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], use_most_probable: bool = True) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

    for _prediction_time in prob_dist_dict_all.keys():
        expected_values: List[Union[int, float]] = []
        observed_values: List[float] = []
        
        for dt in prob_dist_dict_all[_prediction_time].keys():
            preds: Dict[str, Any] = prob_dist_dict_all[_prediction_time][dt]
            
            if use_most_probable:
                expected_value: int = int(preds['agg_predicted'].idxmax().values[0])
            else:
                expected_value: float = float(np.dot(preds['agg_predicted'].index, preds['agg_predicted'].values.flatten()))
            
            observed_value: float = float(preds['agg_observed'])
            
            expected_values.append(expected_value)
            observed_values.append(observed_value)
        
        # Convert to numpy arrays for easier calculations
        expected_array: np.ndarray = np.array(expected_values)
        observed_array: np.ndarray = np.array(observed_values)
        
        # Calculate absolute errors
        absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
        
        # Calculate MAE
        mae: float = float(np.mean(absolute_errors))
        
        # Remove items where observed value is zero
        non_zero_mask: np.ndarray = observed_array != 0
        filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
        filtered_observed_array: np.ndarray = observed_array[non_zero_mask]
        
        # Calculate percentage errors (only for non-zero observed values)
        percentage_errors: np.ndarray = filtered_absolute_errors / filtered_observed_array * 100
        
        # Calculate MPE
        mpe: float = float(np.mean(percentage_errors))
        
        results[_prediction_time] = {
            'expected': expected_values,
            'observed': observed_values,
            'mae': mae,
            'mpe': mpe
        }

    return results



# Calculate actual number admitted within the prediction window, assuming ED targets are met
def calc_observed_with_ED_targets(df, dt, prediction_window, x1, y1, x2, y2, prediction_time=(15,30)):

    prediction_datetime = pd.to_datetime(dt).replace(hour=prediction_time[0], minute=prediction_time[1]) 
 
    # Filter the DataFrame for the relevant date 
    filtered_df = df[df['arrival_datetime'].dt.date == dt]
    arrived_before = filtered_df[filtered_df.arrival_datetime < prediction_datetime].copy()
    arrived_after = filtered_df[filtered_df.arrival_datetime >= prediction_datetime].copy()

    # rows where patient arrived prior to prediction_datetime
    arrived_before['hours_before_pred_window'] = arrived_before['arrival_datetime'].apply(lambda x: ((prediction_datetime - x)).seconds/3600)

    # calculate probability of admission in prediction window
    arrived_before['prob_admission_in_pred_window']= \
        arrived_before['hours_before_pred_window'].apply(lambda x: get_y_from_aspirational_curve(x + prediction_window/60, x1, y1, x2, y2)) - \
            arrived_before['hours_before_pred_window'].apply(lambda x: get_y_from_aspirational_curve(x, x1, y1, x2, y2))

    # rows where patient arrived after prediction_datetime
    arrived_after['hours_after_pred_window'] = arrived_after['arrival_datetime'].apply(lambda x: ((x - prediction_datetime)).seconds/3600)

    arrived_after['prob_admission_in_pred_window']= \
        arrived_after['hours_after_pred_window'].apply(lambda x: get_y_from_aspirational_curve((prediction_window/60) - x, x1, y1, x2, y2)) 

    weighted_observed = arrived_before['prob_admission_in_pred_window'].sum() + arrived_after['prob_admission_in_pred_window'].sum()
    return weighted_observed

# Calculate predicted number using six week average
def predict_using_previous_weeks(df, dt, prediction_time, num_weeks):

    # Offset prediction_time by 3 hours (allowing 3 hours for admitted patients to be processed in ED)
    prediction_datetime = pd.to_datetime(dt).replace(hour=prediction_time[0], minute=prediction_time[1]) 

    
    # Get the day of the week for the target date
    target_day_of_week = dt.weekday()
    
    # Calculate the date range for the previous six weeks
    end_date = dt - timedelta(days=1)  # Exclude the target date
    start_date = end_date - timedelta(weeks=num_weeks)
    
    # Filter the DataFrame for the relevant date range and day of the week
    mask = (
        (df['arrival_datetime'].dt.date >= start_date) &
        (df['arrival_datetime'].dt.date < end_date) &
        (df['arrival_datetime'].dt.weekday == target_day_of_week)
    )
    filtered_df = df[mask]
    
    # Calculate the average count for the number of admissions in previous weeks
    average_count = len(filtered_df) / num_weeks

    # Calculate the number of admissions for the specific date before prediction time
    target_date_count = len(df[(df['arrival_datetime'].dt.date == dt) & 
                                 (df['arrival_datetime'] < prediction_datetime)])
    
    # Calculate how many admissions are left on the target date, based on this average
    still_to_admit = average_count - target_date_count
    return still_to_admit


def evaluate_six_week_average(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], 
                             df,
                             prediction_window,
                             x1, y1, x2, y2,
                            prediction_time, 
                            num_weeks,
                            model_name
                             ) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_name(model_name, prediction_time)
    
    for dt in prob_dist_dict_all[model_name].keys():
        
        expected_value: float = float(predict_using_previous_weeks(df, dt, prediction_time, num_weeks))
        observed_value: float = float(calc_observed_with_ED_targets(df, dt, prediction_window, x1, y1, x2, y2, prediction_time))
        
        expected_values.append(expected_value)
        observed_values.append(observed_value)
    
    # Convert to numpy arrays for easier calculations
    expected_array: np.ndarray = np.array(expected_values)
    observed_array: np.ndarray = np.array(observed_values)
    
    # Calculate absolute errors
    absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
    
    # Calculate MAE
    mae: float = float(np.mean(absolute_errors))
    
    # Remove items where observed value is zero
    non_zero_mask: np.ndarray = observed_array != 0
    filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
    filtered_observed_array: np.ndarray = observed_array[non_zero_mask]
    
    # Calculate percentage errors (only for non-zero observed values)
    percentage_errors: np.ndarray = filtered_absolute_errors / filtered_observed_array * 100
    
    # Calculate MPE
    mpe: float = float(np.mean(percentage_errors))
    
    results[model_name] = {
        'expected': expected_values,
        'observed': observed_values,
        'mae': mae,
        'mpe': mpe
    }

    return results


def combine_distributions(dist1, dist2):
    # Convert distributions to arrays
    arr1 = dist1.values
    arr2 = dist2.values
    
    # Perform convolution
    combined = signal.convolve(arr1, arr2)
    
    # Create new index for the combined distribution
    new_index = range(len(combined))
    
    # Create a new DataFrame
    combined_df = pd.DataFrame(combined, index=new_index, columns=['agg_predicted'])
    
    # Normalize the probabilities
    combined_df['agg_predicted'] = combined_df['agg_predicted'] / combined_df['agg_predicted'].sum()
    
    return combined_df


def evaluate_combined_model(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], 
                             df,
                            yta_preds,
                             prediction_window,
                             x1, y1, x2, y2,
                            prediction_time, 
                            num_weeks,
                            model_name,
                            use_most_probable: bool = True
                             ) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    results: Dict[Any, Dict[str, Union[List[Union[int, float]], float]]] = {}

    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_name(model_name, prediction_time)
    
    for dt in prob_dist_dict_all[model_name].keys():
        print(dt)

        in_ed_preds: Dict[str, Any] = prob_dist_dict_all[model_name][dt]
        combined = combine_distributions(yta_preds, in_ed_preds['agg_predicted'])
        
        if use_most_probable:
            expected_value: int = int(combined['agg_predicted'].idxmax())
        else:
            expected_value: float = float(np.dot(combined['agg_predicted'].index, combined['agg_predicted'].values.flatten()))
        
        observed_value: float = float(calc_observed_with_ED_targets(df, dt, prediction_window, x1, y1, x2, y2, prediction_time))
        
        expected_values.append(expected_value)
        observed_values.append(observed_value)
    
    # Convert to numpy arrays for easier calculations
    expected_array: np.ndarray = np.array(expected_values)
    observed_array: np.ndarray = np.array(observed_values)
    
    # Calculate absolute errors
    absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
    
    # Calculate MAE
    mae: float = float(np.mean(absolute_errors))
    
    # Remove items where observed value is zero
    non_zero_mask: np.ndarray = observed_array != 0
    filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
    filtered_observed_array: np.ndarray = observed_array[non_zero_mask]
    
    # Calculate percentage errors (only for non-zero observed values)
    percentage_errors: np.ndarray = filtered_absolute_errors / filtered_observed_array * 100
    
    # Calculate MPE
    mpe: float = float(np.mean(percentage_errors))
    
    results[model_name] = {
        'expected': expected_values,
        'observed': observed_values,
        'mae': mae,
        'mpe': mpe
    }

    return results