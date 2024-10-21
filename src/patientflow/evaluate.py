"""
Patient Flow Evaluation Module

This module provides functions for evaluating and comparing different prediction models
for patient admissions in a healthcare setting. It includes utilities for calculating
metrics such as Mean Absolute Error (MAE) and Mean Percentage Error (MPE), as well as
functions for predicting admissions based on historical data and combining different
prediction models.

Key Features:
- Evaluation of probability distribution-based prediction models
- Calculation of observed admissions based on ED targets
- Prediction using historical data from previous weeks
- Combination and evaluation of multiple prediction models

Main Functions:
- calc_mae_mpe: Calculate MAE and MPE for probability distribution predictions
- calc_observed_with_ED_targets: Calculate actual admissions assuming ED targets are met
- predict_using_previous_weeks: Predict admissions using average from previous weeks
- evaluate_six_week_average: Evaluate the six-week average prediction model
- evaluate_combined_model: Evaluate a combined prediction model

"""


from typing import Dict, List, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import signal
from patientflow.predict.admission_in_prediction_window import get_y_from_aspirational_curve
from patientflow.load import get_model_name

def calculate_results(expected_values: List[Union[int, float]], observed_values: List[float]) -> Dict[str, Union[List[Union[int, float]], float]]:
    """
    Calculate evaluation metrics based on expected and observed values.

    Args:
        expected_values (List[Union[int, float]]): List of expected values.
        observed_values (List[float]): List of observed values.

    Returns:
        Dict[str, Union[List[Union[int, float]], float]]: Dictionary containing expected values, observed values, MAE, and MPE.
    """
    expected_array: np.ndarray = np.array(expected_values)
    observed_array: np.ndarray = np.array(observed_values)
    
    absolute_errors: np.ndarray = np.abs(expected_array - observed_array)
    mae: float = float(np.mean(absolute_errors))
    
    non_zero_mask: np.ndarray = observed_array != 0
    filtered_absolute_errors: np.ndarray = absolute_errors[non_zero_mask]
    filtered_observed_array: np.ndarray = observed_array[non_zero_mask]
    
    percentage_errors: np.ndarray = filtered_absolute_errors / filtered_observed_array * 100
    mpe: float = float(np.mean(percentage_errors))
    
    return {
        'expected': expected_values,
        'observed': observed_values,
        'mae': mae,
        'mpe': mpe
    }

def calc_mae_mpe(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], use_most_probable: bool = True) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Calculate MAE and MPE for all prediction times in the given probability distribution dictionary.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        use_most_probable (bool, optional): Whether to use the most probable value or expected value. Defaults to True.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Dictionary of results for each prediction time.
    """
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
        
        results[_prediction_time] = calculate_results(expected_values, observed_values)

    return results

def calc_observed_with_ED_targets(df: pd.DataFrame, dt: datetime, prediction_window: int, x1: float, y1: float, x2: float, y2: float, prediction_time: Tuple[int, int] = (15, 30)) -> float:
    """
    Calculate actual number admitted within the prediction window, assuming ED targets are met.

    Args:
        df (pd.DataFrame): DataFrame containing patient data.
        dt (datetime): Date for prediction.
        prediction_window (int): Prediction window in minutes.
        x1 (float), y1 (float), x2 (float), y2 (float): Parameters for aspirational curve.
        prediction_time (Tuple[int, int], optional): Hour and minute of prediction. Defaults to (15, 30).

    Returns:
        float: Weighted observed admissions.
    """
    prediction_datetime = pd.to_datetime(dt).replace(hour=prediction_time[0], minute=prediction_time[1])
    
    filtered_df = df[df['arrival_datetime'].dt.date == dt]
    arrived_before = filtered_df[filtered_df.arrival_datetime < prediction_datetime].copy()
    arrived_after = filtered_df[filtered_df.arrival_datetime >= prediction_datetime].copy()

    arrived_before['hours_before_pred_window'] = arrived_before['arrival_datetime'].apply(lambda x: ((prediction_datetime - x)).seconds/3600)
    arrived_before['prob_admission_in_pred_window'] = \
        arrived_before['hours_before_pred_window'].apply(lambda x: get_y_from_aspirational_curve(x + prediction_window/60, x1, y1, x2, y2)) - \
        arrived_before['hours_before_pred_window'].apply(lambda x: get_y_from_aspirational_curve(x, x1, y1, x2, y2))

    arrived_after['hours_after_pred_window'] = arrived_after['arrival_datetime'].apply(lambda x: ((x - prediction_datetime)).seconds/3600)
    arrived_after['prob_admission_in_pred_window'] = \
        arrived_after['hours_after_pred_window'].apply(lambda x: get_y_from_aspirational_curve((prediction_window/60) - x, x1, y1, x2, y2))

    weighted_observed = arrived_before['prob_admission_in_pred_window'].sum() + arrived_after['prob_admission_in_pred_window'].sum()
    return weighted_observed

def predict_using_previous_weeks(df: pd.DataFrame, dt: datetime, prediction_time: Tuple[int, int], num_weeks: int) -> float:
    """
    Calculate predicted number using average from previous weeks.

    Args:
        df (pd.DataFrame): DataFrame containing patient data.
        dt (datetime): Date for prediction.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.

    Returns:
        float: Predicted number of admissions.
    """
    prediction_datetime = pd.to_datetime(dt).replace(hour=prediction_time[0], minute=prediction_time[1])
    target_day_of_week = dt.weekday()
    
    end_date = dt - timedelta(days=1)
    start_date = end_date - timedelta(weeks=num_weeks)
    
    mask = (
        (df['arrival_datetime'].dt.date >= start_date) &
        (df['arrival_datetime'].dt.date < end_date) &
        (df['arrival_datetime'].dt.weekday == target_day_of_week)
    )
    filtered_df = df[mask]
    
    average_count = len(filtered_df) / num_weeks
    target_date_count = len(df[(df['arrival_datetime'].dt.date == dt) & 
                                 (df['arrival_datetime'] < prediction_datetime)])
    
    still_to_admit = average_count - target_date_count
    return still_to_admit

def evaluate_six_week_average(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], 
                              df: pd.DataFrame,
                              prediction_window: int,
                              x1: float, y1: float, x2: float, y2: float,
                              prediction_time: Tuple[int, int], 
                              num_weeks: int,
                              model_name: str) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Evaluate the six-week average prediction model.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        df (pd.DataFrame): DataFrame containing patient data.
        prediction_window (int): Prediction window in minutes.
        x1 (float), y1 (float), x2 (float), y2 (float): Parameters for aspirational curve.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.
        model_name (str): Name of the model.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Evaluation results.
    """
    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_name(model_name, prediction_time)
    
    for dt in prob_dist_dict_all[model_name].keys():
        expected_value: float = float(predict_using_previous_weeks(df, dt, prediction_time, num_weeks))
        observed_value: float = float(calc_observed_with_ED_targets(df, dt, prediction_window, x1, y1, x2, y2, prediction_time))
        
        expected_values.append(expected_value)
        observed_values.append(observed_value)
    
    results = {model_name: calculate_results(expected_values, observed_values)}
    return results

def combine_distributions(dist1: pd.DataFrame, dist2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two probability distributions using convolution.

    Args:
        dist1 (pd.DataFrame): First probability distribution.
        dist2 (pd.DataFrame): Second probability distribution.

    Returns:
        pd.DataFrame: Combined probability distribution.
    """
    arr1 = dist1.values
    arr2 = dist2.values
    
    combined = signal.convolve(arr1, arr2)
    new_index = range(len(combined))
    
    combined_df = pd.DataFrame(combined, index=new_index, columns=['agg_predicted'])
    combined_df['agg_predicted'] = combined_df['agg_predicted'] / combined_df['agg_predicted'].sum()
    
    return combined_df

def evaluate_combined_model(prob_dist_dict_all: Dict[Any, Dict[Any, Dict[str, Any]]], 
                            df: pd.DataFrame,
                            yta_preds: pd.DataFrame,
                            prediction_window: int,
                            x1: float, y1: float, x2: float, y2: float,
                            prediction_time: Tuple[int, int], 
                            num_weeks: int,
                            model_name: str,
                            use_most_probable: bool = True) -> Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]:
    """
    Evaluate the combined prediction model.

    Args:
        prob_dist_dict_all (Dict[Any, Dict[Any, Dict[str, Any]]]): Nested dictionary containing probability distributions.
        df (pd.DataFrame): DataFrame containing patient data.
        yta_preds (pd.DataFrame): Yet-to-arrive predictions.
        prediction_window (int): Prediction window in minutes.
        x1 (float), y1 (float), x2 (float), y2 (float): Parameters for aspirational curve.
        prediction_time (Tuple[int, int]): Hour and minute of prediction.
        num_weeks (int): Number of previous weeks to consider.
        model_name (str): Name of the model.
        use_most_probable (bool, optional): Whether to use the most probable value or expected value. Defaults to True.

    Returns:
        Dict[Any, Dict[str, Union[List[Union[int, float]], float]]]: Evaluation results.
    """
    expected_values: List[Union[int, float]] = []
    observed_values: List[float] = []

    model_name = get_model_name(model_name, prediction_time)
    
    for dt in prob_dist_dict_all[model_name].keys():
        in_ed_preds: Dict[str, Any] = prob_dist_dict_all[model_name][dt]
        combined = combine_distributions(yta_preds, in_ed_preds['agg_predicted'])
        
        if use_most_probable:
            expected_value: int = int(combined['agg_predicted'].idxmax())
        else:
            expected_value: float = float(np.dot(combined['agg_predicted'].index, combined['agg_predicted'].values.flatten()))
        
        observed_value: float = float(calc_observed_with_ED_targets(df, dt, prediction_window, x1, y1, x2, y2, prediction_time))
        
        expected_values.append(expected_value)
        observed_values.append(observed_value)
    
    results = {model_name: calculate_results(expected_values, observed_values)}
    return results