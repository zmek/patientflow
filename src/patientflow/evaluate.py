import numpy as np
from typing import Dict, List, Any, Union

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