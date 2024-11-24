"""
This module provides functions to calculate and process time-varying arrival rates,
admission probabilities, and true demand rates for inpatient arrivals.

The module leverages aspirational curves and historical arrival data to estimate
demand and arrival patterns, offering utilities for downstream analysis and visualization.

Functions:
    - calculate_time_varying_arrival_rates_lagged: Create lagged arrival rates based on time intervals.
    - process_arrival_rates: Prepare arrival rates for visualization.
    - calculate_admission_probabilities: Compute cumulative and hourly admission probabilities.
    - calculate_weighted_arrival_rates: Aggregate weighted arrival rates for specific hours.
    - get_true_demand_by_hour: Estimate inpatient demand by hour using historical data.
"""

import numpy as np
import datetime
from patientflow.prepare import calculate_time_varying_arrival_rates
from collections import OrderedDict
from typing import Dict, List, Tuple
from patientflow.predict.admission_in_prediction_window import (
    get_y_from_aspirational_curve,
)


def calculate_time_varying_arrival_rates_lagged(
    inpatient_arrivals, lagged_by, time_interval=60
):
    """
    Calculate lagged time-varying arrival rates.

    This function adjusts arrival times by a specified lag and computes
    the corresponding arrival rates, sorted by the lagged times.

    Args:
        inpatient_arrivals (iterable): Historical inpatient arrival data.
        lagged_by (int): Number of hours to lag the arrival times.
        time_interval (int, optional): Interval in minutes for calculating arrival rates. Defaults to 60.

    Returns:
        OrderedDict: A dictionary mapping lagged times (datetime.time) to arrival rates.
    """
    dict_ = calculate_time_varying_arrival_rates(inpatient_arrivals, time_interval)
    # Lag the arrival times by 4 hours
    lagged_dict = OrderedDict()
    for time, rate in dict_.items():
        lagged_time = (
            datetime.datetime.combine(datetime.date.today(), time)
            + datetime.timedelta(hours=lagged_by)
        ).time()
        lagged_dict[lagged_time] = rate

    # Sort the dictionary by the new lagged time
    sorted_lagged_dict = OrderedDict(sorted(lagged_dict.items()))

    return sorted_lagged_dict


def process_arrival_rates(
    arrival_rates_dict: Dict[datetime.time, float],
) -> Tuple[List[float], List[str], List[int]]:
    """
    Process arrival rates dictionary into formats needed for plotting.

    Args:
        arrival_rates_dict (Dict[datetime.time, float]): Mapping of times to arrival rates.

    Returns:
        Tuple[List[float], List[str], List[int]]:
            - hour_labels: List of formatted hour range strings (e.g., "09-\n10").
            - hour_values: List of integers for x-axis positioning.
            - arrival_rates: List of arrival rate values.
    """
    # Extract hours and rates
    hours = list(arrival_rates_dict.keys())
    arrival_rates = list(arrival_rates_dict.values())

    # Create formatted hour labels with line breaks for better plot readability
    hour_labels = [
        f'{hour.strftime("%H")}-\n{str((hour.hour + 1) % 24).zfill(2)}'
        for hour in hours
    ]

    # Generate numerical values for x-axis positioning
    hour_values = list(range(len(hour_labels)))

    return arrival_rates, hour_labels, hour_values


def calculate_admission_probabilities(hours_since_arrival, x1, y1, x2, y2):
    """
    Calculate probability of admission for each hour since arrival.

    Args:
        hours_since_arrival (np.ndarray): Array of hours since arrival.
        x1, y1, x2, y2 (float): Parameters for the aspirational curve.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - prob_admission_by_hour: Cumulative admission probabilities.
            - prob_admission_within_hour: Hourly admission probabilities.
    """
    prob_admission_by_hour = np.array(
        [
            get_y_from_aspirational_curve(hour, x1, y1, x2, y2)
            for hour in hours_since_arrival
        ]
    )
    prob_admission_within_hour = np.diff(prob_admission_by_hour)

    return prob_admission_by_hour, prob_admission_within_hour


def calculate_weighted_arrival_rates(poisson_means_all, elapsed_hours, hour_of_day):
    """Calculate sum of weighted arrival rates for a specific hour of day.

    Args:
        poisson_means_all (np.ndarray): Array of Poisson means
        elapsed_hours (range): Range of elapsed hours
        hour_of_day (int): Current hour of day

    Returns:
        float: Sum of weighted arrival rates
    """
    total = 0
    for elapsed_hour in elapsed_hours:
        hour_index = (hour_of_day - elapsed_hour) % 24
        total += poisson_means_all[elapsed_hour][hour_index]
    return total


def get_true_demand_by_hour(
    inpatient_arrivals, x1, y1, x2, y2, time_interval=60, max_hours_since_arrival=10
):
    """
    Calculate true inpatient demand by hour based on historical arrival data.

    This function estimates demand rates using historical arrival data and an aspirational
    curve for admission probabilities.

    Args:
        inpatient_arrivals (iterable): Historical inpatient arrival data.
        x1, y1, x2, y2 (float): Parameters for the aspirational curve.
        time_interval (int, optional): Time interval in minutes. Defaults to 60.
        max_hours_since_arrival (int, optional): Maximum hours since arrival to consider. Defaults to 10.

    Returns:
        Dict[datetime.time, float]: Mapping of times to demand rates.
    """
    # Calculate admission probabilities
    hours_since_arrival = np.arange(max_hours_since_arrival + 1)
    _, prob_admission_within_hour = calculate_admission_probabilities(
        hours_since_arrival, x1, y1, x2, y2
    )

    # Calculate base arrival rates from historical data
    poisson_means_dict = calculate_time_varying_arrival_rates(
        inpatient_arrivals, time_interval
    )

    # Convert dict to array while preserving order
    hour_keys = sorted(poisson_means_dict.keys())  # Sort to ensure consistent order
    poisson_means = np.array([poisson_means_dict[hour] for hour in hour_keys])

    # Initialize array for weighted arrival rates
    poisson_means_all = np.zeros((max_hours_since_arrival, len(poisson_means)))

    # Calculate weighted arrival rates for each hour and elapsed time
    for hour_idx, hour_key in enumerate(hour_keys):
        arrival_rate = poisson_means[hour_idx]
        for elapsed_hour in range(max_hours_since_arrival):
            weighted_rate = arrival_rate * prob_admission_within_hour[elapsed_hour]
            poisson_means_all[elapsed_hour][hour_idx] = weighted_rate

    # Calculate summed arrival rates for each hour and store in dictionary
    elapsed_hours = range(max_hours_since_arrival)
    demand_by_hour = {}

    for hour_idx, hour_key in enumerate(hour_keys):
        demand_by_hour[hour_key] = calculate_weighted_arrival_rates(
            poisson_means_all, elapsed_hours, hour_idx
        )

    return demand_by_hour
