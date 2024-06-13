from datetime import datetime, timedelta

import pandas as pd


def calculate_rates(df, time_interval):
    """
    Calculate the time-varying arrival rates for a dataset indexed by datetime.

    This function computes the arrival rates for each time interval specified, across the entire date range present in the dataframe. The arrival rate is calculated as the number of entries in the dataframe for each time interval, divided by the number of days in the dataset's timespan.

    Parameters
    df (pandas.DataFrame): A DataFrame indexed by datetime, representing the data for which arrival rates are to be calculated. The index of the DataFrame should be of datetime type.
    time_interval (int): The time interval, in minutes, for which the arrival rates are to be calculated. For example, if `time_interval=60`, the function will calculate hourly arrival rates.

    Returns
    dict: A dictionary where the keys are the start times of each interval (as `datetime.time` objects), and the values are the corresponding arrival rates (as floats).

    Raises
    TypeError: If the index of the DataFrame is not a datetime index.

    """
    # Validate that the DataFrame index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a DatetimeIndex.")

    # Determine the start and end date of the data
    start_dt = df.index.min()
    end_dt = df.index.max()

    # Convert start and end times to datetime if they are not already
    if not isinstance(start_dt, datetime):
        start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S%z")

    if not isinstance(end_dt, datetime):
        end_dt = datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S%z")

    # Calculate the total number of days covered by the dataset
    num_days = (end_dt - start_dt).days
    print(
        f"Calculating time-varying arrival rates for data provided, which spans {num_days} days"
    )

    arrival_rates_dict = {}

    # Initialize a time object to iterate through one day in the specified intervals
    _start_datetime = datetime(1970, 1, 1, 0, 0, 0, 0)
    _stop_datetime = _start_datetime + timedelta(days=1)

    # Iterate over each interval in a single day to calculate the arrival rate
    while _start_datetime != _stop_datetime:
        _start_time = _start_datetime.time()
        _end_time = (_start_datetime + timedelta(minutes=time_interval)).time()

        # Filter the dataframe for entries within the current time interval
        _df = df.between_time(_start_time, _end_time, inclusive="left")

        # Calculate and store the arrival rate for the interval
        arrival_rates_dict[_start_time] = _df.shape[0] / num_days

        # Move to the next interval
        _start_datetime = _start_datetime + timedelta(minutes=time_interval)

    return arrival_rates_dict
