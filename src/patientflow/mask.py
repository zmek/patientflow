import hashlib
import random
from datetime import timedelta
import pandas as pd
from typing import Union, List, Optional


def calculate_shift_delta(
    seed: int, min_weeks: int = 520, max_weeks: int = 520 * 2
) -> timedelta:
    """
    Calculate a consistent time delta based on the provided seed.

    Args:
        seed (int): Random seed for consistent shifts
        min_weeks (int): Minimum number of weeks to shift
        max_weeks (int): Maximum number of weeks to shift

    Returns:
        timedelta: The time shift to apply
    """
    random.seed(seed)
    weeks_to_add = random.randint(min_weeks, max_weeks)
    return timedelta(weeks=weeks_to_add)


def shift_dates(
    data: Union[pd.DataFrame, List[pd.Timestamp], pd.DatetimeIndex],
    seed: int,
    min_weeks: int = 520,
    max_weeks: int = 520 * 2,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Union[pd.DataFrame, List[pd.Timestamp]]:
    """
    Shift dates either in a DataFrame or create shifted dates from a date range.

    Args:
        data: Either a DataFrame with datetime columns to shift, or None if using start/end dates
        seed: Random seed for consistent shifts
        min_weeks: Minimum number of weeks to shift
        max_weeks: Maximum number of weeks to shift
        start_date: Start date if generating a date range (only used if data is None)
        end_date: End date if generating a date range (only used if data is None)

    Returns:
        Union[pd.DataFrame, List[pd.Timestamp]]: Either the shifted DataFrame or list of shifted dates
    """
    shift_delta = calculate_shift_delta(seed, min_weeks, max_weeks)

    # Case 1: Shifting DataFrame columns
    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        datetime_cols = df_copy.select_dtypes(
            include=["datetime64[ns]", "datetime64"]
        ).columns
        for col in datetime_cols:
            df_copy[col] = df_copy[col].apply(
                lambda x: x + shift_delta if pd.notna(x) else x
            )
        return df_copy

    # Case 2: Generating shifted date range
    elif start_date is not None and end_date is not None:
        original_dates = pd.date_range(
            start=start_date, end=end_date, freq="D"
        ).date.tolist()[:-1]
        return [date + shift_delta for date in original_dates]

    # Case 3: Shifting list of dates or DatetimeIndex
    elif isinstance(data, (list, pd.DatetimeIndex)):
        return [date + shift_delta for date in data]

    else:
        raise ValueError(
            "Must provide either a DataFrame, date range parameters, or list of dates"
        )


def hash_csn(df, salt):
    """
    Consistently hash CSN values in a dataframe
    Returns a new dataframe with hashed CSN column
    """
    # Create a copy to avoid modifying original
    df_hashed = df.copy()

    def hash_value(value):
        if pd.isna(value):
            return None
        salted = f"{str(value)}{salt}".encode()
        return hashlib.sha256(salted).hexdigest()[:12]

    # Apply the hash function to the CSN column
    df_hashed["csn"] = df_hashed["csn"].apply(hash_value)

    return df_hashed
