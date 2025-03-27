import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time


def patient_visits(start_date, end_date, mean_patients_per_day):
    """
    Generate fake patient visit data with random arrival and departure times.

    Parameters:
    -----------
    start_date : str or datetime
        The minimum date to sample from (format: 'YYYY-MM-DD' if string)
    end_date : str or datetime
        The maximum date to sample from (exclusive) (format: 'YYYY-MM-DD' if string)
    mean_patients_per_day : float
        The average number of patients to generate per day

    Returns:
    --------
    tuple (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        First DataFrame: visits with columns: visit_number, arrival_datetime, departure_datetime,
        is_admitted, age
        Second DataFrame: observations with columns: visit_number, observation_datetime, triage_score
        Third DataFrame: lab_orders with columns: visit_number, order_datetime, lab_name
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Set random seed for reproducibility
    np.random.seed(42)  # You can change this seed value as needed

    # Calculate total days in range (changed to exclusive end date)
    days_range = (end_date - start_date).days

    # Generate random number of patients for each day using Poisson distribution
    daily_patients = np.random.poisson(mean_patients_per_day, days_range)

    # Define admission probabilities based on triage score
    # Triage 1: 80% admission, Triage 2: 60%, Triage 3: 30%, Triage 4: 10%, Triage 5: 2%
    admission_probabilities = {
        1: 0.80,  # Highest severity - highest admission probability
        2: 0.60,
        3: 0.30,
        4: 0.10,
        5: 0.02,  # Lowest severity - lowest admission probability
    }

    # Define triage score distribution
    # Most common is 3-4, less common are 2 and 5, least common is 1 (most severe)
    triage_probabilities = [0.05, 0.15, 0.35, 0.35, 0.10]  # For scores 1-5

    # Define common ED lab tests and their ordering probabilities based on triage score
    lab_tests = ["CBC", "BMP", "Troponin", "D-dimer", "Urinalysis"]
    lab_probabilities = {
        # Higher severity -> more likely to get labs
        1: {
            "CBC": 0.95,
            "BMP": 0.95,
            "Troponin": 0.90,
            "D-dimer": 0.70,
            "Urinalysis": 0.60,
        },
        2: {
            "CBC": 0.90,
            "BMP": 0.90,
            "Troponin": 0.80,
            "D-dimer": 0.60,
            "Urinalysis": 0.50,
        },
        3: {
            "CBC": 0.80,
            "BMP": 0.80,
            "Troponin": 0.60,
            "D-dimer": 0.40,
            "Urinalysis": 0.40,
        },
        4: {
            "CBC": 0.60,
            "BMP": 0.60,
            "Troponin": 0.30,
            "D-dimer": 0.20,
            "Urinalysis": 0.30,
        },
        5: {
            "CBC": 0.40,
            "BMP": 0.40,
            "Troponin": 0.15,
            "D-dimer": 0.10,
            "Urinalysis": 0.20,
        },
    }

    visits = []
    observations = []
    lab_orders = []
    visit_number = 1

    for day_idx, num_patients in enumerate(daily_patients):
        current_date = start_date + timedelta(days=day_idx)

        # Generate patients for this day
        for _ in range(num_patients):
            # Random hour for arrival (more likely during daytime)
            arrival_hour = np.random.normal(13, 4)  # Mean at 1 PM, std dev of 4 hours
            arrival_hour = max(0, min(23, int(arrival_hour)))  # Clamp between 0-23

            # Random minutes
            arrival_minute = np.random.randint(0, 60)

            # Create arrival datetime
            arrival_datetime = current_date.replace(
                hour=arrival_hour,
                minute=arrival_minute,
                second=np.random.randint(0, 60),
            )

            # Generate triage score (1-5)
            triage_score = np.random.choice([1, 2, 3, 4, 5], p=triage_probabilities)

            # Generate length of stay (in minutes) - log-normal distribution
            # Most visits are 2 to 6 hours, but some can be shorter or longer
            length_of_stay = np.random.lognormal(mean=5.2, sigma=0.4)
            length_of_stay = max(
                30, min(1440, length_of_stay)
            )  # Between 30 min and 24 hours

            # Make higher triage scores (more severe) stay longer on average
            if triage_score <= 2:
                length_of_stay *= 1.5  # 50% longer stays for more severe cases

            # Calculate departure time
            departure_datetime = arrival_datetime + timedelta(
                minutes=int(length_of_stay)
            )

            # Generate admission status based on triage score
            admission_prob = admission_probabilities[triage_score]
            is_admitted = np.random.choice(
                [0, 1], p=[1 - admission_prob, admission_prob]
            )

            # Generate age with a distribution skewed towards older adults
            age = int(
                np.random.lognormal(mean=3.8, sigma=0.5)
            )  # Centers around 45 years
            age = max(0, min(100, age))  # Clamp between 0-100 years

            # Add visit record (without triage score)
            visits.append(
                {
                    "visit_number": visit_number,
                    "arrival_datetime": arrival_datetime,
                    "departure_datetime": departure_datetime,
                    "is_admitted": is_admitted,
                    "age": age,
                }
            )

            # Generate triage observation within first 10 minutes
            minutes_after_arrival = np.random.uniform(0, 10)
            observation_datetime = arrival_datetime + timedelta(
                minutes=minutes_after_arrival
            )

            observations.append(
                {
                    "visit_number": visit_number,
                    "observation_datetime": observation_datetime,
                    "triage_score": triage_score,
                }
            )

            # Generate lab orders if visit is longer than 2 hours
            if length_of_stay > 120:
                # For each lab test, decide if it should be ordered based on triage score
                for lab_test in lab_tests:
                    if np.random.random() < lab_probabilities[triage_score][lab_test]:
                        # Order time is after triage but within first 90 minutes
                        minutes_after_triage = np.random.uniform(
                            0, 90 - minutes_after_arrival
                        )
                        order_datetime = observation_datetime + timedelta(
                            minutes=minutes_after_triage
                        )

                        lab_orders.append(
                            {
                                "visit_number": visit_number,
                                "order_datetime": order_datetime,
                                "lab_name": lab_test,
                            }
                        )

            visit_number += 1

    # Create DataFrames and sort by time
    visits_df = pd.DataFrame(visits)
    visits_df = visits_df.sort_values("arrival_datetime").reset_index(drop=True)

    observations_df = pd.DataFrame(observations)
    observations_df = observations_df.sort_values("observation_datetime").reset_index(
        drop=True
    )

    lab_orders_df = pd.DataFrame(lab_orders)
    if not lab_orders_df.empty:
        lab_orders_df = lab_orders_df.sort_values("order_datetime").reset_index(
            drop=True
        )

    return visits_df, observations_df, lab_orders_df


def create_snapshots(
    df, observations_df, lab_orders_df, prediction_times, start_date, end_date
):
    """
    Create snapshots of patients present at specific times between start_date and end_date.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with patient visit data, must have 'arrival_datetime' and 'departure_datetime' columns
    observations_df : pandas.DataFrame
        DataFrame with triage observations, must have 'visit_number', 'observation_datetime', 'triage_score' columns
    lab_orders_df : pandas.DataFrame
        DataFrame with lab orders, must have 'visit_number', 'order_datetime', 'lab_name' columns
    prediction_times : list of tuples
        List of (hour, minute) tuples representing times to take snapshots
    start_date : str or datetime
        First date to take snapshots (format: 'YYYY-MM-DD' if string)
    end_date : str or datetime
        Last date to take snapshots (exclusive) (format: 'YYYY-MM-DD' if string)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with snapshot information and patient data, including lab order counts
    """
    # Add date conversion at the start
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    # Create date range (changed to exclusive end date)
    snapshot_dates = []
    current_date = start_date
    while current_date < end_date:  # Changed from <= to <
        snapshot_dates.append(current_date)
        current_date += timedelta(days=1)

    # Get unique lab test names
    lab_tests = lab_orders_df["lab_name"].unique() if not lab_orders_df.empty else []

    # Create empty list to store all results
    all_results = []

    # For each combination of date and time
    for date in snapshot_dates:
        for hour, minute in prediction_times:
            snapshot_datetime = datetime.combine(date, time(hour=hour, minute=minute))

            # Filter dataframe for this snapshot
            mask = (df["arrival_datetime"] <= snapshot_datetime) & (
                df["departure_datetime"] > snapshot_datetime
            )
            snapshot_df = df[mask].copy()  # Create copy to avoid SettingWithCopyWarning

            # Skip if no patients at this time
            if len(snapshot_df) == 0:
                continue

            # Get triage scores recorded before the snapshot time
            valid_observations = observations_df[
                (observations_df["visit_number"].isin(snapshot_df["visit_number"]))
                & (observations_df["observation_datetime"] <= snapshot_datetime)
            ].copy()

            # Keep only the most recent triage score for each visit
            if not valid_observations.empty:
                valid_observations = valid_observations.sort_values(
                    "observation_datetime"
                )
                valid_observations = (
                    valid_observations.groupby("visit_number").last().reset_index()
                )
                valid_observations = valid_observations.rename(
                    columns={"triage_score": "latest_triage_score"}
                )

            # Get lab orders placed before the snapshot time
            valid_orders = lab_orders_df[
                (lab_orders_df["visit_number"].isin(snapshot_df["visit_number"]))
                & (lab_orders_df["order_datetime"] <= snapshot_datetime)
            ].copy()

            # Initialize lab_counts with zeros for all visits in snapshot_df
            lab_counts = pd.DataFrame(
                0,
                index=pd.Index(
                    snapshot_df["visit_number"].unique(), name="visit_number"
                ),
                columns=[f"num_{test.lower()}_orders" for test in lab_tests],
            )

            # Update counts if there are any valid orders
            if not valid_orders.empty:
                order_counts = (
                    valid_orders.groupby(["visit_number", "lab_name"])
                    .size()
                    .unstack(fill_value=0)
                )
                order_counts.columns = [
                    f"num_{test.lower()}_orders" for test in order_counts.columns
                ]
                # Update the counts in lab_counts where we have orders
                lab_counts.update(order_counts)

            lab_counts = lab_counts.reset_index()

            # Add snapshot information columns
            snapshot_df["snapshot_date"] = date
            snapshot_df["prediction_time"] = [(hour, minute)] * len(snapshot_df)

            # Merge with valid observations to get triage scores, handling empty case
            if not valid_observations.empty:
                snapshot_df = pd.merge(
                    snapshot_df,
                    valid_observations[["visit_number", "latest_triage_score"]],
                    on="visit_number",
                    how="left",
                )
            else:
                snapshot_df["latest_triage_score"] = pd.Series(
                    [np.nan], dtype="float64", index=snapshot_df.index
                )

            # Merge with lab counts
            snapshot_df = pd.merge(
                snapshot_df, lab_counts, on="visit_number", how="left"
            )

            # Fill NA values in lab count columns with 0
            for col in snapshot_df.columns:
                if col.endswith("_orders"):
                    snapshot_df[col] = snapshot_df[col].fillna(0)
            if not snapshot_df.empty:
                # Optionally check for all-NA in key columns
                snapshot_cols = [
                    "snapshot_date",
                    "prediction_time",
                    "snapshot_datetime",
                ]
                # Only check columns that exist in the DataFrame
                check_cols = [
                    col for col in snapshot_cols if col in snapshot_df.columns
                ]

                if not check_cols or not snapshot_df[check_cols].isna().all().any():
                    all_results.append(snapshot_df)
                else:
                    print(
                        f"Skipping DataFrame with all-NA values in key columns: {check_cols}"
                    )
            else:
                print("Skipping empty DataFrame")

    # Combine all results into single dataframe
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Define column order
        snapshot_cols = ["snapshot_date", "prediction_time"]
        visit_cols = [
            "visit_number",
            "is_admitted",
            "age",
            "latest_triage_score",
        ]
        lab_cols = [col for col in final_df.columns if col.endswith("_orders")]

        # Ensure all required columns exist
        for col in visit_cols:
            if col not in final_df.columns:
                if col == "latest_triage_score":
                    final_df[col] = pd.NA
                else:
                    final_df[col] = None

        # Reorder columns
        final_df = final_df[snapshot_cols + visit_cols + lab_cols]
    else:
        # Create empty dataframe with correct columns if no results found
        lab_cols = [f"num_{test.lower()}_orders" for test in lab_tests]
        columns = [
            "snapshot_date",
            "prediction_time",
            "visit_number",
            "is_admitted",
            "age",
            "latest_triage_score",
        ] + lab_cols
        final_df = pd.DataFrame(columns=columns)

    # Name the index snapshot_id before returning
    final_df.index.name = "snapshot_id"
    return final_df
