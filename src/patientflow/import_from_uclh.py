import pandas as pd
import numpy as np


def prepare_age_and_dates(df):  # conversions necessary for each datetime column
    # calculate age on arrival
    df["age_on_arrival"] = (
        pd.to_timedelta(
            (
                pd.to_datetime(df["arrival_datetime"]).dt.date
                - pd.to_datetime(df["date_of_birth"]).dt.date
            )
        ).dt.days
        / 365.2425
    ).apply(lambda x: np.floor(x) if pd.notna(x) else x)
    # convert to groups
    bins = [-1, 18, 25, 35, 45, 55, 65, 75, 102]
    labels = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-102"]
    df["age_group"] = pd.cut(df["age_on_arrival"], bins=bins, labels=labels, right=True)

    if "snapshot_datetime" in list(df.columns):
        # df['snapshot_datetime'] = df['snapshot_datetime'].dt.tz_localize  ('UTC')
        df["prediction_time"] = (
            df["snapshot_datetime"]
            .dt.strftime("%H,%M")
            .apply(lambda x: tuple(map(int, x.split(","))))
        )
        df["snapshot_date"] = pd.to_datetime(df["snapshot_datetime"]).dt.date

        # Calculate elapsed time in ED
        df["elapsed_los"] = df["snapshot_datetime"] - df["arrival_datetime"]
        df["elapsed_los"] = df["elapsed_los"].dt.total_seconds()

    return df


def shift_dates_into_future(df, yta, seed_path):
    # Adjust all dates to anonymise visits
    print("\nConverting dates to anonymise visits. Current min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())

    # Read the seed from a saved file
    with open(seed_path, "r") as file:
        seed = int(file.read().strip())
    # Set the seed for numpy
    np.random.seed(seed=seed)
    n = np.random.randint(1, 10 * 52)

    # print(new.snapshot_date.min())
    df.loc[:, "snapshot_date"] = df["snapshot_date"] + pd.Timedelta(days=n * 7)
    df.loc[:, "snapshot_datetime"] = df["snapshot_datetime"] + pd.Timedelta(days=n * 7)
    df.loc[:, "arrival_datetime"] = df["arrival_datetime"] + pd.Timedelta(days=n * 7)
    df.loc[:, "departure_datetime"] = df["departure_datetime"] + pd.Timedelta(
        days=n * 7
    )

    print("New min and max snapshot dates:")
    print(df.snapshot_date.min())
    print(df.snapshot_date.max())

    yta["arrival_datetime"] = yta["arrival_datetime"] + pd.Timedelta(days=n * 7)
    yta["departure_datetime"] = yta["departure_datetime"] + pd.Timedelta(days=n * 7)
    return (df, yta)


def map_consultations_to_types(df, name_mapping):
    # Create a dictionary to map consultation codes to types
    code_to_type = dict(zip(name_mapping["code"], name_mapping["type"]))

    # Define a function to map a list of consultation codes to their types
    def map_codes_to_types(codes):
        return [code_to_type.get(code, "unknown") for code in codes]

    # Apply the function to the consultations columns
    df["consultation_sequence"] = df["consultation_sequence"].apply(map_codes_to_types)
    df["final_sequence"] = df["final_sequence"].apply(map_codes_to_types)

    return df
