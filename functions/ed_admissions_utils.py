from joblib import load


def get_model_name(model_name, prediction_time_):
    """
    Create a model name based on the time of day.

    Parameters:
    prediction_time_ (tuple): A tuple representing the time of day (hour, minute).

    Returns:
    str: A string representing the model name based on the time of day.
    """
    hour_, min_ = prediction_time_
    min_ = f"{min_}0" if min_ % 60 == 0 else str(min_)
    model_name = model_name + "_" + f"{hour_:02}" + min_
    return model_name


def select_one_snapshot_per_visit(df):
    max_indices = df.groupby("visit_number")["random_number"].idxmax()
    return df.loc[max_indices].drop(columns=["random_number"])


def preprocess_data(
    df, prediction_time_, exclude_columns, single_snapshot_per_visit=True
):
    # Filter by the time of day while keeping the original index
    df_tod = df[df["prediction_time"] == prediction_time_].copy()

    if single_snapshot_per_visit:
        # Group by 'visit_number' and get the row with the maximum 'random_number'
        df_single = select_one_snapshot_per_visit(df_tod)

        # Create label array with the same index
        y = df_single.pop("is_admitted").astype(int)

        # Drop specified columns and ensure we do not reset the index
        df_single.drop(columns=exclude_columns, inplace=True)

        return df_single, y

    else:
        # Directly modify df_tod without resetting the index
        df_tod.drop(columns=["random_number"] + exclude_columns, inplace=True)
        y = df_tod.pop("is_admitted").astype(int)

        return df_tod, y

    # include one one snapshot per visit and drop the random number


def load_saved_model(model_file_path, model_name, prediction_time=None):
    if prediction_time:
        # retrieve model based on the time of day it is trained for
        model_name = get_model_name(model_name, prediction_time)

    full_path = model_file_path / model_name
    full_path = full_path.with_suffix(".joblib")
    model = load(full_path)

    return model
