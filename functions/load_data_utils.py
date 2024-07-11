from load_config import load_config_file


def set_file_locations(uclh, data_path, config_file_path=None):
    if not uclh:
        csv_filename = "ed_visits.csv"
        yta_csv_filename = "yet_to_arrive.csv"

        visits_csv_path = data_path / csv_filename
        yta_csv_path = data_path / yta_csv_filename

        return visits_csv_path, yta_csv_path

    else:
        start_date, end_date = load_config_file(
            config_file_path, return_start_end_dates=True
        )
        data_filename = (
            "uclh_visits_exc_beds_inc_minority_"
            + start_date
            + "_"
            + end_date
            + ".pickle"
        )
        csv_filename = "uclh_visits.csv"
        yta_filename = "uclh_yet_to_arrive_" + start_date + "_" + end_date + ".pickle"
        yta_csv_filename = "uclh_yet_to_arrive.csv"

        visits_path = data_path / data_filename
        yta_path = data_path / yta_filename

        visits_csv_path = data_path / csv_filename
        yta_csv_path = data_path / yta_csv_filename

    return visits_path, visits_csv_path, yta_path, yta_csv_path


def get_dict_cols(df):
    not_used_in_training_vars = [
        "snapshot_id",
        "snapshot_date",
        "prediction_time",
        "visit_number",
        "training_validation_test",
        "random_number",
    ]
    arrival_and_demographic_vars = [
        "elapsed_los",
        "sex",
        "age_group",
        "age_on_arrival",
        "arrival_method",
    ]
    summary_vars = [
        "num_obs",
        "num_obs_events",
        "num_obs_types",
        "num_lab_batteries_ordered",
    ]

    location_vars = []
    observations_vars = []
    labs_vars = []
    consults_vars = [
        "has_consultation",
        "consultation_sequence",
        "final_sequence",
        "specialty",
    ]
    outcome_vars = ["is_admitted"]

    for col in df.columns:
        if (
            col in not_used_in_training_vars
            or col in arrival_and_demographic_vars
            or col in summary_vars
        ):
            continue
        elif "visited" in col or "location" in col:
            location_vars.append(col)
        elif "num_obs" in col or "latest_obs" in col:
            observations_vars.append(col)
        elif "lab_orders" in col or "latest_lab_results" in col:
            labs_vars.append(col)
        elif col in consults_vars or col in outcome_vars:
            continue  # Already categorized
        else:
            print(f"Column '{col}' did not match any predefined group")

    # Create a list of column groups
    col_group_names = [
        "not used in training",
        "arrival and demographic",
        "summary",
        "location",
        "observations",
        "lab orders and results",
        "consults",
        "outcome",
    ]

    # Create a list of the column names within those groups
    col_groups = [
        not_used_in_training_vars,
        arrival_and_demographic_vars,
        summary_vars,
        location_vars,
        observations_vars,
        labs_vars,
        consults_vars,
        outcome_vars,
    ]

    # Use dictionary to combine them
    dict_col_groups = {
        category: var_list for category, var_list in zip(col_group_names, col_groups)
    }

    return dict_col_groups
