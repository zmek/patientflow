import numpy as np
import xgboost as xgb
import pandas as pd
from joblib import dump
import json
from datetime import datetime, date
from collections import Counter
import sys

# import argparse

from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from patientflow.prepare import (
    get_snapshots_at_prediction_time,
    select_one_snapshot_per_visit,
    create_special_category_objects,
    create_yta_filters,
)
from patientflow.load import (
    load_config_file,
    get_model_name,
    set_file_paths,
    set_data_file_names,
    data_from_csv,
    parse_args,
)
from patientflow.predictors.sequence_predictor import SequencePredictor
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor
from patientflow.predict.realtime_demand import create_predictions


def split_and_check_sets(
    df,
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    date_column="snapshot_date",
    print_dates=True,
):
    df[date_column] = pd.to_datetime(df[date_column]).dt.date

    if print_dates:
        # Separate into training, validation and test sets and print summary for each set
        for value in df.training_validation_test.unique():
            subset = df[df.training_validation_test == value]
            counts = subset.training_validation_test.value_counts().values[0]
            min_date = subset[date_column].min()
            max_date = subset[date_column].max()
            print(
                f"Set: {value}\nNumber of rows: {counts}\nMin Date: {min_date}\nMax Date: {max_date}\n"
            )

    # Split df into training, validation, and test sets
    train_df = df[df.training_validation_test == "train"].drop(
        columns="training_validation_test"
    )
    valid_df = df[df.training_validation_test == "valid"].drop(
        columns="training_validation_test"
    )
    test_df = df[df.training_validation_test == "test"].drop(
        columns="training_validation_test"
    )

    # Assertions with try-except for better error handling
    try:
        assert train_df[date_column].min() == start_training_set
    except AssertionError:
        print(
            f"Assertion failed: train_df min date {train_df[date_column].min()} != {start_training_set}"
        )

    try:
        assert train_df[date_column].max() < start_validation_set
    except AssertionError:
        print(
            f"Assertion failed: train_df max date {train_df[date_column].max()} >= {start_validation_set}"
        )

    try:
        assert valid_df[date_column].min() == start_validation_set
    except AssertionError:
        print(
            f"Assertion failed: valid_df min date {valid_df[date_column].min()} != {start_validation_set}"
        )

    try:
        assert valid_df[date_column].max() < start_test_set
    except AssertionError:
        print(
            f"Assertion failed: valid_df max date {valid_df[date_column].max()} >= {start_test_set}"
        )

    try:
        assert test_df[date_column].min() == start_test_set
    except AssertionError:
        print(
            f"Assertion failed: test_df min date {test_df[date_column].min()} != {start_test_set}"
        )

    try:
        assert test_df[date_column].max() <= end_test_set
    except AssertionError:
        print(
            f"Assertion failed: test_df max date {test_df[date_column].max()} > {end_test_set}"
        )

    return


def chronological_cross_validation(pipeline, X, y, n_splits=5):
    """
    Perform time series cross-validation.

    :param pipeline: The machine learning pipeline (preprocessing + model).
    :param X: Input features.
    :param y: Target variable.
    :param n_splits: Number of splits for cross-validation.
    :return: Dictionary with the average training and validation scores.
    """
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Lists to collect scores for each fold
    train_aucs = []
    train_loglosses = []
    valid_aucs = []
    valid_loglosses = []

    # Iterate over train-test splits
    for train_index, test_index in tscv.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        # Fit the pipeline to the training data
        # Note that you don't need to manually transform the data; the pipeline handles it
        pipeline.fit(X_train, y_train)

        # # To access transformed feature names:
        # transformed_cols = pipeline.named_steps['feature_transformer'].get_feature_names_out()
        # transformed_cols = [col.split('__')[-1] for col in transformed_cols]

        # Evaluate on the training split
        y_train_pred = pipeline.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        train_logloss = log_loss(y_train, y_train_pred)
        train_aucs.append(train_auc)
        train_loglosses.append(train_logloss)

        # Evaluate on the validation split
        y_valid_pred = pipeline.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, y_valid_pred)
        valid_logloss = log_loss(y_valid, y_valid_pred)
        valid_aucs.append(valid_auc)
        valid_loglosses.append(valid_logloss)

    # Calculate mean scores
    mean_train_auc = sum(train_aucs) / n_splits
    mean_train_logloss = sum(train_loglosses) / n_splits
    mean_valid_auc = sum(valid_aucs) / n_splits
    mean_valid_logloss = sum(valid_loglosses) / n_splits

    return {
        "train_auc": mean_train_auc,
        "valid_auc": mean_valid_auc,
        "train_logloss": mean_train_logloss,
        "valid_logloss": mean_valid_logloss,
    }


# Initialise the model with given hyperparameters
def initialise_xgb(params):
    model = xgb.XGBClassifier(n_jobs=-1, eval_metric="logloss")
    model.set_params(**params)
    return model


def create_column_transformer(df, ordinal_mappings=None):
    """
    Create a column transformer for a dataframe with dynamic column handling.

    :param df: Input dataframe.
    :param ordinal_mappings: A dictionary specifying the ordinal mappings for specific columns.
    :return: A configured ColumnTransformer object.
    """
    transformers = []

    # Default to an empty dict if no ordinal mappings are provided
    if ordinal_mappings is None:
        ordinal_mappings = {}

    for col in df.columns:
        if col in ordinal_mappings:
            # Ordinal encoding for specified columns with a predefined ordering
            transformers.append(
                (
                    col,
                    OrdinalEncoder(
                        categories=[ordinal_mappings[col]],
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                    ),
                    [col],
                )
            )
        elif df[col].dtype == "object" or (
            df[col].dtype == "bool" or df[col].nunique() == 2
        ):
            # OneHotEncoding for categorical or boolean columns
            transformers.append((col, OneHotEncoder(handle_unknown="ignore"), [col]))
        else:
            # Scaling for numerical columns
            transformers.append((col, StandardScaler(), [col]))

    return ColumnTransformer(transformers)


def calculate_class_balance(y):
    counter = Counter(y)
    total = len(y)
    return {cls: count / total for cls, count in counter.items()}


def create_json_safe_params(params):
    # Create a shallow copy of the original params
    new_params = params.copy()

    # List of keys to check for date objects
    date_keys = [
        "start_training_set",
        "start_validation_set",
        "start_test_set",
        "end_test_set",
    ]

    # Convert dates to ISO format for the specified keys
    for key in date_keys:
        if key in new_params and isinstance(new_params[key], date):
            new_params[key] = new_params[key].isoformat()

    return new_params


def train_admissions_models(
    visits,
    grid,
    exclude_from_training_data,
    ordinal_mappings,
    prediction_times,
    model_name,
    model_file_path,
    model_metadata,
    filename_results_dict_name,
):
    # separate into training, validation and test sets

    train_visits = visits[visits.training_validation_test == "train"].drop(
        columns="training_validation_test"
    )
    valid_visits = visits[visits.training_validation_test == "valid"].drop(
        columns="training_validation_test"
    )
    test_visits = visits[visits.training_validation_test == "test"].drop(
        columns="training_validation_test"
    )

    # Process each time of day
    for _prediction_time in prediction_times:
        print("\nProcessing :" + str(_prediction_time))

        # create a name for the model based on the time of day it is trained for
        MODEL__ED_ADMISSIONS__NAME = get_model_name(model_name, _prediction_time)

        # use this name in the path for saving best model
        full_path = model_file_path / MODEL__ED_ADMISSIONS__NAME
        full_path = full_path.with_suffix(".joblib")

        # initialise data used for saving attributes of the model
        model_metadata[MODEL__ED_ADMISSIONS__NAME] = {}
        best_valid_logloss = float("inf")
        results_dict = {}

        # get visits that were in at the time of day in question and preprocess the training, validation and test sets
        X_train, y_train = get_snapshots_at_prediction_time(
            train_visits, _prediction_time, exclude_from_training_data
        )
        X_valid, y_valid = get_snapshots_at_prediction_time(
            valid_visits, _prediction_time, exclude_from_training_data
        )
        X_test, y_test = get_snapshots_at_prediction_time(
            test_visits, _prediction_time, exclude_from_training_data
        )

        y_train_class_balance = calculate_class_balance(y_train)
        y_valid_class_balance = calculate_class_balance(y_valid)
        y_test_class_balance = calculate_class_balance(y_test)

        # save size of each set
        model_metadata[MODEL__ED_ADMISSIONS__NAME]["train_valid_test_set_no"] = {
            "train_set_no": len(X_train),
            "valid_set_no": len(X_valid),
            "test_set_no": len(X_test),
        }

        # save class balance of each set
        model_metadata[MODEL__ED_ADMISSIONS__NAME]["train_valid_test_class_balance"] = {
            "y_train_class_balance": y_train_class_balance,
            "y_valid_class_balance": y_valid_class_balance,
            "y_test_class_balance": y_test_class_balance,
        }

        # iterate through the grid of hyperparameters
        for g in ParameterGrid(grid):
            model = initialise_xgb(g)

            # define a column transformer for the ordinal and categorical variables
            column_transformer = create_column_transformer(X_test, ordinal_mappings)

            # create a pipeline with the feature transformer and the model
            pipeline = Pipeline(
                [("feature_transformer", column_transformer), ("classifier", model)]
            )

            # cross-validate on training set using the function created earlier
            cv_results = chronological_cross_validation(
                pipeline, X_train, y_train, n_splits=5
            )

            # Store results for this set of parameters in the results dictionary
            results_dict[str(g)] = {
                "train_auc": cv_results["train_auc"],
                "valid_auc": cv_results["valid_auc"],
                "train_logloss": cv_results["train_logloss"],
                "valid_logloss": cv_results["valid_logloss"],
            }

            # Update and save best model if current model is better on validation set
            if cv_results["valid_logloss"] < best_valid_logloss:
                # save the details of the best model
                best_valid_logloss = cv_results["valid_logloss"]

                # save the best model params
                model_metadata[MODEL__ED_ADMISSIONS__NAME]["best_params"] = str(g)

                # save the model metrics on training and validation set
                model_metadata[MODEL__ED_ADMISSIONS__NAME][
                    "train_valid_set_results"
                ] = results_dict

                # score the model's performance on the test set
                y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
                test_logloss = log_loss(y_test, y_test_pred_proba)

                model_metadata[MODEL__ED_ADMISSIONS__NAME]["test_set_results"] = {
                    "test_auc": test_auc,
                    "test_logloss": test_logloss,
                }

                # save the best features
                # To access transformed feature names:
                transformed_cols = pipeline.named_steps[
                    "feature_transformer"
                ].get_feature_names_out()
                transformed_cols = [col.split("__")[-1] for col in transformed_cols]
                model_metadata[MODEL__ED_ADMISSIONS__NAME]["best_model_features"] = {
                    "feature_names": transformed_cols,
                    "feature_importances": pipeline.named_steps[
                        "classifier"
                    ].feature_importances_.tolist(),
                }

                # save the best model
                dump(pipeline, full_path)

    # save the results dictionary
    filename_results_dict_path = model_file_path / "model-output"
    full_path_results_dict = filename_results_dict_path / filename_results_dict_name

    with open(full_path_results_dict, "w") as f:
        json.dump(model_metadata, f)

    return model_metadata


def get_default_visits(admitted, uclh):
    # Get the special category objects based on the uclh flag
    special_params = create_special_category_objects(uclh)

    # Extract the function from special_params that will be used to identify the visits falling into the default category
    # ie visits that do not require special functionality (in our case, the non-paediatric patients
    opposite_special_category_func = special_params["special_func_map"]["default"]

    # Get the special handling category (e.g., "paediatric") from the dictionary
    special_category_key = next(
        key
        for key, value in special_params["special_category_dict"].items()
        if value == 1.0
    )

    # Apply the function to filter out rows where the default handling (non-paediatric) applies
    # Also, filter out rows where the 'specialty' matches the special handling category
    filtered_admitted = admitted[
        admitted.apply(opposite_special_category_func, axis=1)
        & (admitted["specialty"] != special_category_key)
    ]

    return filtered_admitted


def train_specialty_model(visits, model_name, model_metadata, model_file_path, uclh):
    # Select one snapshot per visit
    visits_single = select_one_snapshot_per_visit(visits, visit_col="visit_number")

    # Prepare dataset of admitted visits only for training specialty model
    admitted = visits_single[
        (visits_single.is_admitted) & ~(visits_single.specialty.isnull())
    ]
    filtered_admitted = get_default_visits(admitted, uclh=uclh)

    # convert consults data format from list to tuple (required input for SequencePredictor)
    filtered_admitted.loc[:, "consultation_sequence"] = filtered_admitted[
        "consultation_sequence"
    ].apply(lambda x: tuple(x) if x else ())
    filtered_admitted.loc[:, "final_sequence"] = filtered_admitted[
        "final_sequence"
    ].apply(lambda x: tuple(x) if x else ())

    # Train model
    train_visits = filtered_admitted[
        filtered_admitted.training_validation_test == "train"
    ]
    spec_model = SequencePredictor(
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
    )
    spec_model.fit(train_visits)

    model_metadata[model_name] = {}
    model_metadata[model_name]["train_set_no"] = {
        "train_set_no": len(train_visits),
    }

    # Save the model
    full_path = model_file_path / model_name
    full_path = full_path.with_suffix(".joblib")
    dump(spec_model, full_path)

    return model_metadata


def train_yet_to_arrive_model(
    yta,
    prediction_window,
    yta_time_interval,
    prediction_times,
    epsilon,
    model_name,
    model_file_path,
    model_metadata,
    uclh,
):
    specialty_filters = create_yta_filters(uclh)

    train_yta = yta[
        yta.training_validation_test == "train"
    ]  # .drop(columns='training_validation_test'
    train_yta.loc[:, "arrival_datetime"] = pd.to_datetime(
        train_yta["arrival_datetime"], utc=True
    )
    train_yta.set_index("arrival_datetime", inplace=True)

    yta_model = WeightedPoissonPredictor(filters=specialty_filters)
    yta_model.fit(
        train_df=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
    )

    model_name = model_name + str(int(prediction_window / 60)) + "_hours"

    model_metadata[model_name] = {}
    model_metadata[model_name]["train_set_no"] = {
        "train_set_no": len(train_yta),
    }

    full_path = model_file_path / model_name
    full_path = full_path.with_suffix(".joblib")

    dump(yta_model, full_path)

    return model_metadata


def main(data_folder_name=None, uclh=None):
    # parse arguments
    if data_folder_name is None or uclh is None:
        args = parse_args()
        data_folder_name = (
            data_folder_name if data_folder_name is not None else args.data_folder_name
        )
        uclh = uclh if uclh is not None else args.uclh

    # Now `data_folder_name` and `uclh` contain the appropriate values
    print(f"Loading data from folder: {data_folder_name}")
    if uclh:
        print("Training models using UCLH dataset")
    else:
        print("Training models using public dataset")

    train_dttm = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # set file location
    data_file_path, media_file_path, model_file_path, config_path = set_file_paths(
        inference_time=False,
        train_dttm=train_dttm,
        data_folder_name=data_folder_name,
        uclh=uclh,
    )

    # load parameters
    params = load_config_file(config_path)

    prediction_times = params["prediction_times"]
    start_training_set, start_validation_set, start_test_set, end_test_set = (
        params["start_training_set"],
        params["start_validation_set"],
        params["start_test_set"],
        params["end_test_set"],
    )
    x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
    prediction_window = params["prediction_window"]
    epsilon = float(params["epsilon"])
    yta_time_interval = params["yta_time_interval"]

    # convert params dates in format that can be saved to json later
    json_safe_params = create_json_safe_params(params)

    # Load data
    if uclh:
        visits_path, visits_csv_path, yta_path, yta_csv_path = set_data_file_names(
            uclh, data_file_path, config_path
        )
    else:
        visits_csv_path, yta_csv_path = set_data_file_names(uclh, data_file_path)

    visits = data_from_csv(
        visits_csv_path,
        index_column="snapshot_id",
        sort_columns=["visit_number", "snapshot_date", "prediction_time"],
        eval_columns=["prediction_time", "consultation_sequence", "final_sequence"],
    )
    yta = pd.read_csv(yta_csv_path)

    # Create snapshot date
    visits["snapshot_date"] = pd.to_datetime(visits["snapshot_date"]).dt.date

    print("\nTimes of day at which predictions will be made")
    print(prediction_times)
    print("\nNumber of rows in dataset that are not in these times of day")
    print(len(visits[~visits.prediction_time.isin(prediction_times)]))

    # Check that input data aligns with specified params in config.yaml ie training, validation and test set dates
    print("Checking dates for ed_visits dataset (used for patients in ED)")
    split_and_check_sets(
        visits, start_training_set, start_validation_set, start_test_set, end_test_set
    )
    print("Checking dates for admissions dataset (used for yet-to-arrive patients)")

    split_and_check_sets(
        yta,
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        date_column="arrival_datetime",
    )

    model_metadata = {
        "data_folder_name": data_folder_name,
        "uclh": uclh,
        "train_dttm": train_dttm,
        "config": json_safe_params,
    }
    filename_results_dict_name = "model_metadata.json"

    # Train admissions model

    # Initialize a dict to save information about the best models for each time of day
    grid = {
        "n_estimators": [30],  # , 40, 50],
        "subsample": [0.7],  # , 0.8,0.9],
        "colsample_bytree": [0.7],  # , 0.8, 0.9]
    }

    # certain columns are not used in training
    exclude_from_training_data = [
        "visit_number",
        "snapshot_date",
        "prediction_time",
        "specialty",
        "consultation_sequence",
        "final_sequence",
    ]

    ordinal_mappings = {
        "age_group": [
            "0-17",
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65-74",
            "75-102",
        ],
        "latest_acvpu": ["A", "C", "V", "P", "U"],
        "latest_obs_manchester_triage_acuity": [
            "Blue",
            "Green",
            "Yellow",
            "Orange",
            "Red",
        ],
        "latest_obs_objective_pain_score": [
            r"Nil",
            r"Mild",
            r"Moderate",
            r"Severe\E\Very Severe",
        ],
        "latest_obs_level_of_consciousness": [
            "A",  # alert
            "C",  # confused
            "V",  # voice - responds to voice stimulus
            "P",  # pain - responds to pain stimulus
            "U",  # unconscious - no response to pain or voice stimulus
        ],
    }

    # Train admission model
    model_name = "admissions"
    model_metadata = train_admissions_models(
        visits,
        grid,
        exclude_from_training_data,
        ordinal_mappings,
        prediction_times,
        model_name,
        model_file_path,
        model_metadata,
        filename_results_dict_name,
    )

    # Train specialty model
    model_name = "ed_specialty"
    model_metadata = train_specialty_model(
        visits=visits,
        model_name=model_name,
        model_metadata=model_metadata,
        model_file_path=model_file_path,
        uclh=uclh,
    )

    # Train yet-to-arrive model
    model_name = "ed_yet_to_arrive_by_spec_"
    model_metadata = train_yet_to_arrive_model(
        yta=yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        model_name=model_name,
        model_metadata=model_metadata,
        model_file_path=model_file_path,
        uclh=uclh,
    )

    # Test creation of real-time predictions
    # Randomly pick a prediction moment to do inference on
    random_row = visits[visits.training_validation_test == "test"].sample(n=1)
    prediction_time = random_row.prediction_time.values[0]
    prediction_date = random_row.snapshot_date.values[0]

    prediction_snapshots = visits[
        (visits.prediction_time == prediction_time)
        & (visits.snapshot_date == prediction_date)
    ]
    special_params = create_special_category_objects(uclh)

    realtime_preds_dict = {
        "prediction_time": str(prediction_time),
        "prediction_date": str(prediction_date),
    }

    try:
        realtime_preds_dict["realtime_preds"] = create_predictions(
            model_file_path=model_file_path,
            prediction_time=prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=["surgical", "haem/onc", "medical", "paediatric"],
            prediction_window_hrs=prediction_window / 60,
            cdf_cut_points=[0.9, 0.7],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            special_params=special_params,
        )
        print("Real-time inference ran correctly")
    except Exception as e:
        print(f"Real-time inference failed due to this error: {str(e)}")
        print(realtime_preds_dict)
        sys.exit(1)

    # save the results dictionary
    model_metadata["realtime_preds"] = realtime_preds_dict

    filename_results_dict_path = model_file_path / "model-output"
    full_path_results_dict = filename_results_dict_path / filename_results_dict_name

    with open(full_path_results_dict, "w") as f:
        json.dump(model_metadata, f)


if __name__ == "__main__":
    main()
