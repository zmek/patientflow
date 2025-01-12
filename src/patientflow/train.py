import numpy as np
from xgboost import XGBClassifier
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
    _df = df.copy()
    _df[date_column] = pd.to_datetime(_df[date_column]).dt.date

    if print_dates:
        # Separate into training, validation and test sets and print summary for each set
        for value in _df.training_validation_test.unique():
            subset = _df[_df.training_validation_test == value]
            counts = subset.training_validation_test.value_counts().values[0]
            min_date = subset[date_column].min()
            max_date = subset[date_column].max()
            print(
                f"Set: {value}\nNumber of rows: {counts}\nMin Date: {min_date}\nMax Date: {max_date}\n"
            )

    # Split df into training, validation, and test sets
    train_df = _df[_df.training_validation_test == "train"].drop(
        columns="training_validation_test"
    )
    valid_df = _df[_df.training_validation_test == "valid"].drop(
        columns="training_validation_test"
    )
    test_df = _df[_df.training_validation_test == "test"].drop(
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
    model = XGBClassifier(
        n_jobs=-1,
        eval_metric="logloss",
        # use_label_encoder=False,
        enable_categorical=True,
        # scikit_learn=True,  # Add this parameter
    )
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


def train_admissions_models(
    visits,
    grid,
    exclude_from_training_data,
    ordinal_mappings,
    prediction_times,
    model_name,
    model_metadata,
):
    # Initialize dictionary to store models
    trained_models = {}
    
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

        # initialise data used for saving attributes of the model
        model_metadata[MODEL__ED_ADMISSIONS__NAME] = {}
        best_valid_logloss = float("inf")
        results_dict = {}
        best_pipeline = None

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

            # Update best model if current model is better on validation set
            if cv_results["valid_logloss"] < best_valid_logloss:
                best_valid_logloss = cv_results["valid_logloss"]
                best_pipeline = pipeline

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

        # Store the best model for this prediction time
        trained_models[MODEL__ED_ADMISSIONS__NAME] = best_pipeline

    return model_metadata, trained_models


def train_specialty_model(visits, model_name, model_metadata, uclh):
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

    return model_metadata, spec_model


def train_yet_to_arrive_model(
    yta,
    prediction_window,
    yta_time_interval,
    prediction_times,
    epsilon,
    model_name,
    model_metadata,
    uclh,
):
    specialty_filters = create_yta_filters(uclh)

    train_yta = yta[yta.training_validation_test == "train"]
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

    return model_metadata, yta_model

def save_model(model, model_name, model_file_path):
    """
    Save trained model(s) to disk.
    
    Args:
        model: Single model or dictionary of models to save
        model_name (str): Base name for the model(s)
        model_file_path (Path): Path where model(s) should be saved
    """
    if isinstance(model, dict):
        # Handle dictionary of models (e.g., admission models)
        for name, m in model.items():
            full_path = model_file_path / name
            full_path = full_path.with_suffix(".joblib")
            dump(m, full_path)
    else:
        # Handle single model (e.g., specialty or yet-to-arrive model)
        full_path = model_file_path / model_name
        full_path = full_path.with_suffix(".joblib")
        dump(model, full_path)

def save_metadata(metadata, base_path, subdir, filename):
    """
    Save model metadata to disk.
    
    Args:
        metadata (dict): Metadata to save
        base_path (Path): Base directory path
        subdir (str, optional): Subdirectory for metadata. Defaults to "model-output"
        filename (str, optional): Name of metadata file. Defaults to "model_metadata.json"
    """
    # Construct full path
    metadata_dir = base_path / subdir if subdir else base_path
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / filename

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def test_real_time_predictions(
    visits,
    model_file_path,
    prediction_window,
    specialties,
    cdf_cut_points,
    curve_params,
    uclh,
    random_seed
):
    """
    Test real-time prediction creation using a random test set sample.
    
    Args:
        visits (pd.DataFrame): DataFrame containing visit data
        model_file_path (Path): Path where models are saved
        prediction_window (int): Window size for predictions in minutes
        specialties (list): List of specialties to consider
        cdf_cut_points (list): CDF cut points for predictions
        curve_params (tuple): Tuple of (x1, y1, x2, y2) coordinates for curve parameters
        uclh (bool): Flag for UCLH dataset usage
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing prediction time, date and results
    """
    # Select random test set row
    random_row = visits[visits.training_validation_test == "test"].sample(
        n=1, random_state=random_seed
    )
    prediction_time = random_row.prediction_time.values[0]
    prediction_date = random_row.snapshot_date.values[0]

    # Get prediction snapshots
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
        x1, y1, x2, y2 = curve_params
        realtime_preds_dict["realtime_preds"] = create_predictions(
            model_file_path=model_file_path,
            prediction_time=prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=specialties,
            prediction_window_hrs=prediction_window / 60,
            cdf_cut_points=cdf_cut_points,
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
        
    return realtime_preds_dict

def train_all_models(
    visits,
    yta,
    model_file_path,
    prediction_times,
    prediction_window,
    yta_time_interval,
    epsilon,
    curve_params,
    grid_params,
    exclude_columns,
    ordinal_mappings,
    model_names,
    specialties,
    cdf_cut_points,
    uclh,
    random_seed,
    metadata_subdir='model-output',  
    metadata_filename="model_metadata.json"
):
    """
    Main function for training and evaluating patient flow models.

    Args:
        visits (pd.DataFrame): DataFrame containing visit data
        yta (pd.DataFrame): DataFrame containing yet-to-arrive data
        model_file_path (Path): Path where models will be saved
        prediction_times (list): Times of day at which predictions will be made
        prediction_window (int): Window size for predictions in minutes
        yta_time_interval (int): Time interval for yet-to-arrive predictions
        epsilon (float): Epsilon parameter for models
        curve_params (tuple): Tuple of (x1, y1, x2, y2) coordinates for curve parameters
        grid_params (dict): XGBoost hyperparameter grid
        exclude_columns (list): Columns to exclude from training
        ordinal_mappings (dict): Mappings for ordinal variables
        model_names (dict): Names for different models
        specialties (list): List of specialties to consider
        cdf_cut_points (list): CDF cut points for predictions
        uclh (bool): Flag for UCLH dataset usage
        random_seed (int): Random seed for reproducibility

    Returns:
        dict: Model metadata including training results and predictions
    """
    # Set random seed
    np.random.seed(random_seed)

    train_dttm = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create metadata dictionary
    model_metadata = {
        "uclh": uclh,
        "train_dttm": train_dttm,
    }

    # Train admission models
    model_metadata, admission_models = train_admissions_models(
        visits=visits,
        grid=grid_params,
        exclude_from_training_data=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        prediction_times=prediction_times,
        model_name=model_names["admissions"],
        model_metadata=model_metadata,
    )
    
    # Save admission models
    save_model(admission_models, model_names["admissions"], model_file_path)

    # Train specialty model
    model_metadata, specialty_model = train_specialty_model(
        visits=visits,
        model_name=model_names["specialty"],
        model_metadata=model_metadata,
        uclh=uclh,
    )
    
    # Save specialty model
    save_model(specialty_model, model_names["specialty"], model_file_path)

    # Train yet-to-arrive model
    model_metadata, yta_model = train_yet_to_arrive_model(
        yta=yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        model_name=model_names["yet_to_arrive"],
        model_metadata=model_metadata,
        uclh=uclh,
    )
    
    # Save yet-to-arrive model with hours appended to name
    model_name = model_names["yet_to_arrive"] + str(int(prediction_window / 60)) + "_hours"
    save_model(yta_model, model_name, model_file_path)

    # Test real-time predictions
    realtime_preds_dict = test_real_time_predictions(
        visits=visits,
        model_file_path=model_file_path,
        prediction_window=prediction_window,
        specialties=specialties,
        cdf_cut_points=cdf_cut_points,
        curve_params=curve_params,
        uclh=uclh,
        random_seed=random_seed
    )

    # Save results in metadata
    model_metadata["realtime_preds"] = realtime_preds_dict
    
    # Save metadata with configurable path and filename
    save_metadata(
        metadata=model_metadata,
        base_path=model_file_path,
        subdir=metadata_subdir,
        filename=metadata_filename
    )

    return model_metadata


def main(data_folder_name=None, uclh=None):
    """
    Main entry point for training patient flow models.

    Args:
        data_folder_name (str, optional): Name of data folder
        uclh (bool, optional): Flag indicating if using UCLH dataset
    """
    # Parse arguments if not provided
    if data_folder_name is None or uclh is None:
        args = parse_args()
        data_folder_name = (
            data_folder_name if data_folder_name is not None else args.data_folder_name
        )
        uclh = uclh if uclh is not None else args.uclh

    print(f"Loading data from folder: {data_folder_name}")
    print(
        "Training models using UCLH dataset"
        if uclh
        else "Training models using public dataset"
    )

    train_dttm = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Set file locations
    data_file_path, _, model_file_path, config_path = set_file_paths(
        inference_time=False,
        train_dttm=train_dttm,
        data_folder_name=data_folder_name,
        uclh=uclh,
    )

    # Load parameters
    config = load_config_file(config_path)

    # Extract parameters
    prediction_times = config["prediction_times"]
    start_training_set = config["start_training_set"]
    start_validation_set = config["start_validation_set"]
    start_test_set = config["start_test_set"]
    end_test_set = config["end_test_set"]
    prediction_window = config["prediction_window"]
    epsilon = float(config["epsilon"])
    yta_time_interval = config["yta_time_interval"]
    x1, y1, x2, y2 = config["x1"], config["y1"], config["x2"], config["y2"]

    # Load data
    if uclh:
        _, visits_csv_path, _, yta_csv_path = set_data_file_names(
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

    # Verify data alignment
    print("\nTimes of day at which predictions will be made")
    print(prediction_times)
    print("\nNumber of rows in dataset that are not in these times of day")
    print(len(visits[~visits.prediction_time.isin(prediction_times)]))

    # Check dataset splits
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

    # Set up model parameters
    grid_params = {"n_estimators": [30], "subsample": [0.7], "colsample_bytree": [0.7]}

    exclude_columns = [
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
            "Nil",
            "Mild",
            "Moderate",
            "Severe\\E\\Very Severe",
        ],
        "latest_obs_level_of_consciousness": ["A", "C", "V", "P", "U"],
    }

    model_names = {
        "admissions": "admissions",
        "specialty": "ed_specialty",
        "yet_to_arrive": "ed_yet_to_arrive_by_spec_",
    }

    specialties = ["surgical", "haem/onc", "medical", "paediatric"]
    cdf_cut_points = [0.9, 0.7]
    curve_params = (x1, y1, x2, y2)
    random_seed = 42

    # Call train_all_models with prepared parameters
    model_metadata = train_all_models(
        visits=visits,
        yta=yta,
        model_file_path=model_file_path,
        prediction_times=prediction_times,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        epsilon=epsilon,
        curve_params=curve_params,
        grid_params=grid_params,
        exclude_columns=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        model_names=model_names,
        specialties=specialties,
        cdf_cut_points=cdf_cut_points,
        random_seed=random_seed,
        uclh=uclh,
    )

    # Add additional metadata
    model_metadata.update(
        {
            "data_folder_name": data_folder_name,
            "uclh": uclh,
            "train_dttm": train_dttm,
            "config": create_json_safe_params(config),
        }
    )

    return model_metadata


if __name__ == "__main__":
    main()
