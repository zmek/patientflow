from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict
import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier
import pandas as pd
from pandas import DataFrame, Series
from joblib import dump
import json
from datetime import datetime, date
from collections import Counter
import sys

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from patientflow.prepare import (
    get_snapshots_at_prediction_time,
    select_one_snapshot_per_visit,
    create_special_category_objects,
    create_yta_filters,
    create_temporal_splits,
)
from patientflow.load import (
    load_config_file,
    get_model_name,
    set_file_paths,
    load_data,
    parse_args,
    set_project_root,
)
from patientflow.predictors.sequence_predictor import SequencePredictor
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor
from patientflow.predict.emergency_demand import create_predictions


def split_and_check_sets(
    df: DataFrame,
    start_training_set: date,
    start_validation_set: date,
    start_test_set: date,
    end_test_set: date,
    date_column: str = "snapshot_date",
    print_dates: bool = True,
) -> None:
    _df = df.copy()
    _df[date_column] = pd.to_datetime(_df[date_column]).dt.date

    if print_dates:
        for value in _df.training_validation_test.unique():
            subset = _df[_df.training_validation_test == value]
            counts = subset.training_validation_test.value_counts().values[0]
            min_date = subset[date_column].min()
            max_date = subset[date_column].max()
            print(
                f"Set: {value}\nNumber of rows: {counts}\nMin Date: {min_date}\nMax Date: {max_date}\n"
            )

    train_df = _df[_df.training_validation_test == "train"].drop(
        columns="training_validation_test"
    )
    valid_df = _df[_df.training_validation_test == "valid"].drop(
        columns="training_validation_test"
    )
    test_df = _df[_df.training_validation_test == "test"].drop(
        columns="training_validation_test"
    )

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


@dataclass
class MetricResults:
    """Store evaluation metrics for a single fold."""

    auc: float
    logloss: float
    auprc: float


def evaluate_predictions(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.float64]
) -> MetricResults:
    """Calculate multiple metrics for given predictions."""
    return MetricResults(
        auc=roc_auc_score(y_true, y_pred),
        logloss=log_loss(y_true, y_pred),
        auprc=average_precision_score(y_true, y_pred),
    )


def chronological_cross_validation(
    pipeline: Pipeline, X: DataFrame, y: Series, n_splits: int = 5
) -> Dict[str, float]:
    """Perform time series cross-validation with multiple metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    train_metrics: List[MetricResults] = []
    valid_metrics: List[MetricResults] = []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict_proba(X_train)[:, 1]
        valid_preds = pipeline.predict_proba(X_valid)[:, 1]

        train_metrics.append(evaluate_predictions(y_train, train_preds))
        valid_metrics.append(evaluate_predictions(y_valid, valid_preds))

    def aggregate_metrics(metrics_list: List[MetricResults]) -> Dict[str, float]:
        return {
            field: np.mean([getattr(m, field) for m in metrics_list])
            for field in MetricResults.__dataclass_fields__
        }

    train_means = aggregate_metrics(train_metrics)
    valid_means = aggregate_metrics(valid_metrics)

    return {f"train_{metric}": value for metric, value in train_means.items()} | {
        f"valid_{metric}": value for metric, value in valid_means.items()
    }


def initialise_xgb(params: Dict[str, Any]) -> XGBClassifier:
    """Initialize the model with given hyperparameters."""
    model = XGBClassifier(
        n_jobs=-1,
        eval_metric="logloss",
        enable_categorical=True,
    )
    model.set_params(**params)
    return model


def create_column_transformer(
    df: DataFrame, ordinal_mappings: Optional[Dict[str, List[Any]]] = None
) -> ColumnTransformer:
    """Create a column transformer for a dataframe with dynamic column handling."""
    transformers: List[
        Tuple[str, Union[OrdinalEncoder, OneHotEncoder, StandardScaler], List[str]]
    ] = []

    if ordinal_mappings is None:
        ordinal_mappings = {}

    for col in df.columns:
        if col in ordinal_mappings:
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
            transformers.append((col, OneHotEncoder(handle_unknown="ignore"), [col]))
        else:
            transformers.append((col, StandardScaler(), [col]))

    return ColumnTransformer(transformers)


def calculate_class_balance(y: Series) -> Dict[Any, float]:
    counter = Counter(y)
    total = len(y)
    return {cls: count / total for cls, count in counter.items()}


def create_json_safe_params(params: Dict[str, Any]) -> Dict[str, Any]:
    new_params = params.copy()
    date_keys = [
        "start_training_set",
        "start_validation_set",
        "start_test_set",
        "end_test_set",
    ]

    for key in date_keys:
        if key in new_params and isinstance(new_params[key], date):
            new_params[key] = new_params[key].isoformat()

    return new_params


def get_default_visits(admitted: DataFrame, uclh: bool) -> DataFrame:
    special_params = create_special_category_objects(uclh)
    opposite_special_category_func = special_params["special_func_map"]["default"]

    special_category_key = next(
        key
        for key, value in special_params["special_category_dict"].items()
        if value == 1.0
    )

    filtered_admitted = admitted[
        admitted.apply(opposite_special_category_func, axis=1)
        & (admitted["specialty"] != special_category_key)
    ]

    return filtered_admitted


@dataclass
class ModelResults:
    """Store model training results and metadata."""

    pipeline: Pipeline
    valid_logloss: float
    feature_names: Union[List[str], List[float]]  # Allow both types
    feature_importances: List[float]
    metrics: Dict[str, Any]


def log_if_verbose(message: str, verbose: bool = False) -> None:
    """Helper function to handle verbose logging."""
    if verbose:
        print(message)


def get_dataset_metadata(
    X_train: DataFrame,
    X_valid: DataFrame,
    X_test: DataFrame,
    y_train: Series,
    y_valid: Series,
    y_test: Series,
) -> Dict[str, Dict[str, Any]]:
    """Get dataset sizes and class balances."""
    return {
        "train_valid_test_set_no": {
            "train_set_no": len(X_train),
            "valid_set_no": len(X_valid),
            "test_set_no": len(X_test),
        },
        "train_valid_test_class_balance": {
            "y_train_class_balance": calculate_class_balance(y_train),
            "y_valid_class_balance": calculate_class_balance(y_valid),
            "y_test_class_balance": calculate_class_balance(y_test),
        },
    }


def evaluate_model(
    pipeline: Pipeline, X_test: DataFrame, y_test: Series
) -> Dict[str, float]:
    """Evaluate model on test set."""
    y_test_pred = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_auc": roc_auc_score(y_test, y_test_pred),
        "test_logloss": log_loss(y_test, y_test_pred),
        "test_auprc": average_precision_score(y_test, y_test_pred),
    }


class FeatureMetadata(TypedDict):
    feature_names: List[str]
    feature_importances: List[float]


def get_feature_metadata(
    pipeline: Pipeline,
) -> FeatureMetadata:
    """Extract feature names and importances from pipeline."""
    transformed_cols = pipeline.named_steps[
        "feature_transformer"
    ].get_feature_names_out()
    return {
        "feature_names": [col.split("__")[-1] for col in transformed_cols],
        "feature_importances": pipeline.named_steps[
            "classifier"
        ].feature_importances_.tolist(),
    }


def train_single_model(
    X_train: DataFrame,
    X_valid: DataFrame,
    X_test: DataFrame,
    y_train: Series,
    y_valid: Series,
    y_test: Series,
    grid: Dict[str, List[Any]],
    ordinal_mappings: Dict[str, List[Any]],
    verbose: bool = False,
) -> ModelResults:
    """Train a single model for one prediction time."""
    best_model = ModelResults(
        pipeline=None,
        valid_logloss=float("inf"),
        feature_names=[],
        feature_importances=[],
        metrics={},
    )
    results_dict: Dict[str, Dict[str, float]] = {}

    for params in ParameterGrid(grid):
        model = initialise_xgb(params)
        column_transformer = create_column_transformer(X_test, ordinal_mappings)
        pipeline = Pipeline(
            [("feature_transformer", column_transformer), ("classifier", model)]
        )

        cv_results = chronological_cross_validation(
            pipeline, X_train, y_train, n_splits=5
        )
        results_dict[str(params)] = cv_results

        if cv_results["valid_logloss"] < best_model.valid_logloss:
            best_model.pipeline = pipeline
            best_model.valid_logloss = cv_results["valid_logloss"]
            best_model.metrics = {
                "params": str(params),
                "train_valid_set_results": results_dict,
                "test_set_results": evaluate_model(pipeline, X_test, y_test),
            }
            best_model.feature_names = get_feature_metadata(pipeline)["feature_names"]
            best_model.feature_importances = get_feature_metadata(pipeline)[
                "feature_importances"
            ]

            if verbose:
                log_if_verbose("\nNew best model found:", verbose)
                log_if_verbose(
                    f"Valid LogLoss: {best_model.valid_logloss:.4f}", verbose
                )
                log_if_verbose(f"Valid AUPRC: {cv_results['valid_auprc']:.4f}", verbose)
                log_if_verbose(f"Valid AUC: {cv_results['valid_auc']:.4f}", verbose)

    return best_model


def train_admissions_models(
    train_visits: DataFrame,
    valid_visits: DataFrame,
    test_visits: DataFrame,
    grid: Dict[str, List[Any]],
    exclude_from_training_data: List[str],
    ordinal_mappings: Dict[str, List[Any]],
    prediction_times: List[str],
    model_name: str,
    model_metadata: Dict[str, Any],
    visit_col: str,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Pipeline]]:
    """
    Train admission prediction models for different prediction times.

    Args:
        train_visits: Training dataset
        valid_visits: Validation dataset
        test_visits: Test dataset
        grid: Hyperparameter grid
        exclude_from_training_data: Columns to exclude
        ordinal_mappings: Mappings for ordinal features
        prediction_times: List of prediction times
        model_name: Base name for the model
        model_metadata: Dictionary to store model metadata
        visit_col: Name of visit column
        verbose: Whether to print progress messages (default: False)

    Returns:
        Tuple of (model_metadata, trained_models)
    """
    trained_models: Dict[str, Pipeline] = {}

    for prediction_time in prediction_times:
        print(f"\nProcessing: {prediction_time}")
        model_key = get_model_name(model_name, prediction_time)

        # Prepare datasets
        X_train, y_train = get_snapshots_at_prediction_time(
            train_visits, prediction_time, exclude_from_training_data, visit_col
        )
        X_valid, y_valid = get_snapshots_at_prediction_time(
            valid_visits, prediction_time, exclude_from_training_data, visit_col
        )
        X_test, y_test = get_snapshots_at_prediction_time(
            test_visits, prediction_time, exclude_from_training_data, visit_col
        )

        if verbose:
            log_if_verbose(
                f"Train set size: {len(X_train)}, Positive rate: {y_train.mean():.3f}",
                verbose,
            )
            log_if_verbose(
                f"Valid set size: {len(X_valid)}, Positive rate: {y_valid.mean():.3f}",
                verbose,
            )
            log_if_verbose(
                f"Test set size: {len(X_test)}, Positive rate: {y_test.mean():.3f}",
                verbose,
            )

        # Initialize metadata for this model
        model_metadata[model_key] = get_dataset_metadata(
            X_train, X_valid, X_test, y_train, y_valid, y_test
        )

        # Train model for this prediction time
        best_model = train_single_model(
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            grid,
            ordinal_mappings,
            verbose=verbose,
        )

        # Store results
        model_metadata[model_key].update(
            {
                "best_params": best_model.metrics["params"],
                "train_valid_set_results": best_model.metrics[
                    "train_valid_set_results"
                ],
                "test_set_results": best_model.metrics["test_set_results"],
                "best_model_features": {
                    "feature_names": best_model.feature_names,
                    "feature_importances": best_model.feature_importances,
                },
            }
        )

        trained_models[model_key] = best_model.pipeline

        if verbose:
            test_metrics = best_model.metrics["test_set_results"]
            log_if_verbose(f"\nFinal model performance for {prediction_time}:", verbose)
            log_if_verbose(f"Test AUPRC: {test_metrics['test_auprc']:.4f}", verbose)
            log_if_verbose(f"Test AUC: {test_metrics['test_auc']:.4f}", verbose)
            log_if_verbose(f"Test LogLoss: {test_metrics['test_logloss']:.4f}", verbose)

    return model_metadata, trained_models


def train_specialty_model(
    train_visits: DataFrame,
    model_name: str,
    model_metadata: Dict[str, Any],
    uclh: bool,
    visit_col: str,
    input_var: str,
    grouping_var: str,
    outcome_var: str,
) -> Tuple[Dict[str, Any], SequencePredictor]:
    """Train a specialty prediction model.

    Args:
        train_visits: Training data containing visit information
        model_name: Name identifier for the model
        model_metadata: Dictionary to store model metadata
        uclh: Flag for UCLH specific processing
        visit_col: Column name containing visit identifiers
        input_var: Column name for input sequence
        grouping_var: Column name for grouping sequence
        outcome_var: Column name for target variable

    Returns:
        Tuple of updated model metadata dictionary and trained SequencePredictor model
    """
    visits_single = select_one_snapshot_per_visit(train_visits, visit_col)
    admitted = visits_single[
        (visits_single.is_admitted) & ~(visits_single.specialty.isnull())
    ]
    filtered_admitted = get_default_visits(admitted, uclh=uclh)

    filtered_admitted.loc[:, input_var] = filtered_admitted[input_var].apply(
        lambda x: tuple(x) if x else ()
    )
    filtered_admitted.loc[:, grouping_var] = filtered_admitted[grouping_var].apply(
        lambda x: tuple(x) if x else ()
    )

    spec_model = SequencePredictor(
        input_var=input_var,
        grouping_var=grouping_var,
        outcome_var=outcome_var,
    )
    spec_model.fit(filtered_admitted)
    model_metadata[model_name] = {}
    model_metadata[model_name]["train_set_no"] = {
        "train_set_no": len(filtered_admitted),
    }
    return model_metadata, spec_model


def train_yet_to_arrive_model(
    train_yta: DataFrame,
    prediction_window: int,
    yta_time_interval: int,
    prediction_times: List[float],
    epsilon: float,
    model_name: str,
    model_metadata: Dict[str, Any],
    uclh: bool,
    specialty_filters: Dict[str, Any],
    num_days: int,
) -> Tuple[Dict[str, Any], WeightedPoissonPredictor]:
    """Train a yet-to-arrive prediction model.

    Args:
        train_yta: Training data for yet-to-arrive predictions
        prediction_window: Time window for predictions
        yta_time_interval: Time interval for predictions
        prediction_times: List of prediction times
        epsilon: Epsilon parameter for model
        model_name: Name identifier for the model
        model_metadata: Dictionary to store model metadata
        uclh: Flag for UCLH specific processing
        specialty_filters: Filters for specialties
        num_days: Number of days to consider

    Returns:
        Tuple of updated model metadata dictionary and trained WeightedPoissonPredictor model
    """
    if train_yta.index.name is None:
        if "arrival_datetime" in train_yta.columns:
            train_yta.loc[:, "arrival_datetime"] = pd.to_datetime(
                train_yta["arrival_datetime"], utc=True
            )
            train_yta.set_index("arrival_datetime", inplace=True)

    elif train_yta.index.name != "arrival_datetime":
        print("Dataset needs arrival_datetime column")

    yta_model = WeightedPoissonPredictor(filters=specialty_filters)
    yta_model.fit(
        train_df=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        num_days=num_days,
    )

    model_metadata[model_name] = {}
    model_metadata[model_name]["train_set_no"] = {
        "train_set_no": len(train_yta),
    }

    return model_metadata, yta_model


def save_model(model, model_name, model_file_path):
    """
    Save trained model(s) to disk.

    Parameters
    ----------
    model : object or dict
        A single model instance or a dictionary of models to save.
    model_name : str
        Base name to use for saving the model(s).
    model_file_path : Path
        Directory path where the model(s) will be saved.

    Returns
    -------
    None
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

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to save as a JSON file.
    base_path : Path
        Base directory where the metadata will be stored.
    subdir : str, optional
        Subdirectory within the base directory for saving metadata. Defaults to "model-output".
    filename : str, optional
        Name of the metadata file. Defaults to "model_metadata.json".

    Returns
    -------
    None
    """
    # Construct full path
    metadata_dir = base_path / subdir if subdir else base_path
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / filename

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def test_real_time_predictions(
    visits,
    models,
    model_names,
    prediction_window,
    specialties,
    cdf_cut_points,
    curve_params,
    uclh,
    random_seed,
):
    """
    Test real-time predictions on a sample from a test dataset.

    Parameters
    ----------
    visits : pd.DataFrame
        DataFrame containing visit data.
    model_file_path : Path
        Path where the models are stored.
    prediction_window : int
        Size of the prediction window in minutes.
    specialties : list
        List of specialties for which predictions are made.
    cdf_cut_points : list
        Cumulative distribution function cut points for predictions.
    curve_params : tuple
        Tuple containing curve parameters (x1, y1, x2, y2).
    uclh : bool
        Indicates if the UCLH dataset is being used.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the prediction time, date, and results.
    """
    # Select random test set row
    random_row = visits.sample(n=1, random_state=random_seed)
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
            models=models,
            model_names=model_names,
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
    start_training_set,
    start_validation_set,
    start_test_set,
    end_test_set,
    yta,
    prediction_times,
    prediction_window,
    yta_time_interval,
    epsilon,
    grid_params,
    exclude_columns,
    ordinal_mappings,
    uclh,
    random_seed,
    visit_col="visit_number",
    specialties=None,
    cdf_cut_points=None,
    curve_params=None,
    model_file_path=None,
    metadata_subdir="model-output",
    metadata_filename="model_metadata.json",
    save_models=True,
    test_realtime=True,
):
    """
    Train and evaluate patient flow models.

    Parameters
    ----------
    visits : pd.DataFrame
        DataFrame containing visit data.
    yta : pd.DataFrame
        DataFrame containing yet-to-arrive data.
    prediction_times : list
        List of times for making predictions.
    prediction_window : int
        Prediction window size in minutes.
    yta_time_interval : int
        Interval size for yet-to-arrive predictions in minutes.
    epsilon : float
        Epsilon parameter for model training.
    grid_params : dict
        Hyperparameter grid for model training.
    exclude_columns : list
        Columns to exclude during training.
    ordinal_mappings : dict
        Ordinal variable mappings for categorical features.
    uclh : bool
        Indicates if the UCLH dataset is used.
    random_seed : int
        Random seed for reproducibility.
    visit_col : str, optional
        Name of column in dataset that is used to identify a hospital visit (eg visit_number, csn).
    specialties : list, optional
        List of specialties to consider. Required if test_realtime is True.
    cdf_cut_points : list, optional
        CDF cut points for predictions. Required if test_realtime is True.
    curve_params : tuple, optional
        Curve parameters (x1, y1, x2, y2). Required if test_realtime is True.
    model_file_path : Path, optional
        Path to save trained models. Required if save_models is True.
    metadata_subdir : str, optional
        Subdirectory for metadata. Only used if save_models is True. Defaults to "model-output".
    metadata_filename : str, optional
        Metadata filename. Only used if save_models is True. Defaults to "model_metadata.json".
    save_models : bool, optional
        Whether to save the trained models to disk. Defaults to True.
    test_realtime : bool, optional
        Whether to run real-time prediction tests. Defaults to True.

    Returns
    -------
    dict or tuple
        If save_models is True, returns a dict with model metadata including training and evaluation details.
        If save_models is False, returns a tuple (model_metadata, models) where models is a dict of trained models.

    Raises
    ------
    ValueError
        If save_models is True but model_file_path is not provided,
        or if test_realtime is True but any of specialties, cdf_cut_points, or curve_params are not provided.

    Notes
    -----
    The function generates model names internally:
    - "admissions": "admissions"
    - "specialty": "ed_specialty"
    - "yet_to_arrive": f"yet_to_arrive_{int(prediction_window/60)}_hours"
    """
    # Validate parameters
    if save_models and model_file_path is None:
        raise ValueError("model_file_path must be provided when save_models is True")

    if test_realtime:
        if specialties is None:
            raise ValueError("specialties must be provided when test_realtime is True")
        if cdf_cut_points is None:
            raise ValueError(
                "cdf_cut_points must be provided when test_realtime is True"
            )
        if curve_params is None:
            raise ValueError("curve_params must be provided when test_realtime is True")

    # Set random seed
    np.random.seed(random_seed)

    train_dttm = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create metadata dictionary
    model_metadata = {
        "uclh": uclh,
        "train_dttm": train_dttm,
    }

    # Define model names internally
    model_names = {
        "admissions": "admissions",
        "specialty": "ed_specialty",
        "yet_to_arrive": f"yet_to_arrive_{int(prediction_window/60)}_hours",
    }

    # Add model names to metadata
    model_metadata["model_names"] = model_names

    models = dict.fromkeys(model_names)

    if "arrival_datetime" in visits.columns:
        col_name = "arrival_datetime"
    else:
        col_name = "snapshot_date"

    train_visits, valid_visits, test_visits = create_temporal_splits(
        visits,
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        col_name=col_name,
    )

    train_yta, _, _ = create_temporal_splits(
        yta[(~yta.specialty.isnull())],
        start_training_set,
        start_validation_set,
        start_test_set,
        end_test_set,
        col_name="arrival_datetime",
    )

    # Use predicted_times from visits if not explicitly provided
    if prediction_times is None:
        prediction_times = visits.prediction_time.unique()

    # Train admission models
    model_metadata, admission_models = train_admissions_models(
        train_visits=train_visits,
        valid_visits=valid_visits,
        test_visits=test_visits,
        grid=grid_params,
        exclude_from_training_data=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        prediction_times=prediction_times,
        model_name=model_names["admissions"],
        model_metadata=model_metadata,
        visit_col=visit_col,
    )

    # Save admission models if requested
    models[model_names["admissions"]] = admission_models
    if save_models:
        save_model(admission_models, model_names["admissions"], model_file_path)

    # Train specialty model
    model_metadata, specialty_model = train_specialty_model(
        train_visits=train_visits,
        model_name=model_names["specialty"],
        model_metadata=model_metadata,
        uclh=uclh,
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
        visit_col=visit_col,
    )

    # Save specialty model if requested
    models[model_names["specialty"]] = specialty_model
    if save_models:
        save_model(specialty_model, model_names["specialty"], model_file_path)

    # Train yet-to-arrive model
    specialty_filters = create_yta_filters(uclh)
    yta_model_name = model_names["yet_to_arrive"]

    num_days = (start_validation_set - start_training_set).days

    model_metadata, yta_model = train_yet_to_arrive_model(
        train_yta=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        model_name=yta_model_name,
        model_metadata=model_metadata,
        uclh=uclh,
        specialty_filters=specialty_filters,
        num_days=num_days,
    )

    # Save yet-to-arrive model if requested
    models[model_names["yet_to_arrive"]] = yta_model
    if save_models:
        save_model(yta_model, yta_model_name, model_file_path)

    # Test real-time predictions if requested
    realtime_preds_dict = None
    if test_realtime:
        realtime_preds_dict = test_real_time_predictions(
            visits=visits,
            models=models,
            model_names=model_names,
            prediction_window=prediction_window,
            specialties=specialties,
            cdf_cut_points=cdf_cut_points,
            curve_params=curve_params,
            uclh=uclh,
            random_seed=random_seed,
        )

        # Save results in metadata
        model_metadata["realtime_preds"] = realtime_preds_dict

    # Save metadata with configurable path and filename
    if save_models:
        save_metadata(
            metadata=model_metadata,
            base_path=model_file_path,
            subdir=metadata_subdir,
            filename=metadata_filename,
        )

    # Return both models and metadata if not saving to disk
    if not save_models:
        return model_metadata, models

    return model_metadata


def main(data_folder_name=None):
    """
    Main entry point for training patient flow models.

    Args:
        data_folder_name (str, optional): Name of data folder
    """
    # Parse arguments if not provided
    if data_folder_name is None:
        args = parse_args()
        data_folder_name = (
            data_folder_name if data_folder_name is not None else args.data_folder_name
        )
    print(f"Loading data from folder: {data_folder_name}")

    train_dttm = datetime.now().strftime("%Y-%m-%d-%H-%M")
    project_root = set_project_root()

    # Set file locations
    data_file_path, _, model_file_path, config_path = set_file_paths(
        project_root=project_root,
        inference_time=False,
        train_dttm=None,
        data_folder_name=data_folder_name,
        config_file="config.yaml",
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
    ed_visits = load_data(
        data_file_path=data_file_path,
        file_name="ed_visits.csv",
        index_column="snapshot_id",
        sort_columns=["visit_number", "snapshot_date", "prediction_time"],
        eval_columns=["prediction_time", "consultation_sequence", "final_sequence"],
    )
    inpatient_arrivals = load_data(
        data_file_path=data_file_path, file_name="inpatient_arrivals.csv"
    )

    # Create snapshot date
    ed_visits["snapshot_date"] = pd.to_datetime(ed_visits["snapshot_date"]).dt.date

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
        visits=ed_visits,
        start_training_set=start_training_set,
        start_validation_set=start_validation_set,
        start_test_set=start_test_set,
        end_test_set=end_test_set,
        yta=inpatient_arrivals,
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
        uclh=False,
    )

    # Add additional metadata
    model_metadata.update(
        {
            "data_folder_name": data_folder_name,
            "uclh": False,
            "train_dttm": train_dttm,
            "config": create_json_safe_params(config),
        }
    )

    return model_metadata


if __name__ == "__main__":
    main()
