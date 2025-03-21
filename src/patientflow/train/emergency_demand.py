from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, Type
import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier
import pandas as pd
from pandas import DataFrame, Series
from joblib import dump
from datetime import date
from collections import Counter
import sys

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from patientflow.prepare import (
    get_snapshots_at_prediction_time,
    select_one_snapshot_per_visit,
    create_special_category_objects,
    create_yta_filters,
    create_temporal_splits,
)
from patientflow.load import (
    load_config_file,
    get_model_key,
    set_file_paths,
    load_data,
    parse_args,
    set_project_root,
)
from patientflow.predictors.sequence_predictor import SequencePredictor
from patientflow.predictors.weighted_poisson_predictor import WeightedPoissonPredictor
from patientflow.predict.emergency_demand import create_predictions
from patientflow.metrics import FoldResults, TrainingResults, TrainedClassifier


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


def evaluate_predictions(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.float64]
) -> FoldResults:
    """Calculate multiple metrics for given predictions."""
    return FoldResults(
        auc=roc_auc_score(y_true, y_pred),
        logloss=log_loss(y_true, y_pred),
        auprc=average_precision_score(y_true, y_pred),
    )


def chronological_cross_validation(
    pipeline: Pipeline, X: DataFrame, y: Series, n_splits: int = 5
) -> Dict[str, float]:
    """Perform time series cross-validation with multiple metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    train_metrics: List[FoldResults] = []
    valid_metrics: List[FoldResults] = []

    for train_idx, valid_idx in tscv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline.fit(X_train, y_train)
        train_preds = pipeline.predict_proba(X_train)[:, 1]
        valid_preds = pipeline.predict_proba(X_valid)[:, 1]

        train_metrics.append(evaluate_predictions(y_train, train_preds))
        valid_metrics.append(evaluate_predictions(y_valid, valid_preds))

    def aggregate_metrics(metrics_list: List[FoldResults]) -> Dict[str, float]:
        return {
            field: np.mean([getattr(m, field) for m in metrics_list])
            for field in FoldResults.__dataclass_fields__
        }

    train_means = aggregate_metrics(train_metrics)
    valid_means = aggregate_metrics(valid_metrics)

    return {f"train_{metric}": value for metric, value in train_means.items()} | {
        f"valid_{metric}": value for metric, value in valid_means.items()
    }


def initialise_model(
    model_class: Type,
    params: Dict[str, Any],
    xgb_specific_params: Dict[str, Any] = {
        "n_jobs": -1,
        "eval_metric": "logloss",
        "enable_categorical": True,
    },
) -> Any:
    """
    Initialize a model with given hyperparameters.

    Parameters
    ----------
    model_class : Type
        The classifier class to instantiate
    params : Dict[str, Any]
        Model-specific parameters to set
    xgb_specific_params : Dict[str, Any], optional
        XGBoost-specific default parameters

    Returns
    -------
    Any
        Initialized model instance
    """
    if model_class == XGBClassifier:
        model = model_class(**xgb_specific_params)
        model.set_params(**params)
    else:
        model = model_class(**params)

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


def get_default_visits(admitted: DataFrame) -> DataFrame:
    """
    Filters a dataframe of patient visits to include only non-pediatric patients.

    This function identifies and removes pediatric patients from the dataset based on
    both age criteria and specialty assignment. It automatically detects the appropriate
    age column format from the provided dataframe.

    Parameters:
    -----------
    admitted : DataFrame
        A pandas DataFrame containing patient visit information. Must include either
        'age_on_arrival' or 'age_group' columns, and a 'specialty' column.

    Returns:
    --------
    DataFrame
        A filtered DataFrame containing only non-pediatric patients (adults).

    Notes:
    ------
    The function automatically detects which age-related columns are present in the
    dataframe and configures the appropriate filtering logic. It removes patients who
    are either:
    1. Identified as pediatric based on age criteria, or
    2. Assigned to a pediatric specialty

    Examples:
    ---------
    >>> adult_visits = get_default_visits(all_patient_visits)
    >>> print(f"Reduced from {len(all_patient_visits)} to {len(adult_visits)} adult visits")
    """
    # Get configuration for categorizing patients based on age columns
    special_params = create_special_category_objects(admitted.columns)

    # Extract function that identifies non-pediatric patients
    opposite_special_category_func = special_params["special_func_map"]["default"]

    # Determine which category is the special category (should be "paediatric")
    special_category_key = next(
        key
        for key, value in special_params["special_category_dict"].items()
        if value == 1.0
    )

    # Filter out pediatric patients based on both age criteria and specialty
    filtered_admitted = admitted[
        admitted.apply(opposite_special_category_func, axis=1)
        & (admitted["specialty"] != special_category_key)
    ]

    return filtered_admitted


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


def create_balance_info(
    is_balanced: bool,
    original_size: int,
    balanced_size: int,
    original_positive_rate: float,
    balanced_positive_rate: float,
    majority_to_minority_ratio: float,
) -> Dict[str, Union[bool, int, float]]:
    """Create a dictionary with balance information."""
    return {
        "is_balanced": is_balanced,
        "original_size": original_size,
        "balanced_size": balanced_size,
        "original_positive_rate": original_positive_rate,
        "balanced_positive_rate": balanced_positive_rate,
        "majority_to_minority_ratio": majority_to_minority_ratio,
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


def get_feature_metadata(pipeline: Pipeline) -> FeatureMetadata:
    """
    Extract feature names and importances from pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline containing feature transformer and classifier

    Returns
    -------
    FeatureMetadata
        Dictionary containing feature names and their importance scores (if available)

    Raises
    ------
    AttributeError
        If the classifier doesn't support feature importance
    """
    transformed_cols = pipeline.named_steps[
        "feature_transformer"
    ].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    # Try different common feature importance attributes
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = (
            np.abs(classifier.coef_[0])
            if classifier.coef_.ndim > 1
            else np.abs(classifier.coef_)
        )
    else:
        raise AttributeError("Classifier doesn't provide feature importance scores")

    return {
        "feature_names": [col.split("__")[-1] for col in transformed_cols],
        "feature_importances": importances.tolist(),
    }


def train_classifier(
    train_visits: DataFrame,
    valid_visits: DataFrame,
    test_visits: DataFrame,
    prediction_time: Tuple[int, int],
    exclude_from_training_data: List[str],
    grid: Dict[str, List[Any]],
    ordinal_mappings: Dict[str, List[Any]],
    visit_col: str,
    model_class: Type = XGBClassifier,
    use_balanced_training: bool = True,
    majority_to_minority_ratio: float = 1.0,
    calibrate_probabilities: bool = True,
    calibration_method: str = "isotonic",
) -> TrainedClassifier:
    """
    Train a single model including data preparation and balancing.

    Parameters:
    -----------
    train_visits : DataFrame
        Training visits dataset
    valid_visits : DataFrame
        Validation visits dataset
    test_visits : DataFrame
        Test visits dataset
    prediction_time : Tuple[int, int]
        The prediction time point to use
    exclude_from_training_data : List[str]
        Columns to exclude from training
    grid : Dict[str, List[Any]]
        Parameter grid for hyperparameter tuning
    ordinal_mappings : Dict[str, List[Any]]
        Mappings for ordinal categorical features
    visit_col : str
        Name of the visit column
    model_class : Type, optional
        The classifier class to use. Must be sklearn-compatible with fit() and predict_proba().
        Defaults to XGBClassifier.
    use_balanced_training : bool, default=True
        Whether to use balanced training data
    majority_to_minority_ratio : float, default=1.0
        Ratio of majority to minority class samples
    calibrate_probabilities : bool, default=True
        Whether to apply probability calibration to the best model
    calibration_method : str, default='isotonic'
        Method for probability calibration ('isotonic' or 'sigmoid')

    Returns:
    --------
    TrainedClassifier
        Trained model, including metrics, and feature information
    """
    # Get snapshots for each set
    X_train, y_train = get_snapshots_at_prediction_time(
        train_visits, prediction_time, exclude_from_training_data, visit_col=visit_col
    )
    X_valid, y_valid = get_snapshots_at_prediction_time(
        valid_visits, prediction_time, exclude_from_training_data, visit_col=visit_col
    )
    X_test, y_test = get_snapshots_at_prediction_time(
        test_visits, prediction_time, exclude_from_training_data, visit_col=visit_col
    )

    # Get dataset metadata before any balancing
    dataset_metadata = get_dataset_metadata(
        X_train, X_valid, X_test, y_train, y_valid, y_test
    )

    # Store original size and positive rate before any balancing
    original_size = len(X_train)
    original_positive_rate = y_train.mean()

    if use_balanced_training:
        pos_indices = y_train[y_train == 1].index
        neg_indices = y_train[y_train == 0].index

        n_pos = len(pos_indices)
        n_neg = int(n_pos * majority_to_minority_ratio)

        neg_indices_sampled = np.random.choice(
            neg_indices, size=min(n_neg, len(neg_indices)), replace=False
        )

        train_balanced_indices = np.concatenate([pos_indices, neg_indices_sampled])
        np.random.shuffle(train_balanced_indices)

        X_train = X_train.loc[train_balanced_indices]
        y_train = y_train.loc[train_balanced_indices]

    # Create balance info after any balancing is done
    balance_info = create_balance_info(
        is_balanced=use_balanced_training,
        original_size=original_size,
        balanced_size=len(X_train),
        original_positive_rate=original_positive_rate,
        balanced_positive_rate=y_train.mean(),
        majority_to_minority_ratio=majority_to_minority_ratio
        if use_balanced_training
        else 1.0,
    )

    # Initialize best training results with default values
    best_training = TrainingResults(
        valid_logloss=float("inf"),
        feature_names=[],
        feature_importances=[],
        metadata={},
        balance_info=balance_info,
        prediction_time=prediction_time,
    )

    # Initialize best model container
    best_model = TrainedClassifier(
        metrics=best_training,
        pipeline=None,
        calibrated_pipeline=None,
    )

    results_dict: Dict[str, Dict[str, float]] = {}

    for params in ParameterGrid(grid):
        # Initialize model based on provided class
        model = initialise_model(model_class, params)

        column_transformer = create_column_transformer(X_train, ordinal_mappings)
        pipeline = Pipeline(
            [("feature_transformer", column_transformer), ("classifier", model)]
        )

        cv_results = chronological_cross_validation(
            pipeline, X_train, y_train, n_splits=5
        )
        results_dict[str(params)] = cv_results

        if cv_results["valid_logloss"] < best_training.valid_logloss:
            best_model.pipeline = pipeline

            # Get feature metadata if available
            try:
                feature_metadata = get_feature_metadata(pipeline)
                has_feature_importance = True
            except (AttributeError, NotImplementedError):
                feature_metadata = {
                    "feature_names": column_transformer.get_feature_names_out().tolist(),
                    "feature_importances": [],
                }
                has_feature_importance = False

            # Update training results
            best_training.valid_logloss = cv_results["valid_logloss"]
            best_training.feature_names = feature_metadata["feature_names"]
            best_training.feature_importances = feature_metadata["feature_importances"]
            best_training.metadata = {
                "best_params": str(params),
                "train_valid_set_results": results_dict,
                "training_balance_info": balance_info,
                "best_model_features": feature_metadata,
                "dataset_metadata": dataset_metadata,
                "has_feature_importance": has_feature_importance,
            }

            if calibrate_probabilities:
                best_training.metadata["calibration_method"] = calibration_method

    # Apply probability calibration to the best model if requested
    if calibrate_probabilities and best_model.pipeline is not None:
        best_feature_transformer = best_model.pipeline.named_steps[
            "feature_transformer"
        ]
        best_classifier = best_model.pipeline.named_steps["classifier"]

        X_valid_transformed = best_feature_transformer.transform(X_valid)

        calibrated_classifier = CalibratedClassifierCV(
            estimator=best_classifier,
            method=calibration_method,
            cv="prefit",
        )
        calibrated_classifier.fit(X_valid_transformed, y_valid)

        calibrated_pipeline = Pipeline(
            [
                ("feature_transformer", best_feature_transformer),
                ("classifier", calibrated_classifier),
            ]
        )

        best_model.calibrated_pipeline = calibrated_pipeline
        best_training.metadata["test_set_results"] = evaluate_model(
            calibrated_pipeline, X_test, y_test
        )

    else:
        best_training.metadata["test_set_results"] = evaluate_model(
            best_model.pipeline, X_test, y_test
        )

    return best_model


def train_multiple_classifiers(
    train_visits: DataFrame,
    valid_visits: DataFrame,
    test_visits: DataFrame,
    grid: Dict[str, List[Any]],
    exclude_from_training_data: List[str],
    ordinal_mappings: Dict[str, List[Any]],
    prediction_times: List[Tuple[int, int]],
    model_name: str = "admissions",
    visit_col: str = "visit_number",
    calibrate_probabilities: bool = True,
    calibration_method: str = "isotonic",
    use_balanced_training: bool = True,
    majority_to_minority_ratio: float = 1.0,
) -> Dict[str, TrainedClassifier]:
    """Train admission prediction models for multiple prediction times."""
    trained_models: Dict[str, TrainedClassifier] = {}

    for prediction_time in prediction_times:
        print(f"\nProcessing: {prediction_time}")
        model_key = get_model_key(model_name, prediction_time)

        # Train model with the new simplified interface
        best_model = train_classifier(
            train_visits,
            valid_visits,
            test_visits,
            prediction_time,
            exclude_from_training_data,
            grid,
            ordinal_mappings,
            visit_col,
            use_balanced_training=use_balanced_training,
            majority_to_minority_ratio=majority_to_minority_ratio,
            calibrate_probabilities=calibrate_probabilities,
            calibration_method=calibration_method,
        )

        trained_models[model_key] = best_model

    return trained_models


def train_specialty_model(
    train_visits: DataFrame,
    model_name: str,
    visit_col: str,
    input_var: str,
    grouping_var: str,
    outcome_var: str,
) -> Tuple[Dict[str, Any], SequencePredictor]:
    """Train a specialty prediction model.

    Args:
        train_visits: Training data containing visit information
        model_name: Name identifier for the model
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
    filtered_admitted = get_default_visits(admitted)

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

    return spec_model


def train_yet_to_arrive_model(
    train_visits: DataFrame,
    train_yta: DataFrame,
    prediction_window: int,
    yta_time_interval: int,
    prediction_times: List[float],
    epsilon: float,
    num_days: int,
) -> WeightedPoissonPredictor:
    """Train a yet-to-arrive prediction model.

    Args:
        ed_visits: Visits dataset (used for identifying special categories)
        train_yta: Training data for yet-to-arrive predictions
        prediction_window: Time window for predictions
        yta_time_interval: Time interval for predictions
        prediction_times: List of prediction times
        epsilon: Epsilon parameter for model
        model_name: Name identifier for the model
        uclh: Flag for UCLH specific processing
        specialty_filters: Filters for specialties
        num_days: Number of days to consider

    Returns:
        Trained WeightedPoissonPredictor model
    """
    if train_yta.index.name is None:
        if "arrival_datetime" in train_yta.columns:
            train_yta.loc[:, "arrival_datetime"] = pd.to_datetime(
                train_yta["arrival_datetime"], utc=True
            )
            train_yta.set_index("arrival_datetime", inplace=True)

    elif train_yta.index.name != "arrival_datetime":
        print("Dataset needs arrival_datetime column")

    specialty_filters = create_yta_filters(train_visits)

    yta_model = WeightedPoissonPredictor(filters=specialty_filters)
    yta_model.fit(
        train_df=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        num_days=num_days,
    )

    return yta_model


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


def test_real_time_predictions(
    visits,
    models: Tuple[
        Dict[str, TrainedClassifier], SequencePredictor, WeightedPoissonPredictor
    ],
    prediction_window,
    specialties,
    cdf_cut_points,
    curve_params,
    random_seed,
):
    """
    Test real-time predictions by selecting a random sample from the visits dataset
    and generating predictions using the trained models.

    Parameters
    ----------
    visits : pd.DataFrame
        DataFrame containing visit data with columns including 'prediction_time',
        'snapshot_date', and other required features for predictions.
    models : Tuple[Dict[str, TrainedClassifier], SequencePredictor, WeightedPoissonPredictor]
        Tuple containing:
        - trained_classifiers: TrainedClassifier containing admission predictions
        - spec_model: SequencePredictor for specialty predictions
        - yet_to_arrive_model: WeightedPoissonPredictor for yet-to-arrive predictions
    prediction_window : int
        Size of the prediction window in minutes for which to generate forecasts.
    specialties : list[str]
        List of specialty names to generate predictions for (e.g., ['surgical',
        'medical', 'paediatric']).
    cdf_cut_points : list[float]
        List of probability thresholds for cumulative distribution function
        cut points (e.g., [0.9, 0.7]).
    curve_params : tuple[float, float, float, float]
        Parameters (x1, y1, x2, y2) defining the curve used for predictions.
    random_seed : int
        Random seed for reproducible sampling of test cases.

    Returns
    -------
    dict
        Dictionary containing:
        - 'prediction_time': str, The time point for which predictions were made
        - 'prediction_date': str, The date for which predictions were made
        - 'realtime_preds': dict, The generated predictions for the sample

    Raises
    ------
    Exception
        If real-time inference fails, with detailed error message printed before
        system exit.

    Notes
    -----
    The function selects a single random row from the visits DataFrame and
    generates predictions for that specific time point using all provided models.
    The predictions are made using the create_predictions() function with the
    specified parameters.
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

    trained_classifiers, spec_model, yet_to_arrive_model = models

    # Find the model matching the required prediction time
    classifier = None
    for model_key, trained_model in trained_classifiers.items():
        if trained_model.metrics.prediction_time == prediction_time:
            classifier = trained_model
            break

    if classifier is None:
        raise ValueError(f"No model found for prediction time {prediction_time}")

    try:
        x1, y1, x2, y2 = curve_params
        _ = create_predictions(
            models=(classifier, spec_model, yet_to_arrive_model),
            prediction_time=prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=specialties,
            prediction_window_hrs=prediction_window / 60,
            cdf_cut_points=cdf_cut_points,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        print("Real-time inference ran correctly")
    except Exception as e:
        print(f"Real-time inference failed due to this error: {str(e)}")
        sys.exit(1)

    return


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
    random_seed,
    visit_col="visit_number",
    specialties=None,
    cdf_cut_points=None,
    curve_params=None,
    model_file_path=None,
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
    save_models : bool, optional
        Whether to save the trained models to disk. Defaults to True.
    test_realtime : bool, optional
        Whether to run real-time prediction tests. Defaults to True.

    Returns
    -------
    None

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

    # Define model names internally
    model_names = {
        "admissions": "admissions",
        "specialty": "ed_specialty",
        "yet_to_arrive": f"yet_to_arrive_{int(prediction_window/60)}_hours",
    }

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
    admission_models = train_multiple_classifiers(
        train_visits=train_visits,
        valid_visits=valid_visits,
        test_visits=test_visits,
        grid=grid_params,
        exclude_from_training_data=exclude_columns,
        ordinal_mappings=ordinal_mappings,
        prediction_times=prediction_times,
        model_name=model_names["admissions"],
        visit_col=visit_col,
    )

    # Save admission models if requested

    if save_models:
        save_model(admission_models, model_names["admissions"], model_file_path)

    # Train specialty model
    specialty_model = train_specialty_model(
        train_visits=train_visits,
        model_name=model_names["specialty"],
        input_var="consultation_sequence",
        grouping_var="final_sequence",
        outcome_var="specialty",
        visit_col=visit_col,
    )

    # Save specialty model if requested
    if save_models:
        save_model(specialty_model, model_names["specialty"], model_file_path)

    # Train yet-to-arrive model
    yta_model_name = model_names["yet_to_arrive"]

    num_days = (start_validation_set - start_training_set).days

    yta_model = train_yet_to_arrive_model(
        train_visits=train_visits,
        train_yta=train_yta,
        prediction_window=prediction_window,
        yta_time_interval=yta_time_interval,
        prediction_times=prediction_times,
        epsilon=epsilon,
        num_days=num_days,
    )

    # Save yet-to-arrive model if requested
    if save_models:
        save_model(yta_model, yta_model_name, model_file_path)
        print(f"Models have been saved to {model_file_path}")

    # Test real-time predictions if requested
    if test_realtime:
        test_real_time_predictions(
            visits=visits,
            models=(admission_models, specialty_model, yta_model),
            prediction_window=prediction_window,
            specialties=specialties,
            cdf_cut_points=cdf_cut_points,
            curve_params=curve_params,
            random_seed=random_seed,
        )

    return


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

    specialties = ["surgical", "haem/onc", "medical", "paediatric"]
    cdf_cut_points = [0.9, 0.7]
    curve_params = (x1, y1, x2, y2)
    random_seed = 42

    # Call train_all_models with prepared parameters
    train_all_models(
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
        specialties=specialties,
        cdf_cut_points=cdf_cut_points,
        random_seed=random_seed,
    )

    return


if __name__ == "__main__":
    main()
