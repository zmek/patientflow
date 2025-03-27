from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, Type
import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier
import pandas as pd
from pandas import DataFrame, Series
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.load import get_model_key
from patientflow.metrics import FoldResults, TrainingResults, TrainedClassifier


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