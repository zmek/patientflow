import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


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
def initialise_model(params):
    model = xgb.XGBClassifier(n_jobs=-1, use_label_encoder=False, eval_metric="logloss")
    model.set_params(**params)
    return model


# Set up the feature transformation
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
            # Passthrough for other types of columns (e.g., numerical)
            transformers.append((col, "passthrough", [col]))

    return ColumnTransformer(transformers)
