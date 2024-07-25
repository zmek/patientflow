import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)
from joblib import dump
import json

from ed_admissions_utils import get_model_name, preprocess_data


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


def train_models(visits, 
                grid, 
                exclude_from_training_data,
                ordinal_mappings,
                prediction_times,
                model_name,
                model_file_path,
                filename_results_dict_name
                ):


    best_model_results_dict = {}

    # separate into training, validation and test sets

    train_visits = visits[visits.training_validation_test == 'train'].drop(columns='training_validation_test')
    valid_visits = visits[visits.training_validation_test == 'valid'].drop(columns='training_validation_test')
    test_visits = visits[visits.training_validation_test == 'test'].drop(columns='training_validation_test')


    # Process each time of day
    for _prediction_time in prediction_times:

        print("\nProcessing :" + str(_prediction_time))

        # create a name for the model based on the time of day it is trained for
        MODEL__ED_ADMISSIONS__NAME = get_model_name(model_name, _prediction_time)

        # use this name in the path for saving best model
        full_path = model_file_path / MODEL__ED_ADMISSIONS__NAME 
        full_path = full_path.with_suffix('.joblib')

        # initialise data used for saving attributes of the model
        best_model_results_dict[MODEL__ED_ADMISSIONS__NAME] = {}
        best_valid_logloss = float('inf')
        results_dict = {}
        
        # get visits that were in at the time of day in question and preprocess the training, validation and test sets 
        X_train, y_train = preprocess_data(train_visits, _prediction_time, exclude_from_training_data)
        X_valid, y_valid = preprocess_data(valid_visits, _prediction_time, exclude_from_training_data)
        X_test, y_test = preprocess_data(test_visits, _prediction_time, exclude_from_training_data)
        
        # save size of each set
        best_model_results_dict[MODEL__ED_ADMISSIONS__NAME]['train_valid_test_set_no'] = {
            'train_set_no' : len(X_train),
            'valid_set_no' : len(X_valid),
            'test_set_no' : len(X_test),
        }

        # iterate through the grid of hyperparameters
        for g in ParameterGrid(grid):
            model = initialise_model(g)
            
            # define a column transformer for the ordinal and categorical variables
            column_transformer = create_column_transformer(X_test, ordinal_mappings)
            
            # create a pipeline with the feature transformer and the model
            pipeline = Pipeline([
                ('feature_transformer', column_transformer),
                ('classifier', model)
            ])

            # cross-validate on training set using the function created earlier
            cv_results = chronological_cross_validation(pipeline, X_train, y_train, n_splits=5)

            # Store results for this set of parameters in the results dictionary
            results_dict[str(g)] = {
                'train_auc': cv_results['train_auc'],
                'valid_auc': cv_results['valid_auc'],
                'train_logloss': cv_results['train_logloss'],
                'valid_logloss': cv_results['valid_logloss'],
            }
            
            # Update and save best model if current model is better on validation set
            if cv_results['valid_logloss'] < best_valid_logloss:

                # save the details of the best model
                best_model = str(g)
                best_valid_logloss = cv_results['valid_logloss']

                # save the best model params
                best_model_results_dict[MODEL__ED_ADMISSIONS__NAME]['best_params'] = str(g)

                # save the model metrics on training and validation set
                best_model_results_dict[MODEL__ED_ADMISSIONS__NAME]['train_valid_set_results'] = results_dict

                # score the model's performance on the test set  
                y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
                test_logloss = log_loss(y_test,y_test_pred_proba)
            
                best_model_results_dict[MODEL__ED_ADMISSIONS__NAME]['test_set_results'] = {
                    'test_auc' : test_auc,
                    'test_logloss' : test_logloss
                }

                # save the best features
                # To access transformed feature names:
                transformed_cols = pipeline.named_steps['feature_transformer'].get_feature_names_out()
                transformed_cols = [col.split('__')[-1] for col in transformed_cols]
                best_model_results_dict[MODEL__ED_ADMISSIONS__NAME]['best_model_features'] = {
                        'feature_names': transformed_cols,
                        'feature_importances': pipeline.named_steps['classifier'].feature_importances_.tolist()
                    }

                # save the best model
                dump(pipeline, full_path)

    # save the results dictionary      
    filename_results_dict = filename_results_dict_name
    full_path_results_dict = model_file_path / filename_results_dict

    with open(full_path_results_dict, 'w') as f:
        json.dump(best_model_results_dict, f)  