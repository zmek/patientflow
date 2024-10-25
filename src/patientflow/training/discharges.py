import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from patientflow.load import get_model_name 
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.train import initialise_xgb, calculate_class_balance #create_column_transformer, chronological_cross_validation
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_auc_score,
    average_precision_score
)
from joblib import dump, load
import json




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
    train_auprcs = []

    valid_aucs = []
    valid_loglosses = []
    valid_auprcs = []

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
        train_auprc = average_precision_score(y_train, y_train_pred)
        train_aucs.append(train_auc)
        train_loglosses.append(train_logloss)
        train_auprcs.append(train_auprc)
        
        # Evaluate on the validation split
        y_valid_pred = pipeline.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, y_valid_pred)
        valid_logloss = log_loss(y_valid, y_valid_pred)
        valid_auprc = average_precision_score(y_valid, y_valid_pred)
        valid_aucs.append(valid_auc)
        valid_loglosses.append(valid_logloss)
        valid_auprcs.append(valid_auprc)

    # Calculate mean scores
    mean_train_auc = sum(train_aucs) / n_splits
    mean_train_logloss = sum(train_loglosses) / n_splits
    mean_train_auprc = sum(train_auprcs) / n_splits
    mean_valid_auc = sum(valid_aucs) / n_splits
    mean_valid_logloss = sum(valid_loglosses) / n_splits
    mean_valid_auprc = sum(valid_auprcs) / n_splits
    
    return {
        "train_auc": mean_train_auc,
        "valid_auc": mean_valid_auc,
        "train_logloss": mean_train_logloss,
        "valid_logloss": mean_valid_logloss,
        "train_auprc": mean_train_auprc,
        "valid_auprc": mean_valid_auprc
    }




def create_column_transformer(df, ordinal_mappings=None, top_n_subspecialties=20):
    """
    Create a column transformer for a dataframe with dynamic column handling.
    :param df: Input dataframe.
    :param ordinal_mappings: A dictionary specifying the ordinal mappings for specific columns.
    :param top_n_subspecialties: Number of top subspecialties to one-hot encode.
    :return: A configured ColumnTransformer object.
    """
    transformers = []
    # Default to an empty dict if no ordinal mappings are provided
    if ordinal_mappings is None:
        ordinal_mappings = {}

    for col in df.columns:
        if col == 'Subspecialty':
            # Get the top N subspecialties
            top_subspecialties = df[col].value_counts().nlargest(top_n_subspecialties).index.tolist()
            
            # Create a custom OneHotEncoder for Subspecialty
            subspecialty_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            subspecialty_encoder.fit(np.array(top_subspecialties).reshape(-1, 1))
            
            transformers.append((col, subspecialty_encoder, [col]))
        elif col in ordinal_mappings:
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




def train_discharges_models(
    visits,
    grid,
    exclude_from_training_data,
    ordinal_mappings,
    prediction_times,
    model_name,
    model_file_path,
    model_metadata,
    filename_results_dict_name,
    label_col
):
    
    _exclude_from_training_data = exclude_from_training_data.copy()
    _exclude_from_training_data.remove(label_col)
    
    train_visits = visits[visits.training_validation_test == 'train'].drop(columns='training_validation_test')
    valid_visits = visits[visits.training_validation_test == 'valid'].drop(columns='training_validation_test')
    test_visits = visits[visits.training_validation_test == 'test'].drop(columns='training_validation_test')

    # Process each time of day
    for _prediction_time in prediction_times:

        print("\nProcessing :" + str(_prediction_time))

        # create a name for the model based on the time of day it is trained for
        _model_name = get_model_name(model_name, _prediction_time)

        # use this name in the path for saving best model
        full_path = model_file_path / _model_name 
        full_path = full_path.with_suffix('.joblib')

        # initialise data used for saving attributes of the model
        model_metadata[_model_name] = {}
        best_valid_logloss = float('inf')
        results_dict = {}

        # get visits that were in at the time of day in question and preprocess the training, validation and test sets 
        X_train, y_train = get_snapshots_at_prediction_time(train_visits, _prediction_time, _exclude_from_training_data, 
                                          single_snapshot_per_visit=True, label_col = label_col)
        X_valid, y_valid = get_snapshots_at_prediction_time(valid_visits, _prediction_time, _exclude_from_training_data, 
                                          single_snapshot_per_visit=True, label_col = label_col)
        X_test, y_test = get_snapshots_at_prediction_time(test_visits, _prediction_time, _exclude_from_training_data, 
                                          single_snapshot_per_visit=True, label_col = label_col)

        y_train_class_balance = calculate_class_balance(y_train)
        y_valid_class_balance = calculate_class_balance(y_valid)
        y_test_class_balance = calculate_class_balance(y_test)

        # save size of each set
        model_metadata[_model_name]["train_valid_test_set_no"] = {
            "train_set_no": len(X_train),
            "valid_set_no": len(X_valid),
            "test_set_no": len(X_test),
        }

        # save class balance of each set
        model_metadata[_model_name]["train_valid_test_class_balance"] = {
            "y_train_class_balance": y_train_class_balance,
            "y_valid_class_balance": y_valid_class_balance,
            "y_test_class_balance": y_test_class_balance,
        }


        # iterate through the grid of hyperparameters
        for g in ParameterGrid(grid):
            model = initialise_xgb(g)

            # define a column transformer for the ordinal and categorical variables
            column_transformer = create_column_transformer(X_train, ordinal_mappings)

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
                'train_auprc': cv_results['train_auprc'],  # New metric
                'valid_auprc': cv_results['valid_auprc']   # New metric
            }

            # Update and save best model if current model is better on validation set
            if cv_results['valid_logloss'] < best_valid_logloss:

                # save the details of the best model
                best_model = str(g)
                best_valid_logloss = cv_results['valid_logloss']

                # save the best model params
                model_metadata[_model_name]['best_params'] = str(g)

                # save the model metrics on training and validation set
                model_metadata[_model_name]['train_valid_set_results'] = results_dict

                # score the model's performance on the test set  
                y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
                test_logloss = log_loss(y_test,y_test_pred_proba)
                test_auprc = average_precision_score(y_test, y_test_pred_proba)
                model_metadata[_model_name]['test_set_results'] = {
                    'test_auc' : test_auc,
                    'test_logloss' : test_logloss,
                    'test_auprc' : test_auprc
                }

                # save the best features
                # To access transformed feature names:
                transformed_cols = pipeline.named_steps['feature_transformer'].get_feature_names_out()
                transformed_cols = [col.split('__')[-1] for col in transformed_cols]
                model_metadata[_model_name]['best_model_features'] = {
                        'feature_names': transformed_cols,
                        'feature_importances': pipeline.named_steps['classifier'].feature_importances_.tolist()
                    }

                # save the best model
                dump(pipeline, full_path)

    # save the results dictionary
    filename_results_dict_path = model_file_path 
    full_path_results_dict = filename_results_dict_path / filename_results_dict_name

    with open(full_path_results_dict, "w") as f:
        json.dump(model_metadata, f) 
        
    return model_metadata