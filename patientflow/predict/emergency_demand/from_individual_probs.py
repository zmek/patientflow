"""
Emergency Demand Prediction From Patient-Level Probababilities

This submodule provides functionalities to predict demand as a probability distributions, based on inputs that are patient-level probabilities. The module uses symbolic mathematics to build and manipulate 
expressions dynamically, facilitating the computation of aggregate demand probabilities.

Dependencies:
    - numpy: Used for array and numerical operations.
    - pandas: Utilized for handling data structures like DataFrames, enabling data manipulation and analysis. The module expects a DataFrame `df` with a specific structure, notably including a 'horizon_dt' column, which represents the horizon dates for the demand predictions. This column is crucial for filtering data within the `get_prob_dist` functions.
    - sympy: A Python library for symbolic mathematics, used here to dynamically create and manipulate symbolic expressions, particularly for the calculation of probabilities.


Functions:
    - create_symbols(n): Creates a set of symbolic variables.
    - compute_core_expression(ri, s): Computes a core expression for given inputs.
    - build_expression(syms, n): Builds an overall expression from core expressions.
    - expression_subs(expression, predictions): Substitutes predictions into an expression.
    - return_coeff(expression, i): Returns a specific coefficient from an expanded expression.
    - model_input_to_pred_proba(model_input, model): Converts model input to predicted probabilities.
    - pred_proba_to_pred_demand(predictions_proba): Converts prediction probabilities to predicted demand.
    - get_prob_dist_for_horizon_dt(X_test, y_test, model): Computes probability distribution for a horizon date.
    - get_prob_dist(horizon_dts, df, X_test, y_test, model): Computes probability distributions over horizon dates.
    
These functions can work with any model object as long as it provides the predict_proba method. This icludes libraries (like scikit-learn, TensorFlow, or PyTorch), which generally offer this method

Example Usage:
    # Assuming a predictive model and test data are available
    horizon_dates = ['2023-01-01', '2023-01-02']
    predicted_distribution = get_prob_dist(horizon_dates, dataset, X_test, y_test, model)
    print(predicted_distribution)

Note:
    This module is designed to be generic and can be adapted to various domains where probabilistic demand
    prediction is applicable. Ensure that the input data and the model adhere to the expected formats and
    conventions as required by the functions.

Author: Zella King
Date: 25.03.24
Version: 0.1
"""

import numpy as np
import pandas as pd
import sympy as sym
from sympy import symbols, expand


def create_symbols(n):
    """
    Dynamically create a set of symbols based on the input size.

    :param n: Number of symbols to create.
    :return: A tuple of symbols.
    """
    return symbols(f"r0:{n}")


def compute_core_expression(ri, s):
    """
    Compute the core expression for a given ri and symbol s.

    :param ri: The ri value to substitute in the expression.
    :param s: The symbol s used in the expression.
    :return: The computed core expression.
    """
    r = sym.Symbol("r")
    core_expression = (1 - r) + r * s
    return core_expression.subs({r: ri})


def build_expression(syms, n):
    """
    Build the overall expression by multiplying core expressions for each symbol.

    :param syms: The symbols used in the expressions.
    :param n: The number of terms to include in the product.
    :return: The built expression.
    """
    s = sym.Symbol("s")
    expression = 1
    for i in range(n):
        expression *= compute_core_expression(syms[i], s)
    return expression


def expression_subs(expression, n, predictions):
    """
    Substitute the predictions into the expression.

    :param expression: The expression to substitute into.
    :param predictions: The predictions to use for substitution.
    :return: The substituted expression.
    """
    syms = create_symbols(n)
    substitution = dict(zip(syms[0:n], predictions[0:n]))
    return expression.subs(substitution)


def return_coeff(expression, i):
    """
    Return the coefficient of s^i in the expanded expression.

    :param expression: The expression to expand.
    :param i: The power of s to find the coefficient for.
    :return: The coefficient of s^i.
    """
    s = sym.Symbol("s")
    return expand(expression).coeff(s, i)


def model_input_to_pred_proba(model_input, model):
    """
    Convert model input to predicted probabilities.

    This function takes the model input and uses the provided model to predict probabilities. It then
    organizes these probabilities into a pandas DataFrame.

    :param model_input: The input data to the model, typically features used for prediction.
    :param model: The predictive model that provides a predict_proba method.
    :return: A pandas DataFrame containing the predicted probabilities for the positive class.
    """
    predictions = model.predict_proba(model_input)[:, 1]
    return pd.DataFrame(predictions, columns=["pred_proba"])


def pred_proba_to_pred_demand(predictions_proba, weights=None):
    """
    Convert individual predictions to aggregate demand over a number of beds, optionally weighting the predictions.

    This function takes a DataFrame containing individual probability predictions and aggregates them to
    calculate the predicted demand. The DataFrame should contain a single column named 'pred_proba' where
    each row represents the probability prediction for a single instance.

    :param predictions_proba: A DataFrame containing the probability predictions. It must have a single
                              column named 'pred_proba' with each row representing a probability value
                              (ranging from 0 to 1) of the corresponding instance being positive.
    :param weights: Optional array of weights, of the same length as predictions_proba, to weight the predictions.

    :return: A DataFrame containing the predicted demand. The DataFrame will have a single column
             'agg_proba' where each row corresponds to the aggregated demand probability for that
             number of instances (from 0 to n, where n is the number of predictions).
    """

    n = len(predictions_proba)
    local_proba = predictions_proba.copy()
    if weights is not None:
        local_proba['pred_proba'] *= weights
    
    syms = create_symbols(n)
    expression = build_expression(syms, n)
    expression = expression_subs(expression, n, local_proba['pred_proba'])
    pred_demand_dict = {i: return_coeff(expression, i) for i in range(n + 1)}
    pred_demand = pd.DataFrame.from_dict(
        pred_demand_dict, orient="index", columns=["agg_proba"]
    )
    return pred_demand

def get_prob_dist_for_horizon_dt(X_test, y_test, model, weights=None):
    """
    Get the probability distribution for a specific horizon date.

    This function computes the predicted demand and the actual demand for a given horizon date.
    It utilizes the model to predict probabilities and then converts these predictions to a distribution
    over the number of beds.

    :param X_test: The test set features corresponding to the horizon date.
    :param y_test: The actual outcomes corresponding to the horizon date.
    :param model: The predictive model used for generating probabilities. The model class must provide a predict_proba method

    :return: A dictionary containing the predicted demand ('pred_demand') and the actual demand
             ('actual_demand') for the horizon date.
    """
    horizon_dt_dict = {}

    if len(X_test) > 0:
        pred_proba = model_input_to_pred_proba(X_test, model)
        pred_demand = pred_proba_to_pred_demand(pred_proba, weights)
        horizon_dt_dict["pred_demand"] = pred_demand
        horizon_dt_dict["actual_demand"] = sum(y_test)
    else:
        horizon_dt_dict["pred_demand"] = pd.DataFrame({"agg_proba": [1]}, index=[0])
        horizon_dt_dict["actual_demand"] = 0

    return horizon_dt_dict


def get_prob_dist(episode_slices_dict, X_test, y_test, model, weights=None):
    """
    Calculate probability distributions for each horizon date based on given model predictions.

    Parameters
    ----------
    episode_slices_dict : dict
        A dictionary mapping horizon dates (as datetime objects) to indices in `X_test` and `y_test`
        that correspond to the episode slices to be tested for each date.
    X_test : pandas.DataFrame
        A DataFrame containing the test features for prediction.
    y_test : pandas.Series
        A Series containing the true outcome values corresponding to the test features in `X_test`.
    model : any
        A predictive model object with a `predict_proba` method that takes features from `X_test` and
        optionally weights, and returns a probability distribution over possible outcomes.
    weights : pandas.Series, optional
        A Series containing weights for the test data points, which may influence the prediction,
        by default None. If provided, the weights should be indexed similarly to `X_test` and `y_test`.

    Returns
    -------
    dict
        A dictionary where each key is a horizon date and each value is the resulting probability
        distribution for that date, obtained by applying the model on the corresponding test slices.

    Notes
    -----
    - The function asserts that the length of the test features and outcomes are equal for each
      slice before proceeding with predictions.
    - It notifies the user of progress in processing horizon dates, especially if there are more
      than 10 horizon dates.
    """
    prob_dist_dict = {}
    print(f"Calculating probability distributions for {len(episode_slices_dict)} horizon dates")

    if len(episode_slices_dict) > 10:
        print("This may take a minute or more")

    # Initialize a counter for notifying the user every 10 horizon dates processed
    count = 0

    for dt, episode_slices_to_test in episode_slices_dict.items():
        # Ensure the lengths of test features and outcomes are equal
        assert len(X_test.loc[episode_slices_to_test]) == len(
            y_test.loc[episode_slices_to_test]
        ), "Mismatch in lengths of X_test and y_test slices."

        if weights is None:
            horizon_dt_weights = None
        else:
            horizon_dt_weights = weights.loc[episode_slices_to_test].values

        # Compute the predicted and actual demand for the current horizon date
        prob_dist_dict[dt] = get_prob_dist_for_horizon_dt(
            X_test=X_test.loc[episode_slices_to_test],
            y_test=y_test.loc[episode_slices_to_test],
            model=model,
            weights=horizon_dt_weights
        )

        # Increment the counter and notify the user every 10 horizon dates processed
        count += 1
        if count % 10 == 0 and count != len(episode_slices_dict):
            print(f"Processed {count} horizon dates")

    print(f"Processed {len(episode_slices_dict)} horizon dates")

    return prob_dist_dict
