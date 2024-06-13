"""
Emergency Demand Prediction From Patient-Level Probababilities

This submodule provides functions to predict demand as a probability distribution, based on inputs that are patient-level probabilities. The module uses symbolic mathematics to build and manipulate expressions dynamically, facilitating the computation of aggregate demand probabilities.

Dependencies:
    - numpy: Used for array and numerical operations.
    - pandas: Utilized for handling data structures like DataFrames, enabling data manipulation and analysis.
    - sympy: A Python library for symbolic mathematics, used here to dynamically create and manipulate symbolic expressions, particularly for the calculation of probabilities.


Functions:
- create_symbols(n): Generates symbolic variables.
- compute_core_expression(ri, s): Computes a symbolic expression involving both symbols and constants.
- build_expression(syms, n): Constructs a cumulative product of symbolic expressions.
- expression_subs(expression, n, predictions): Substitutes numerical values into a symbolic expression.
- return_coeff(expression, i): Extracts coefficients from expanded symbolic expressions.
- model_input_to_pred_proba(model_input, model): Converts model input data into predicted probabilities.
- pred_proba_to_pred_demand(predictions_proba, weights): Aggregates probability predictions into demand predictions.
- get_prob_dist_for_prediction_moment(X_test, y_test, model, weights): Calculates predicted and actual demands for a specific date.
- get_prob_dist(snapshots_dict, X_test, y_test, model, weights): Computes probability distributions for multiple horizon dates.

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

import pandas as pd
import sympy as sym
from sympy import expand, symbols


def create_symbols(n):
    """
    Generate a sequence of symbolic objects intended for use in mathematical expressions.

    Parameters
    ----------
    n : int
        Number of symbols to create.

    Returns
    -------
    tuple
        A tuple containing the generated symbolic objects.

    """
    return symbols(f"r0:{n}")


def compute_core_expression(ri, s):
    """
    Compute a symbolic expression involving a basic mathematical operation with a symbol and a constant.

    Parameters
    ----------
    ri : float
        The constant value to substitute into the expression.
    s : Symbol
        The symbolic object used in the expression.

    Returns
    -------
    Expr
        The symbolic expression after substitution.

    """
    r = sym.Symbol("r")
    core_expression = (1 - r) + r * s
    return core_expression.subs({r: ri})


def build_expression(syms, n):
    """
    Construct a cumulative product expression by combining individual symbolic expressions.

    Parameters
    ----------
    syms : iterable
        Iterable containing symbols to use in the expressions.
    n : int
        The number of terms to include in the cumulative product.

    Returns
    -------
    Expr
        The cumulative product of the expressions.

    """
    s = sym.Symbol("s")
    expression = 1
    for i in range(n):
        expression *= compute_core_expression(syms[i], s)
    return expression


def expression_subs(expression, n, predictions):
    """
    Substitute values into a symbolic expression based on a mapping from symbols to predictions.

    Parameters
    ----------
    expression : Expr
        The symbolic expression to perform substitution on.
    n : int
        Number of symbols and corresponding predictions.
    predictions : list
        List of numerical predictions to substitute.

    Returns
    -------
    Expr
        The expression after performing the substitution.

    """
    syms = create_symbols(n)
    substitution = dict(zip(syms, predictions))
    return expression.subs(substitution)


def return_coeff(expression, i):
    """
    Extract the coefficient of a specified power from an expanded symbolic expression.

    Parameters
    ----------
    expression : Expr
        The expression to expand and extract from.
    i : int
        The power of the term whose coefficient is to be extracted.

    Returns
    -------
    number
        The coefficient of the specified power in the expression.

    """
    s = sym.Symbol("s")
    return expand(expression).coeff(s, i)


def model_input_to_pred_proba(model_input, model):
    """
    Use a predictive model to convert model input data into predicted probabilities.

    Parameters
    ----------
    model_input : array-like
        The input data to the model, typically as features used for predictions.
    model : object
        A model object with a `predict_proba` method that computes probability estimates.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the predicted probabilities for the positive class,
        with one column labeled 'pred_proba'.

    """
    predictions = model.predict_proba(model_input)[:, 1]
    return pd.DataFrame(predictions, columns=["pred_proba"])


def pred_proba_to_pred_demand(predictions_proba, weights=None):
    """
    Aggregate individual probability predictions into predicted demand using optional weights.

    Parameters
    ----------
    predictions_proba : DataFrame
        A DataFrame containing the probability predictions; must have a single column named 'pred_proba'.
    weights : array-like, optional
        An array of weights, of the same length as the DataFrame rows, to apply to each prediction.

    Returns
    -------
    DataFrame
        A DataFrame with a single column 'agg_proba' showing the aggregated probability demand,
        indexed from 0 to n, where n is the number of predictions.

    """
    n = len(predictions_proba)
    local_proba = predictions_proba.copy()
    if weights is not None:
        local_proba["pred_proba"] *= weights

    syms = create_symbols(n)
    expression = build_expression(syms, n)
    expression = expression_subs(expression, n, local_proba["pred_proba"])
    pred_demand_dict = {i: return_coeff(expression, i) for i in range(n + 1)}
    pred_demand = pd.DataFrame.from_dict(
        pred_demand_dict, orient="index", columns=["agg_proba"]
    )
    return pred_demand


def get_prob_dist_for_prediction_moment(X_test, y_test, model, weights=None):
    """
    Calculate both predicted and actual demand distributions for a given date using test data.

    Parameters
    ----------
    X_test : array-like
        Test features for a specific horizon date.
    y_test : array-like
        Actual outcomes corresponding to the test features.
    model : object
        A predictive model which should provide a `predict_proba` method.
    weights : array-like, optional
        Weights to apply to the predictions for demand calculation.

    Returns
    -------
    dict
        A dictionary with keys 'pred_demand' and 'actual_demand' containing the predicted and actual demands
        respectively for the horizon date. Each is presented as a DataFrame or an integer.

    """
    prediction_moment_dict = {}

    if len(X_test) > 0:
        pred_proba = model_input_to_pred_proba(X_test, model)
        pred_demand = pred_proba_to_pred_demand(pred_proba, weights)
        prediction_moment_dict["pred_demand"] = pred_demand
        prediction_moment_dict["actual_demand"] = sum(y_test)
    else:
        prediction_moment_dict["pred_demand"] = pd.DataFrame(
            {"agg_proba": [1]}, index=[0]
        )
        prediction_moment_dict["actual_demand"] = 0

    return prediction_moment_dict


def get_prob_dist(snapshots_dict, X_test, y_test, model, weights=None):
    """
    Calculate probability distributions for each horizon date based on given model predictions.

    Parameters
    ----------
    snapshots_dict : dict
        A dictionary mapping horizon dates (as datetime objects) to indices in `X_test` and `y_test`
        that correspond to the snapshots to be tested for each date.
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
        distribution for that date, obtained by applying the model on the corresponding test snapshots.

    Notes
    -----
    - The function asserts that the length of the test features and outcomes are equal for each
      snapshot before proceeding with predictions.
    - It notifies the user of progress in processing horizon dates, especially if there are more
      than 10 horizon dates.

    """
    prob_dist_dict = {}
    print(
        f"Calculating probability distributions for {len(snapshots_dict)} horizon dates"
    )

    if len(snapshots_dict) > 10:
        print("This may take a minute or more")

    # Initialize a counter for notifying the user every 10 horizon dates processed
    count = 0

    for dt, snaptshots_to_include in snapshots_dict.items():
        # Ensure the lengths of test features and outcomes are equal
        assert len(X_test.loc[snaptshots_to_include]) == len(
            y_test.loc[snaptshots_to_include]
        ), "Mismatch in lengths of X_test and y_test snapshots."

        if weights is None:
            prediction_moment_weights = None
        else:
            prediction_moment_weights = weights.loc[snaptshots_to_include].values

        # Compute the predicted and actual demand for the current horizon date
        prob_dist_dict[dt] = get_prob_dist_for_prediction_moment(
            X_test=X_test.loc[snaptshots_to_include],
            y_test=y_test.loc[snaptshots_to_include],
            model=model,
            weights=prediction_moment_weights,
        )

        # Increment the counter and notify the user every 10 horizon dates processed
        count += 1
        if count % 10 == 0 and count != len(snapshots_dict):
            print(f"Processed {count} horizon dates")

    print(f"Processed {len(snapshots_dict)} horizon dates")

    return prob_dist_dict
