import numpy as np
import pandas as pd
from scipy.stats import binom, poisson


def weighted_poisson_binomial(i, lam, theta):
    """
    Calculate weighted probabilities using Poisson and Binomial distributions.

    Parameters
    i (int): The upper bound of the range for the binomial distribution.
    lam (float): The lambda parameter for the Poisson distribution.
    theta (float): The probability of success for the binomial distribution.

    Returns
    numpy.ndarray: An array of weighted probabilities.

    """
    if i < 0 or lam < 0 or not 0 <= theta <= 1:
        raise ValueError("Ensure i >= 0, lam >= 0, and 0 <= theta <= 1.")

    arr_seq = np.arange(i + 1)
    probabilities = binom.pmf(arr_seq, i, theta)
    return poisson.pmf(i, lam) * probabilities


def aggregate_probabilities(lam, kmax, theta, time_index):
    """
    Aggregate probabilities for a range of values using the weighted Poisson-Binomial distribution.

    Parameters
    lam (numpy.ndarray): An array of lambda values for each time interval.
    kmax (int): The maximum number of events to consider.
    theta (numpy.ndarray): An array of theta values for each time interval.
    time_index (int): The current time index for which to calculate probabilities.

    Returns
    numpy.ndarray: Aggregated probabilities for the given time index.

    """
    if kmax < 0 or time_index < 0 or len(lam) <= time_index or len(theta) <= time_index:
        raise ValueError("Invalid kmax, time_index, or array lengths.")

    probabilities_matrix = np.zeros((kmax + 1, kmax + 1))
    for i in range(kmax + 1):
        probabilities_matrix[: i + 1, i] = weighted_poisson_binomial(
            i, lam[time_index], theta[time_index]
        )
    return probabilities_matrix.sum(axis=1)


def convolute_distributions(dist_a, dist_b):
    """
    Convolutes two probability distributions represented as dataframes.

    Parameters
    dist_a (pd.DataFrame): The first distribution with columns ['sum', 'prob'].
    dist_b (pd.DataFrame): The second distribution with columns ['sum', 'prob'].

    Returns
    pd.DataFrame: The convoluted distribution.

    """
    if not {"sum", "prob"}.issubset(dist_a.columns) or not {"sum", "prob"}.issubset(
        dist_b.columns
    ):
        raise ValueError("DataFrames must contain 'sum' and 'prob' columns.")

    sums = [x + y for x in dist_a["sum"] for y in dist_b["sum"]]
    probs = [x * y for x in dist_a["prob"] for y in dist_b["prob"]]
    result = pd.DataFrame(zip(sums, probs), columns=["sum", "prob"])
    return result.groupby("sum")["prob"].sum().reset_index()


def poisson_binom_generating_function(NTimes, lambda_t, theta, epsilon):
    """
    Generate a distribution based on the aggregate of Poisson and Binomial distributions over time intervals.

    Parameters
    NTimes (int): The number of time intervals.
    lambda_t (numpy.ndarray): An array of lambda values for each time interval.
    theta (numpy.ndarray): An array of theta values for each time interval.
    epsilon (float): The desired error threshold.

    Returns
    pd.DataFrame: The generated distribution.

    """
    if NTimes <= 0 or epsilon <= 0 or epsilon >= 1:
        raise ValueError("Ensure NTimes > 0 and 0 < epsilon < 1.")

    maxlam = max(lambda_t)
    kmax = int(poisson.ppf(1 - epsilon, maxlam))
    distribution = np.zeros((kmax + 1, NTimes))

    for j in range(NTimes):
        distribution[:, j] = aggregate_probabilities(lambda_t, kmax, theta, j)

    df_list = [
        pd.DataFrame({"sum": range(kmax + 1), "prob": distribution[:, j]})
        for j in range(NTimes)
    ]
    total_distribution = df_list[0]

    for df in df_list[1:]:
        total_distribution = convolute_distributions(total_distribution, df)

    total_distribution = total_distribution.rename(
        columns={"prob": "agg_proba"}
    ).set_index("sum")

    return total_distribution
