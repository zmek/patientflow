"""
This module generates Quantile-Quantile (QQ) plots to compare observed values with model predictions in a healthcare context, specifically focusing on the prediction of hospital bed demand. The QQ plot is a graphical technique for determining if two data sets come from populations with a common distribution. If so, we should see the points forming a line approximately along the reference line y=x.

To prepare the predicted distribution
- Treat the predicted distributions (saved as cdfs) for all time points of interest as if they were one distribution
- Within this predicted distribution, because each probability is over a discrete rather than continuous number of input values, the upper and lower of values of the probability range are saved at each value
- The mid point between upper and lower is calculated and saved
- The distribution of cdf mid points (one for each horizon date) is sorted by value of the mid point and a cdf of this is calculated (this is a cdf of cdfs, in effect)
- These are weighted by the probability of each value occurring

To prepare the observed distribution
- Take boserved number each horizon date and save the cdf of that value from its predicted distribution
- The distribution of cdf values (one per horizon date) is sorted
- These are weighted by the probability of each value occurring, which is a uniform probability (1 / over the number of horizon dates)

Key Functions:
- qq_plot: Generates and plots the QQ plot based on the provided observed and predicted data.

The QQ plots generated by this module can help healthcare administrators understand the accuracy of their predictive models in forecasting bed demand

Author: Zella King
Date: 25.03.24
Version: 0.1
"""

# Import necessary libraries for data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def qq_plot(prediction_moments, prob_dist_dict, title_):
    """
    Generate a QQ plot comparing observed values with model predictions.

    The function aggregates the predicted and observed distributions, calculates their CDF (Cumulative Distribution Function),
    and plots the observed CDF against the predicted CDF to visualize the accuracy of the predictions.

    Parameters
    - prediction_moments (list): A list of time points of interest.
    - prob_dist_dict (dict): A nested dictionary containing predicted and actual demands for each time point.
      The structure is {time_point: {'pred_demand': pd.DataFrame of predicted value, 'actual_demand': integer}}.
    - title_ (str): Title for the plot.

    Returns
    - matplotlib.figure.Figure: A figure object containing the QQ plot.

    """
    # Initialize lists to store CDF and observed data
    cdf_data = []
    observed_data = []

    # Loop through each time point to process predicted and observed data
    for dt in prediction_moments:
        # Check if there is data for the current time point
        if dt in prob_dist_dict:
            # Extract predicted demand and actual demand
            pred_demand = np.array(prob_dist_dict[dt]["pred_demand"])
            actual_demand = prob_dist_dict[dt]["actual_demand"]

            # Calculate the CDF for predicted demand
            upper = pred_demand.cumsum()
            lower = np.hstack((0, upper[:-1]))
            mid = (upper + lower) / 2

            # Collect the CDF data and the observed data point
            cdf_data.append(np.column_stack((upper, lower, mid, pred_demand)))
            observed_data.append(
                mid[actual_demand]
            )  # CDF value at the observed admission count

    # Return None if there is no data to plot
    if not cdf_data:
        return None

    # Consolidate CDF data and prepare the dataframe for model predictions
    cdf_data = np.vstack(cdf_data)
    qq_model = pd.DataFrame(
        cdf_data, columns=["cdf_upper", "cdf_mid", "cdf_lower", "weights"]
    )
    qq_model = qq_model.sort_values("cdf_mid")
    qq_model["cum_weight"] = qq_model["weights"].cumsum()
    qq_model["cum_weight_normed"] = (
        qq_model["cum_weight"] / qq_model["weights"].sum()
    )

    # Prepare the observed data for plotting
    qq_observed = pd.DataFrame(observed_data, columns=["cdf_observed"])
    qq_observed = qq_observed.sort_values("cdf_observed")
    qq_observed["weights"] = 1 / len(observed_data)
    qq_observed["cum_weight_normed"] = qq_observed["weights"].cumsum()

    # Calculate the maximum model CDF value corresponding to each observed value
    qq_observed["max_model_cdf_at_this_value"] = qq_observed[
        "cdf_observed"
    ].apply(
        lambda x: qq_model[qq_model["cdf_mid"] <= x]["cum_weight_normed"].max()
    )

    # Plotting the QQ plot
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--")  # Reference line y=x
    ax.plot(
        qq_observed["max_model_cdf_at_this_value"],
        qq_observed["cum_weight_normed"],
        marker=".",
        linewidth=0,
    )
    ax.set_xlabel("Cdf of model distribution")
    ax.set_ylabel("Cdf of observed distribution")
    plt.title(title_)

    return fig  # Return the figure object containing the QQ plot
