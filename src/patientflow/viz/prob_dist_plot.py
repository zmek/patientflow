"""
Module: probability_distribution_visualization
==============================================

This module provides functionality to visualize probability distributions using bar plots.
The main function, `prob_dist_plot`, can handle both custom probability data, predefined
distributions such as the Poisson distribution, and dictionary input.

Functions
---------
prob_dist_plot(prob_dist_data, title, directory_path=None, figsize=(6, 3), 
               include_titles=False, truncate_at_beds=20, text_size=None, 
               bar_colour="#5B9BD5", file_name=None, min_beds_lines=None)
    Plots a bar chart of a probability distribution with optional customization for
    titles, labels, and additional markers.

Dependencies
------------
- numpy
- pandas
- matplotlib.pyplot
- scipy.stats
- itertools
"""

import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

def prob_dist_plot(
    prob_dist_data,
    title,
    directory_path=None,
    figsize=(6, 3),
    include_titles=False,
    truncate_at_beds=20,
    text_size=None,
    bar_colour="#5B9BD5",
    file_name=None,
    min_beds_lines=None,
    xlabel="Number of beds"
):
    """
    Plot a probability distribution as a bar chart.

    This function generates a bar plot for a given probability distribution, either
    as a pandas DataFrame, a scipy.stats distribution object (e.g., Poisson), or a
    dictionary. The plot can be customized with titles, axis labels, markers, and
    additional visual properties.

    Parameters
    ----------
    prob_dist_data : pandas.DataFrame, dict, or scipy.stats distribution
        The probability distribution data to be plotted. If a dictionary is provided,
        it is converted into a pandas DataFrame where keys are indices and values are
        probabilities. If a `scipy.stats` distribution (e.g., Poisson) is provided,
        the function computes probabilities for integer values within the specified range.

    title : str
        The title of the plot, used for display and optionally as the file name.

    directory_path : str or pathlib.Path, optional
        Directory where the plot image will be saved. If not provided, the plot is
        displayed without saving.

    figsize : tuple of float, optional, default=(6, 3)
        The size of the figure, specified as (width, height).

    include_titles : bool, optional, default=False
        Whether to include titles and axis labels in the plot.

    truncate_at_beds : int, optional, default=20
        The maximum value of the x-axis to include in the plot.

    text_size : int, optional
        Font size for plot text, including titles and tick labels.

    bar_colour : str, optional, default="#5B9BD5"
        The color of the bars in the plot.

    file_name : str, optional
        Name of the file to save the plot. If not provided, the title is used to generate
        a file name.

    min_beds_lines : dict, optional
        A dictionary where keys are percentages (as decimals) and values are the x-axis
        positions to draw vertical lines. Used to indicate thresholds.
        
    xlabel : str, optional
        A label for the x axis

    Returns
    -------
    None
        The function does not return any value. It displays and optionally saves the plot.

    Examples
    --------
    # Example with a Poisson distribution
    poisson_dist = stats.poisson(mu=5)
    prob_dist_plot(
        prob_dist_data=poisson_dist,
        title="Poisson Distribution Example",
        include_titles=True,
        min_beds_lines={0.5: 5, 0.8: 7}
    )

    # Example with a custom DataFrame
    data = pd.DataFrame({'agg_proba': [0.1, 0.3, 0.4, 0.2]}, index=[0, 1, 2, 3])
    prob_dist_plot(
        prob_dist_data=data,
        title="Custom Distribution Example",
        bar_colour='orange',
        include_titles=True
    )

    # Example with a dictionary
    data_dict = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}
    prob_dist_plot(
        prob_dist_data=data_dict,
        title="Dictionary Example",
        bar_colour='green',
        include_titles=True
    )
    """
    # Handle Poisson distribution input
    if isinstance(prob_dist_data, (stats._distn_infrastructure.rv_frozen, 
                                 stats._discrete_distns.poisson_gen)):
        x = np.arange(0, truncate_at_beds + 1)
        probs = prob_dist_data.pmf(x)
        prob_dist_data = pd.DataFrame({
            'agg_proba': probs
        }, index=x)

    # Handle dictionary input
    elif isinstance(prob_dist_data, dict):
        prob_dist_data = pd.DataFrame(
            {"agg_proba": list(prob_dist_data.values())},
            index=list(prob_dist_data.keys())
        )
    
    plt.figure(figsize=figsize)
    if not file_name:
        file_name = title.replace(" ", "_").replace("/n", "_").replace("%", "percent") + ".png"

    plt.bar(
        prob_dist_data.index[0 : truncate_at_beds + 1],
        prob_dist_data["agg_proba"].values[0 : truncate_at_beds + 1],
        color=bar_colour,
    )
    
    plt.xlim(-0.5, truncate_at_beds + 0.5)
    plt.xticks(np.arange(0, truncate_at_beds + 1, 5))

    if min_beds_lines:
        colors = itertools.cycle(
            plt.cm.gray(np.linspace(0.3, 0.7, len(min_beds_lines)))
        )
        for point in min_beds_lines:
            plt.axvline(
                x=min_beds_lines[point],
                linestyle="--",
                linewidth=2,
                color=next(colors),
                label=f"{point*100:.0f}% probability",
            )
        plt.legend(loc="upper right")

    if text_size:
        plt.tick_params(axis="both", which="major", labelsize=text_size)

    if include_titles:
        plt.title(title, fontsize=text_size)
        plt.xlabel(xlabel)
        plt.ylabel("Probability")

    plt.tight_layout()
    
    if directory_path:
        plt.savefig(directory_path / file_name.replace(" ", "_"), dpi=300)
    
    plt.show()
