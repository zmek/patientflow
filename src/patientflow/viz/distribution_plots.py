import matplotlib.pyplot as plt

# Define a consistent color palette
# color_palette = {0: 'blue', 1: 'orange'}  # Assuming is_admitted can only be 0 or 1
import seaborn as sns


def plot_distributions(
    df,
    col_name,
    grouping_var,
    plot_type="both",
    title=None,
    rotate_x_labels=False,
    is_discrete=False,
):
    """
    Creates side-by-side plots comparing the distributions of a variable
    for each value of a grouping variable. Option to plot kde, which is useful for visualizing the distribution of data points in a smooth curve.

    Parameters
    df (pandas.DataFrame): The dataframe containing the data.
    col_name (str): The name of the variable column to plot.
    grouping_var (str): The name of the grouping variable column.
    plot_type (str): The type of plot to display ('both', 'hist', 'kde').
                     'both' displays both histogram and KDE,
                     'hist' displays only the histogram,
                     'kde' displays only the KDE plot.
    title (str): The overall title for the plot.
    rotate_x_labels (bool): Whether to rotate x-axis labels.
    is_discrete (bool): Whether the variable is discrete. If True, sets number of bins to max value.

    """
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Create a FacetGrid for side-by-side plots
    g = sns.FacetGrid(df, col=grouping_var, height=3, aspect=1.5)  

    # Determine the number of bins if discrete
    if is_discrete:
        bins = int(df[col_name].max()) + 1
    else:
        bins = "auto"

    # Map the appropriate plot type to each facet
    if plot_type == "both":
        g.map(sns.histplot, col_name, kde=True, bins=bins)
    elif plot_type == "hist":
        g.map(sns.histplot, col_name, kde=False, bins=bins)
    elif plot_type == "kde":
        g.map(sns.kdeplot, col_name, fill=True)
    else:
        raise ValueError("Invalid plot_type. Choose from 'both', 'hist', or 'kde'.")

    g.set_axis_labels(
        col_name, "Frequency" if plot_type != "kde" else "Density", fontsize=9
    )

    if rotate_x_labels:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)

    # Force integer x-axis if discrete
    if is_discrete:
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlim(df[col_name].min() - 0.5, df[col_name].max() + 0.5)

    # Set the overall title
    plt.subplots_adjust(top=0.85)
    if title:
        g.figure.suptitle(title, fontsize=12)
    else:
        g.figure.suptitle(f"Distribution of {col_name} by {grouping_var}", fontsize=12)

    # Show the plot
    plt.show()
