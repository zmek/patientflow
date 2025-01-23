import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_distributions(
    df,
    col_name,
    grouping_var,
    grouping_var_name,
    plot_type="both",
    title=None,
    rotate_x_labels=False,
    is_discrete=False,
    ordinal_order=None,
):
    sns.set_theme(style="whitegrid")

    if ordinal_order is not None:
        df[col_name] = pd.Categorical(
            df[col_name], categories=ordinal_order, ordered=True
        )

    g = sns.FacetGrid(df, col=grouping_var, height=3, aspect=1.5)

    if is_discrete:
        valid_values = sorted([x for x in df[col_name].unique() if pd.notna(x)])
        min_val = min(valid_values)
        max_val = max(valid_values)
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    else:
        # Handle numeric data
        values = df[col_name].dropna()
        if pd.api.types.is_numeric_dtype(values):
            if np.allclose(values, values.round()):
                bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
            else:
                n_bins = min(100, max(10, int(np.sqrt(len(values)))))
                bins = n_bins
        else:
            bins = "auto"

    if plot_type == "both":
        g.map(sns.histplot, col_name, kde=True, bins=bins)
    elif plot_type == "hist":
        g.map(sns.histplot, col_name, kde=False, bins=bins)
    elif plot_type == "kde":
        g.map(sns.kdeplot, col_name, fill=True)
    else:
        raise ValueError("Invalid plot_type. Choose from 'both', 'hist', or 'kde'.")

    g.set_axis_labels(
        col_name, "Frequency" if plot_type != "kde" else "Density", fontsize=10
    )
    
    # Set facet titles with smaller font
    g.set_titles(col_template="{col_name}", size=11)

    if rotate_x_labels:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)

    if is_discrete:
        for ax in g.axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlim(min_val - 0.5, max_val + 0.5)

    plt.subplots_adjust(top=0.85)
    if title:
        g.figure.suptitle(title, fontsize=14)
    else:
        g.figure.suptitle(f"Distribution of {col_name} grouped by {grouping_var_name}", fontsize=14)

    plt.show()