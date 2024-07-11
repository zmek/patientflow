import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a consistent color palette
color_palette = {0: "blue", 1: "orange"}  # Assuming is_admitted can only be 0 or 1


def calculate_bins(data):
    q25, q75 = np.percentile(data.dropna(), [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return 10  # Fallback to a default number of bins if IQR is zero
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    if bin_width == 0:
        return 10  # Fallback to a default number of bins if bin width is zero
    bins = (data.max() - data.min()) / bin_width
    if not np.isfinite(bins):
        return (
            10  # Fallback to a default number of bins if bins calculation is infinite
        )
    return int(np.ceil(bins))


def plot_binned_histograms(df, col_name, group_by_col="is_admitted"):
    df_copy = df.copy()
    if np.issubdtype(df_copy[col_name].dtype, np.timedelta64):
        df_copy[col_name] = df_copy[col_name].dt.total_seconds()
    num_bins = calculate_bins(df_copy[col_name])
    bins = np.linspace(df_copy[col_name].min(), df_copy[col_name].max(), num_bins + 1)
    unique_labels = df_copy[group_by_col].unique()
    fig, axes = plt.subplots(
        nrows=len(unique_labels), ncols=1, figsize=(8, 3 * len(unique_labels))
    )  # Adjusted figsize
    for idx, label in enumerate(unique_labels):
        ax = axes[idx] if len(unique_labels) > 1 else axes
        group = df_copy[df_copy[group_by_col] == label]
        if group[col_name].dropna().empty:
            print(
                f"Skipping group {label}: column {col_name} contains only NaN values."
            )
            continue
        ax.hist(
            group[col_name].dropna(),
            bins=bins,
            edgecolor="black",
            color=color_palette[label],
        )
        ax.set_title(f"{label} {group_by_col}")
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_categorical_histograms(df, col_name, group_by_col="is_admitted"):
    df_copy = df.copy()
    unique_labels = df_copy[group_by_col].unique()
    fig, axes = plt.subplots(
        nrows=len(unique_labels), ncols=1, figsize=(8, 3 * len(unique_labels))
    )  # Adjusted figsize
    for idx, label in enumerate(unique_labels):
        ax = axes[idx] if len(unique_labels) > 1 else axes
        group = df_copy[df_copy[group_by_col] == label]
        value_counts = group[col_name].value_counts().sort_index()
        if value_counts.empty:
            print(
                f"Skipping group {label}: column {col_name} contains only NaN values or no values."
            )
            continue
        value_counts.plot(kind="bar", ax=ax, color=color_palette[label])
        ax.set_title(f"{label} {group_by_col}")
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=0
        )  # Ensure x-tick labels are not rotated
    plt.tight_layout()
    plt.show()


def plot_text_histograms(
    df, col_name, group_by_col="is_admitted", max_unique_values=20
):
    df_copy = df.copy()
    unique_values = df_copy[col_name].nunique()
    if unique_values > max_unique_values:
        print(f"Skipping column {col_name}: too many unique values ({unique_values}).")
        return

    df_copy[col_name] = df_copy[col_name].astype("category")
    unique_labels = df_copy[group_by_col].unique()

    fig, ax = plt.subplots(figsize=(8, 3))
    category_value_counts = {}
    for label in unique_labels:
        group = df_copy[df_copy[group_by_col] == label]
        value_counts = group[col_name].value_counts().sort_index()
        if value_counts.empty:
            print(
                f"Skipping group {label}: column {col_name} contains only NaN values or no values."
            )
            continue
        category_value_counts[label] = value_counts

    if not category_value_counts:
        print(f"Skipping column {col_name}: no valid data to plot.")
        return

    value_counts_df = pd.DataFrame(category_value_counts).fillna(0)
    value_counts_df.plot(
        kind="bar",
        ax=ax,
        color=[color_palette[label] for label in value_counts_df.columns],
    )

    ax.set_title(f"Distribution of {col_name} grouped by {group_by_col}")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Count")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=0
    )  # Ensure x-tick labels are not rotated
    plt.tight_layout()
    plt.show()


def main_plot_function(df, exclude_from_plot, group_by_col="is_admitted"):
    for column in df.columns:
        if column not in exclude_from_plot:
            if pd.api.types.is_bool_dtype(df[column]):
                plot_categorical_histograms(df, column, group_by_col)
            elif pd.api.types.is_numeric_dtype(
                df[column]
            ) or pd.api.types.is_timedelta64_dtype(df[column]):
                plot_binned_histograms(df, column, group_by_col)
            elif isinstance(df[column].dtype, pd.CategoricalDtype):
                plot_categorical_histograms(df, column, group_by_col)
            elif pd.api.types.is_object_dtype(df[column]):
                plot_text_histograms(df, column, group_by_col)
            else:
                print(f"Skipping column {column}: unsupported data type.")
