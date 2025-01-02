"""
This module contains functions to visualize inpatient arrival rates and their cumulative
statistics. The visualizations support time-varying analysis, comparison of datasets,
and statistical distributions for resource planning in healthcare facilities.

Functions:
    - annotate_hour_line: Annotates hour lines on a matplotlib plot.
    - plot_arrival_rates: Plots arrival rates for one or two datasets with optional
                          lagged or spread rates.
    - plot_cumulative_arrival_rates: Plots cumulative arrival rates with options for
                                     statistical distribution visualization.

Dependencies:
    - matplotlib.pyplot
    - numpy
    - scipy.stats
    - patientflow.calculate (for rate calculations)
    - patientflow.viz.utils (for utility functions)
"""

import matplotlib.pyplot as plt
import numpy as np

from patientflow.calculate import (
    time_varying_arrival_rates,
    time_varying_arrival_rates_lagged,
    process_arrival_rates,
    true_demand_by_hour,
)

from patientflow.viz.utils import clean_title_for_filename
import scipy.stats as stats


# def get_arrival_rates_spread(inpatient_arrivals, curve_params, time_interval=60):


def annotate_hour_line(
    hour_line,
    y_value,
    hour_values,
    start_plot_index,
    line_styles,
    x_margin,
    annotation_prefix,
    text_y_offset=1,  # New parameter with default of 1
    text_x_position=None,  # New parameter to control horizontal text position
    slope=None,
    x1=None,
    y1=None,
):
    """
    Annotate hour lines on a matplotlib plot with consistent formatting.

    Parameters
    ----------
    hour_line : int
        The hour to annotate on the plot.
    y_value : float
        The y-coordinate for annotation positioning.
    hour_values : list of int
        Hour values corresponding to the x-axis positions.
    start_plot_index : int
        Starting index for the plot's data.
    line_styles : dict
        Line styles for annotations keyed by hour.
    x_margin : float
        Margin added to x-axis for annotation positioning.
    annotation_prefix : str
        Prefix for the annotation text (e.g., "On average").
    text_y_offset : float, optional
        Vertical offset for the annotation text from the line (default is 1).
    text_x_position : float, optional
        Horizontal position for annotation text (default is calculated).
    slope : float, optional
        Slope of a line for extended annotations (used with x1 and y1).
    x1 : float, optional
        Reference x-coordinate for slope-based annotation.
    y1 : float, optional
        Reference y-coordinate for slope-based annotation.

    Returns
    -------
    None
        Annotates the matplotlib plot in place.
    """
    a = hour_values[hour_line - start_plot_index]
    if slope is not None and x1 is not None:
        y_a = slope * (a - x1) + y1
        plt.plot([a, a], [0, y_a], color="grey", linestyle=line_styles[hour_line])
        plt.plot(
            [0 - x_margin, a],
            [y_a, y_a],
            color="grey",
            linestyle=line_styles[hour_line],
        )
        annotation_text = (
            f"{annotation_prefix}, {int(y_a)} beds needed by {hour_line}:00"
        )
        y_position = y_a + text_y_offset
    else:
        plt.annotate(
            "",
            xy=(a, y_value),
            xytext=(a, 0),
            arrowprops=dict(
                arrowstyle="-", linestyle=line_styles[hour_line], color="grey"
            ),
        )
        plt.annotate(
            "",
            xy=(a, y_value),
            xytext=(hour_values[0] - x_margin, y_value),
            arrowprops=dict(
                arrowstyle="-", linestyle=line_styles[hour_line], color="grey"
            ),
        )
        annotation_text = (
            f"{annotation_prefix}, {int(y_value)} beds needed by {hour_line}:00"
        ).strip()  # strip() removes leading comma if prefix is empty
        y_position = y_value + text_y_offset

    # Use custom text x position if provided, otherwise use default
    x_position = (
        text_x_position if text_x_position is not None else (hour_values[1] - x_margin)
    )

    plt.annotate(
        annotation_text,
        xy=(a / 2 if slope is not None else a, y_value),
        xytext=(x_position, y_position),
        va="bottom",
        ha="left",
        fontsize=10,
    )


def plot_arrival_rates(
    inpatient_arrivals,
    title,
    inpatient_arrivals_2=None,  # Optional second dataset
    labels=None,  # Optional tuple of labels for dual plots
    lagged_by=None,
    curve_params=None,
    time_interval=60,
    start_plot_index=0,
    x_margin=0.5,
    file_prefix="",
    media_file_path=None,
):
    """
    Plot arrival rates for one or two datasets with optional lagged and spread rates.

    Parameters
    ----------
    inpatient_arrivals : array-like
        Primary dataset of inpatient arrivals.
    title : str
        Title of the plot.
    inpatient_arrivals_2 : array-like, optional
        Optional second dataset for comparison (default is None).
    labels : tuple of str, optional
        Labels for the datasets when comparing two datasets (default is None).
    lagged_by : int, optional
        Time lag in hours to apply to the arrival rates (default is None).
    curve_params : tuple of float, optional
        Parameters for spread arrival rates as (x1, y1, x2, y2) (default is None).
    time_interval : int, optional
        Time interval in minutes for arrival rate calculations (default is 60).
    start_plot_index : int, optional
        Starting hour index for plotting (default is 0).
    x_margin : float, optional
        Margin on the x-axis (default is 0.5).
    file_prefix : str, optional
        Prefix for the saved file name (default is "").
    media_file_path : str or Path, optional
        Directory path to save the plot (default is None).

    Returns
    -------
    None
        Displays or saves the matplotlib plot.
    """
    is_dual_plot = inpatient_arrivals_2 is not None
    if is_dual_plot and labels is None:
        labels = ("Dataset 1", "Dataset 2")

    datasets = [(inpatient_arrivals, "C0", "o")]
    if is_dual_plot:
        datasets.append((inpatient_arrivals_2, "C1", "s"))

    # Calculate and process arrival rates for all datasets
    processed_data = []
    max_y_values = []

    for dataset, color, marker in datasets:
        # Calculate base arrival rates
        arrival_rates_dict = time_varying_arrival_rates(dataset, time_interval)
        arrival_rates, hour_labels, hour_values = process_arrival_rates(
            arrival_rates_dict
        )
        max_y_values.append(max(arrival_rates))

        # Calculate lagged rates if needed
        arrival_rates_lagged = None
        if lagged_by is not None:
            arrival_rates_lagged_dict = time_varying_arrival_rates_lagged(
                dataset, lagged_by, time_interval
            )
            arrival_rates_lagged, _, _ = process_arrival_rates(
                arrival_rates_lagged_dict
            )
            max_y_values.append(max(arrival_rates_lagged))

        # Calculate spread rates if needed
        arrival_rates_spread = None
        if curve_params is not None:
            x1, y1, x2, y2 = curve_params
            arrival_rates_spread_dict = true_demand_by_hour(dataset, x1, y1, x2, y2)
            arrival_rates_spread, _, _ = process_arrival_rates(
                arrival_rates_spread_dict
            )
            max_y_values.append(max(arrival_rates_spread))

        processed_data.append(
            {
                "arrival_rates": arrival_rates,
                "arrival_rates_lagged": arrival_rates_lagged,
                "arrival_rates_spread": arrival_rates_spread,
                "color": color,
                "marker": marker,
                "label": labels[len(processed_data)] if is_dual_plot else None,
            }
        )

    # Helper function to create cyclic data
    def get_cyclic_data(data):
        return data[start_plot_index:] + data[0:start_plot_index]

    # Plot setup
    plt.figure(figsize=(10, 6))
    x_values = get_cyclic_data(hour_labels)

    # Plot data for each dataset
    for data in processed_data:
        label_suffix = f" ({data['label']})" if data["label"] else ""

        if curve_params is not None and data["arrival_rates_spread"] is not None:
            # Base arrival rates
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates"]),
                marker="x",
                color=data["color"],
                markersize=4,
                linestyle=":",
                linewidth=1,
                label=f"Arrival rates of admitted patients{label_suffix}",
            )

            if lagged_by is not None:
                # Lagged arrival rates
                plt.plot(
                    x_values,
                    get_cyclic_data(data["arrival_rates_lagged"]),
                    marker="o",
                    markersize=4,
                    color=data["color"],
                    linestyle="--",
                    linewidth=1,
                    label=f"Average number of beds needed {lagged_by} hours after arrival{label_suffix}",
                )

            # Spread arrival rates
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates_spread"]),
                marker=data["marker"],
                color=data["color"],
                label=f"Average number of beds applying ED targets of {int(y1*100)}% in {int(x1)} hours{label_suffix}",
            )

        elif lagged_by is not None:
            # Base arrival rates
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates"]),
                marker=data["marker"],
                linestyle=":",
                color=data["color"],
                label=f"Arrival rates of admitted patients{label_suffix}",
            )

            # Lagged arrival rates
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates_lagged"]),
                marker=data["marker"],
                color=data["color"],
                label=f"Average number of beds needed {lagged_by} hours after arrival{label_suffix}",
            )

        else:
            # Only base arrival rates
            plt.plot(
                x_values,
                get_cyclic_data(data["arrival_rates"]),
                marker=data["marker"],
                color=data["color"],
                label=data["label"] if data["label"] else None,
            )

    # Set plot limits and labels
    plt.ylim(0, max(max_y_values) + 0.25)
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)

    plt.xlabel("Hour of day")
    plt.ylabel("Arrival Rate (patients per hour)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if any(d["label"] for d in processed_data):
        plt.legend()
    plt.tight_layout()

    # Save if path provided
    if media_file_path:
        filename = f"{file_prefix}{clean_title_for_filename(title)}"
        plt.savefig(media_file_path / filename, dpi=300)

    plt.show()


def plot_cumulative_arrival_rates(
    inpatient_arrivals,
    title,
    curve_params=None,
    lagged_by=None,
    time_interval=60,
    start_plot_index=0,
    draw_window=None,
    x_margin=0.5,
    file_prefix="",
    set_y_lim=None,
    hour_lines=[12, 17],
    line_styles={12: "--", 17: ":", 20: "--"},
    annotation_prefix="On average",
    line_colour="red",
    media_file_path=None,
    plot_centiles=False,
    highlight_centile=0.9,
    centiles=[0.3, 0.5, 0.7, 0.9, 0.99],
    markers=["D", "s", "^", "o", "v"],
    line_styles_centiles=["-.", "--", ":", "-", "-"],
    bed_type_spec="",
    text_y_offset=1,
):
    """
    Plot cumulative arrival rates with optional statistical distributions.

    Parameters
    ----------
    inpatient_arrivals : array-like
        Dataset of inpatient arrivals.
    title : str
        Title of the plot.
    curve_params : tuple of float, optional
        Parameters for spread rates as (x1, y1, x2, y2) (default is None).
    lagged_by : int, optional
        Time lag in hours for cumulative rates (default is None).
    time_interval : int, optional
        Time interval in minutes for rate calculations (default is 60).
    start_plot_index : int, optional
        Starting hour index for plotting (default is 0).
    draw_window : tuple of int, optional
        Time window for detailed annotation (default is None).
    x_margin : float, optional
        Margin on the x-axis (default is 0.5).
    file_prefix : str, optional
        Prefix for the saved file name (default is "").
    set_y_lim : float, optional
        Upper limit for the y-axis (default is None).
    hour_lines : list of int, optional
        Specific hours to annotate (default is [12, 17]).
    line_styles : dict, optional
        Line styles for hour annotations keyed by hour (default is {12: "--", 17: ":", 20: "--"}).
    annotation_prefix : str, optional
        Prefix for annotations (default is "On average").
    line_colour : str, optional
        Color for the main line plot (default is "red").
    media_file_path : str or Path, optional
        Directory path to save the plot (default is None).
    plot_centiles : bool, optional
        Whether to include percentile visualization (default is False).
    highlight_centile : float, optional
        Percentile to emphasize (default is 0.9).
    centiles : list of float, optional
        List of percentiles to calculate (default is [0.3, 0.5, 0.7, 0.9, 0.99]).
    markers : list of str, optional
        Marker styles for percentile lines (default is ["D", "s", "^", "o", "v"]).
    line_styles_centiles : list of str, optional
        Line styles for percentile visualization (default is ["-.", "--", ":", "-", "-"]).
    bed_type_spec : str, optional
        Specification for bed type in annotations (default is "").
    text_y_offset : float, optional
        Vertical offset for text annotations (default is 1).

    Returns
    -------
    None
        Displays or saves the matplotlib plot.
    """

    # Data processing
    if curve_params is not None:
        x1, y1, x2, y2 = curve_params
        arrival_rates_dict = true_demand_by_hour(inpatient_arrivals, x1, y1, x2, y2)
    elif lagged_by is not None:
        arrival_rates_dict = time_varying_arrival_rates_lagged(
            inpatient_arrivals, lagged_by, time_interval
        )
    else:
        arrival_rates_dict = time_varying_arrival_rates(
            inpatient_arrivals, time_interval
        )

    # Process arrival rates
    arrival_rates, hour_labels, hour_values = process_arrival_rates(arrival_rates_dict)

    # Reindex based on start_plot_index
    rates_reindexed = (
        list(arrival_rates)[start_plot_index:] + list(arrival_rates)[0:start_plot_index]
    )
    labels_reindexed = (
        list(hour_labels)[start_plot_index:] + list(hour_labels)[0:start_plot_index]
    )

    # Calculate percentiles
    percentiles = [[] for _ in range(len(centiles))]
    cumulative_value_at_centile = np.zeros(len(centiles))

    for hour in range(len(rates_reindexed)):
        for i, centile in enumerate(centiles):
            value_at_centile = stats.poisson.ppf(centile, rates_reindexed[hour])
            cumulative_value_at_centile[i] += value_at_centile
            percentiles[i].append(value_at_centile)

    # Set up plot
    plt.figure(figsize=(10, 6))

    # Plot mean line with appropriate label
    label_suffix = f" {bed_type_spec} beds needed" if bed_type_spec else " beds needed"

    cumsum_rates = np.cumsum(rates_reindexed)
    plt.plot(
        labels_reindexed,
        cumsum_rates,
        marker="o",
        markersize=3,
        color=line_colour,
        linewidth=2,
        alpha=0.7,
        label=f"Average number of{label_suffix}",
    )

    max_y = cumsum_rates[-1]

    if plot_centiles:
        # Calculate and plot percentiles
        percentiles = [[] for _ in range(len(centiles))]
        cumulative_value_at_centile = np.zeros(len(centiles))

        for hour in range(len(rates_reindexed)):
            for i, centile in enumerate(centiles):
                value_at_centile = stats.poisson.ppf(centile, rates_reindexed[hour])
                cumulative_value_at_centile[i] += value_at_centile
                percentiles[i].append(value_at_centile)

        # Plot percentile lines
        for i, centile in enumerate(centiles):
            marker = markers[i % len(markers)]
            line_style = line_styles_centiles[i % len(line_styles_centiles)]
            linewidth = 2 if centile == highlight_centile else 1
            alpha = 1.0 if centile == highlight_centile else 0.7

            cumsum_percentile = np.cumsum(percentiles[i])
            plt.plot(
                labels_reindexed,
                cumsum_percentile,
                marker=marker,
                markersize=3,
                linestyle=line_style,
                color="C0",
                linewidth=linewidth,
                alpha=alpha,
                label=f"{centile*100:.0f}% probability",
            )

            if centile == highlight_centile:
                for hour_line in hour_lines:
                    cumsum_at_hour = sum(
                        percentiles[i][0 : hour_line + 1 - start_plot_index]
                    )

                    annotate_hour_line(
                        hour_line=hour_line,
                        y_value=cumsum_at_hour,
                        hour_values=hour_values,
                        start_plot_index=start_plot_index,
                        line_styles=line_styles,
                        x_margin=x_margin,
                        annotation_prefix=annotation_prefix,
                        text_y_offset=text_y_offset,
                        # No slope/x1/y1 needed since we're using the simpler case
                    )

        max_y = max(cumulative_value_at_centile)

        # Reverse legend order
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc="upper left")
    else:
        plt.legend(loc="upper left")

    # Handle draw window if specified
    if not plot_centiles:
        if draw_window:
            start_window, end_window = draw_window

            plt.hlines(
                y=cumsum_rates[-1],
                xmin=hour_values[end_window - start_plot_index],
                xmax=hour_values[-1],
                color="blue",
                linestyle="--",
            )

            x1 = hour_values[start_window - start_plot_index]
            y1 = cumsum_rates[start_window - start_plot_index]
            x2 = hour_values[end_window - start_plot_index]
            y2 = cumsum_rates[-1]

            plt.plot([x1, x2], [y1, y2], color="blue", linestyle="--")

            slope = (y2 - y1) / (x2 - x1)
            plt.annotate(
                f"{annotation_prefix}, {int(slope)} beds need to be vacated\n"
                f"each hour between {start_window}:00 and {end_window}:00\n"
                f"to create capacity for all overnight arrivals\n"
                f"by {end_window}:00",
                xy=(hour_values[-1], y2 * 0.25),
                xytext=(hour_values[-1], y2 * 0.25),
                va="top",
                ha="right",
            )
            for hour_line in hour_lines:
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=y1,
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                    slope=slope,
                    x1=x1,
                    y1=y1,
                )

        else:
            for hour_line in hour_lines:
                annotate_hour_line(
                    hour_line=hour_line,
                    y_value=cumsum_rates[hour_line - start_plot_index],
                    hour_values=hour_values,
                    start_plot_index=start_plot_index,
                    line_styles=line_styles,
                    x_margin=x_margin,
                    annotation_prefix=annotation_prefix,
                )

    # Final plot formatting
    plt.xlabel("Hour of day")
    plt.ylabel("Cumulative number of beds needed")
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)
    plt.ylim(0, set_y_lim if set_y_lim else max_y + 2)
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(5))

    plt.title(title)
    plt.tight_layout()

    if media_file_path:
        filename = f"{file_prefix}{clean_title_for_filename(title)}"
        plt.savefig(media_file_path / filename, dpi=300)
    plt.show()
