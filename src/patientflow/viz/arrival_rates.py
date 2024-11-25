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
    slope=None,
    x1=None,
    y1=None,
):
    """
    Annotate hour lines on a matplotlib plot with consistent formatting.

    Args:
        hour_line (int): Hour to annotate
        y_value (float): Y-value for the annotation
        hour_values (list): List of hour values for x-axis
        start_plot_index (int): Starting index for plot
        line_styles (dict): Dictionary of line styles for different hours
        x_margin (float): Margin for x-axis
        annotation_prefix (str): Prefix for annotation text
        slope (float, optional): Slope for calculating y position if using window
        x1 (float, optional): Starting x position if using window
        y1 (float, optional): Starting y position if using window
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
        y_position = y_a + 1
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
        )
        y_position = y_value + 1

    plt.annotate(
        annotation_text,
        xy=(a / 2 if slope is not None else a, y_value),
        xytext=(hour_values[1] - x_margin, y_position),
        va="bottom",
        ha="left",
        fontsize=10,
    )


def plot_arrival_rates(
    inpatient_arrivals,
    title,
    lagged_by=None,
    curve_params=None,  # specify for spread arrival rates
    time_interval=60,
    start_plot_index=0,
    x_margin=0.5,
    file_prefix="",
    media_file_path=None,
):
    # Calculate arrival rates - returns a dict
    arrival_rates_dict = time_varying_arrival_rates(inpatient_arrivals, time_interval)

    # Get values, hour labels and hour values from the dict
    arrival_rates, hour_labels, hour_values = process_arrival_rates(arrival_rates_dict)

    if lagged_by is not None:
        arrival_rates_lagged_dict = time_varying_arrival_rates_lagged(
            inpatient_arrivals, lagged_by, time_interval
        )
        arrival_rates_lagged, _, _ = process_arrival_rates(arrival_rates_lagged_dict)

    if curve_params is not None:
        x1, y1, x2, y2 = curve_params
        arrival_rates_spread_dict = true_demand_by_hour(
            inpatient_arrivals, x1, y1, x2, y2
        )
        arrival_rates_spread, _, _ = process_arrival_rates(arrival_rates_spread_dict)
    else:
        arrival_rates_spread = None

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Helper function to create cyclic data
    def get_cyclic_data(data):
        return data[start_plot_index:] + data[0:start_plot_index]

    # Get x-axis values
    x_values = get_cyclic_data(hour_labels)
    # Get base arrival rates
    y_values = get_cyclic_data(arrival_rates)

    # Define plotting styles based on scenario
    if arrival_rates_spread is not None and len(arrival_rates_spread) > 0:
        # Base arrival rates
        plt.plot(
            x_values,
            y_values,
            marker="x",
            color="grey",
            markersize=4,
            linestyle=":",
            linewidth=1,
            label="Arrival rates of admitted patients",
        )

        # Lagged arrival rates
        plt.plot(
            x_values,
            get_cyclic_data(arrival_rates_lagged),
            marker="o",
            markersize=4,
            color="grey",
            linestyle="--",
            linewidth=1,
            label="Average number of beds needed 4 hours after arrival",
        )

        # Spread arrival rates
        plt.plot(
            x_values,
            get_cyclic_data(arrival_rates_spread),
            marker="o",
            label=f"Average number of beds applying ED targets of {int(y1*100)}% in {int(x1)} hours",
        )

    elif lagged_by is not None:
        # Base arrival rates
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linestyle=":",
            label="Arrival rates of admitted patients",
        )

        # Lagged arrival rates
        plt.plot(
            x_values,
            get_cyclic_data(arrival_rates_lagged),
            marker="o",
            color="C0",
            label="Average number of beds needed 4 hours after arrival",
        )

    else:
        # Only base arrival rates
        plt.plot(x_values, y_values, marker="o")

    # plt.xticks(rotation=45)
    plt.ylim(0, max(arrival_rates) + 0.25)
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)

    plt.xlabel("Hour of day")
    plt.ylabel("Arrival Rate (patients per hour)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
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
):
    """
    Plot cumulative arrival rates with statistical distributions for inpatient data.

    Additional Parameters:
    highlight_centile: Percentile to emphasize in visualization
    centiles: List of percentiles to calculate and display
    markers: List of markers for different percentile lines
    line_styles_centiles: List of line styles for percentile visualization
    bed_type_spec: Specification for bed type (if any)
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

                    plt.vlines(
                        hour_values[hour_line - start_plot_index],
                        0,
                        cumsum_at_hour,
                        linestyles=line_styles[hour_line],
                        colors="C0",
                    )

                    plt.hlines(
                        cumsum_at_hour,
                        hour_values[0] - x_margin,
                        hour_values[hour_line - start_plot_index],
                        linestyles=line_styles[hour_line],
                        colors="C0",
                    )

                    plt.annotate(
                        f"{int(cumsum_at_hour)} beds needed by {hour_line}:00",
                        xy=(hour_values[0] - x_margin / 2, cumsum_at_hour + 2),
                        ha="left",
                        va="bottom",
                        fontsize=10,
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


def plot_dual_arrival_rates(
    inpatient_arrivals_1,
    inpatient_arrivals_2,
    title,
    labels=("Set 1", "Set 2"),
    lagged_by=None,
    curve_params=None,
    time_interval=60,
    start_plot_index=0,
    x_margin=0.5,
    file_prefix="",
    media_file_path=None,
):
    """
    Plot two sets of inpatient arrivals on the same chart for comparison.
    When spread rates are present, they use the main colors while other rates are grey.
    """
    # Calculate arrival rates for both sets
    def process_dataset(arrivals):
        rates_dict = time_varying_arrival_rates(arrivals, time_interval)
        rates, hour_labels, hour_values = process_arrival_rates(rates_dict)
        
        if lagged_by is not None:
            lagged_dict = time_varying_arrival_rates_lagged(arrivals, lagged_by, time_interval)
            rates_lagged, _, _ = process_arrival_rates(lagged_dict)
        else:
            rates_lagged = None
            
        if curve_params is not None:
            x1, y1, x2, y2 = curve_params
            spread_dict = true_demand_by_hour(arrivals, x1, y1, x2, y2)
            rates_spread, _, _ = process_arrival_rates(spread_dict)
        else:
            rates_spread = None
            
        return rates, rates_lagged, rates_spread, hour_labels, hour_values
    
    # Process both datasets
    rates1, rates1_lagged, rates1_spread, hour_labels, hour_values = process_dataset(inpatient_arrivals_1)
    rates2, rates2_lagged, rates2_spread, _, _ = process_dataset(inpatient_arrivals_2)
    
    # Helper function for cyclic data
    def get_cyclic_data(data):
        return data[start_plot_index:] + data[0:start_plot_index]
    
    # Create plot
    plt.figure(figsize=(12, 7))
    x_values = get_cyclic_data(hour_labels)
    
    # Define main colors for each group
    color1 = '#1f77b4'  # Blue
    color2 = '#9467bd'  # Purple
    
    # Define grey shades
    grey1 = '#888888'  # Darker grey
    grey2 = '#BBBBBB'  # Lighter grey
    
    # Plot first dataset
    y_values1 = get_cyclic_data(rates1)
    if rates1_spread is not None and len(rates1_spread) > 0:
        # Base arrival rates in grey
        plt.plot(
            x_values,
            y_values1,
            marker="x",
            color=grey1,
            markersize=4,
            linestyle=":",
            linewidth=1,
            label=f"{labels[0]} - Arrivals",
        )
        
        # Lagged arrival rates in lighter grey
        if rates1_lagged is not None:
            plt.plot(
                x_values,
                get_cyclic_data(rates1_lagged),
                marker="o",
                markersize=4,
                color=grey2,
                linestyle="--",
                linewidth=1,
                label=f"{labels[0]} - Lagged arrivals",
            )
        
        # Spread arrival rates in main color
        plt.plot(
            x_values,
            get_cyclic_data(rates1_spread),
            marker="o",
            color=color1,
            label=f"{labels[0]} - Aspirational ({int(curve_params[1]*100)}% in {int(curve_params[0])}h)",
        )
    else:
        # If no spread rates, use normal coloring
        plt.plot(
            x_values,
            y_values1,
            marker="o",
            color=color1,
            linestyle="-",
            label=f"{labels[0]} - Arrivals",
        )
        
        if rates1_lagged is not None:
            plt.plot(
                x_values,
                get_cyclic_data(rates1_lagged),
                marker="s",
                color=color1,
                linestyle="--",
                label=f"{labels[0]} - Lagged arrivals",
            )
    
    # Plot second dataset
    y_values2 = get_cyclic_data(rates2)
    if rates2_spread is not None and len(rates2_spread) > 0:
        # Base arrival rates in grey
        plt.plot(
            x_values,
            y_values2,
            marker="x",
            color=grey1,
            markersize=4,
            linestyle=":",
            linewidth=1,
            label=f"{labels[1]} - Arrivals",
        )
        
        # Lagged arrival rates in lighter grey
        if rates2_lagged is not None:
            plt.plot(
                x_values,
                get_cyclic_data(rates2_lagged),
                marker="o",
                markersize=4,
                color=grey2,
                linestyle="--",
                linewidth=1,
                label=f"{labels[1]} - Lagged arrivals",
            )
        
        # Spread arrival rates in main color
        plt.plot(
            x_values,
            get_cyclic_data(rates2_spread),
            marker="o",
            color=color2,
            label=f"{labels[1]} - Aspirational ({int(curve_params[1]*100)}% in {int(curve_params[0])}h)",
        )
    else:
        # If no spread rates, use normal coloring
        plt.plot(
            x_values,
            y_values2,
            marker="o",
            color=color2,
            linestyle="-",
            label=f"{labels[1]} - Arrivals",
        )
        
        if rates2_lagged is not None:
            plt.plot(
                x_values,
                get_cyclic_data(rates2_lagged),
                marker="s",
                color=color2,
                linestyle="--",
                label=f"{labels[1]} - Lagged arrivals",
            )
    
    # Set plot parameters
    plt.ylim(0, max(max(rates1), max(rates2)) + 0.25)
    plt.xlim(hour_values[0] - x_margin, hour_values[-1] + x_margin)
    plt.xlabel("Hour of day")
    plt.ylabel("Arrival Rate (patients per hour)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if media_file_path:
        filename = f"{file_prefix}{clean_title_for_filename(title)}"
        plt.savefig(media_file_path / filename, dpi=300, bbox_inches='tight')
    
    plt.show()