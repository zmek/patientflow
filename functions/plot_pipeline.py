import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def create_colour_dict():
    # Define the color mapping dictionary
    spec_colour_dict = {
        "single": {
            "medical": "#ED7D31",  # red
            "surgical": "#70AD47",  # green
            "haem/onc": "#FFC000",  # yellow
            "paediatric": "#5B9BD5",  # blue
            "all": "#44546A",  # dark blue
            "window": "#A9A9A9",
        },
        "spectrum": {},
    }

    # Function to generate a continuous colormap
    def generate_continuous_colormap(base_color):
        base = mcolors.to_rgb(base_color)
        cdict = {
            "red": [(0.0, 1.0, 1.0), (1.0, base[0], base[0])],
            "green": [(0.0, 1.0, 1.0), (1.0, base[1], base[1])],
            "blue": [(0.0, 1.0, 1.0), (1.0, base[2], base[2])],
        }
        return mcolors.LinearSegmentedColormap(
            "custom_colormap", segmentdata=cdict, N=256
        )

    # Populate the spectrum dictionary with continuous colormaps
    for spec, color in spec_colour_dict["single"].items():
        spec_colour_dict["spectrum"][spec] = generate_continuous_colormap(color)

    return spec_colour_dict


def in_ED_now_plot(
    directory_path,
    file_name,
    ex,
    horizon_datetime,
    figsize,
    title_,
    include_titles=False,
    truncate_at_hours=8,
    colour=False,
    text_size=None,
    jitter_amount=0.1,
    size=50,
    preds_col="preds",
    colour_map="Spectral_r",
    title_suffix="admission",
):
    spec_colour_dict = create_colour_dict()

    figsize_x, figsize_y = figsize

    ex = ex[ex.elapsed_los / 3600 < truncate_at_hours]

    # Create a dictionary to map ordinal categories to numerical values
    unique_locations = sorted(ex["loc_new"].unique())
    loc_to_num = {loc: i for i, loc in enumerate(unique_locations)}

    if colour:
        plt.figure(figsize=(figsize_x, figsize_y))
        title_ = title_ + " with predicted probability of " + title_suffix
        scatter_plots = []
        for location, group in ex.groupby("loc_new"):
            jittered_y = loc_to_num[location] + np.random.uniform(
                -jitter_amount, jitter_amount, size=len(group)
            )
            # Collect scatter plots
            scatter = plt.scatter(
                group["elapsed_los"] / 3600,
                jittered_y,
                c=group[preds_col],
                cmap=colour_map,
                vmin=0,
                vmax=1,
                label=location,
                s=size,
            )
            scatter_plots.append(scatter)
        # plt.colorbar(scatter_plots[-1], orientation='vertical')
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=colour_map, norm=plt.Normalize(vmin=0, vmax=1)),
            ax=plt.gca(),
            orientation="vertical",
        )

    else:
        plt.figure(figsize=(figsize_x - 1, figsize_y))
        for location, group in ex.groupby("loc_new"):
            # Add jitter: modify the numerical y-coordinate with a small, random offset
            jittered_y = loc_to_num[location] + np.random.uniform(
                -jitter_amount, jitter_amount, size=len(group)
            )
            plt.scatter(
                group["elapsed_los"] / 3600,
                jittered_y,
                color=spec_colour_dict["single"]["all"],
                label=location,
                s=size,
            )

    plt.xlim(0, truncate_at_hours)
    plt.gca().invert_yaxis()

    # Replace numerical ticks with the original category names
    plt.yticks(ticks=range(len(unique_locations)), labels=unique_locations)

    if text_size:
        plt.tick_params(axis="both", which="major", labelsize=text_size)

        if colour:
            cbar.ax.tick_params(labelsize=text_size)

    if include_titles:
        plt.title(title_, fontsize=text_size)
        plt.xlabel("Hours since admission")
        plt.ylabel("ED Pathway")

    # file_name = title_.replace(' ', '_').replace('/n', '_').replace('%', 'percent') + '.png'

    plt.tight_layout()
    os.makedirs(directory_path, exist_ok=True)
    plt.savefig(directory_path / file_name.replace(" ", "_"), dpi=300)

    plt.show()
