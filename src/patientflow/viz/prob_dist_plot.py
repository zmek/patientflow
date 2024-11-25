import itertools
import numpy as np
from matplotlib import pyplot as plt

def prob_dist_plot(
    prob_dist_data,
    title,
    directory_path=None,
    figsize=(6, 3),
    include_titles=False,
    truncate_at_beds=(0, 20),
    text_size=None,
    bar_colour="#5B9BD5",
    file_name=None,
    min_beds_lines=None,
    plot_min_beds_lines=True,
    plot_bed_base=None,
):
    plt.figure(figsize=figsize)
    if not file_name:
        file_name = title.replace(" ", "_").replace("/n", "_").replace("%", "percent") + ".png"
    
    if isinstance(truncate_at_beds, (int, float)):
        upper_bound = truncate_at_beds
        lower_bound = 0
    else:
        lower_bound, upper_bound = truncate_at_beds
        lower_bound = max(0, lower_bound) if lower_bound > 0 else lower_bound
    
    mask = (prob_dist_data.index >= lower_bound) & (prob_dist_data.index <= upper_bound)
    filtered_data = prob_dist_data[mask]
    
    plt.bar(
        filtered_data.index,
        filtered_data["agg_proba"].values,
        color=bar_colour,
    )
    
    tick_start = (lower_bound // 5) * 5
    tick_end = upper_bound + 1
    plt.xticks(np.arange(tick_start, tick_end, 5))
    
    if plot_min_beds_lines and min_beds_lines:
        colors = itertools.cycle(plt.cm.gray(np.linspace(0.3, 0.7, len(min_beds_lines))))
        for point in min_beds_lines:
            plt.axvline(
                x=prob_dist_data.index[min_beds_lines[point]],
                linestyle='--',
                linewidth=2,
                color=next(colors),
                label=f'{point*100:.0f}% probability'
            )
        plt.legend(loc='upper right', fontsize=14)
    
    if plot_bed_base:
        for point in plot_bed_base:
            plt.axvline(
                x=plot_bed_base[point],
                linewidth=2,
                color='red',
                label='bed balance'
            )
        plt.legend(loc='upper right', fontsize=14)
    
    if text_size:
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('Number of beds', fontsize=text_size)
        if include_titles:
            plt.title(title, fontsize=text_size)
            plt.ylabel("Probability", fontsize=text_size)
    
    plt.tight_layout()
    if directory_path:
        plt.savefig(directory_path / file_name.replace(" ", "_"), dpi=300)
    plt.show()