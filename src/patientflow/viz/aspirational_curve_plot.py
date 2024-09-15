import os

import matplotlib.pyplot as plt
import numpy as np
from .predict.admission_in_prediction_window import (
    create_curve,
)


def plot_curve(
    title,
    x1,
    y1,
    x2,
    y2,
    figsize=(6, 3),
    include_titles=False,
    text_size=None,
    directory_path=None,
    file_name=None,
):
    gamma, lamda, a, x_values, y_values = create_curve(
        x1, y1, x2, y2, generate_values=True
    )

    # Plot the curve
    plt.figure(figsize=figsize)

    if not file_name:
        file_name = (
            title.replace(" ", "_").replace("/n", "_").replace("%", "percent") + ".png"
        )

    plt.plot(x_values, y_values)
    plt.scatter(x1, y1, color="red")  # Mark the point (x1, y1)
    plt.scatter(x2, y2, color="red")  # Mark the point (x2, y2)

    if text_size:
        plt.tick_params(axis="both", which="major", labelsize=text_size)

    x_ticks = np.arange(min(x_values), max(x_values) + 1, 2)
    plt.xticks(x_ticks)

    if include_titles:
        plt.title(title, fontsize=text_size)
        plt.xlabel("Hours since admission", fontsize=text_size)
        plt.ylabel("Probability of admission by this point", fontsize=text_size)

    plt.axhline(y=y1, color="green", linestyle="--", label=f"y ={int(y1*100)}%")
    plt.axvline(x=x1, color="gray", linestyle="--", label="x = 4 hours")
    plt.legend(fontsize=text_size)

    plt.tight_layout()

    if directory_path:
        os.makedirs(directory_path, exist_ok=True)
        plt.savefig(directory_path / file_name, dpi=300)

    plt.show()
