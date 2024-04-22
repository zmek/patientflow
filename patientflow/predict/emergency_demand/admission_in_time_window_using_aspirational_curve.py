import numpy as np
import matplotlib.pyplot as plt


def growth_curve(x, a, k_growth):
    """
    Logistic-like growth function.

    Parameters:
    - x (float): The x-value at which to evaluate the curve.
    - a (float): Scaling factor for the growth phase.
    - k_growth (float): Growth rate.

    Returns:
    - float: The y-value of the growth curve at x.
    """
    return a * (np.exp(k_growth * x) - 1)


def decay_curve(x, x1, b, k_decay):
    """
    Exponential decay function.

    Parameters:
    - x (float): The x-value at which to evaluate the curve.
    - x1 (float): The x-value where decay starts.
    - b (float): Scaling factor for the decay phase.
    - k_decay (float): Decay rate.

    Returns:
    - float: The y-value of the decay curve at x.
    """
    return 1 - b * np.exp(-k_decay * (x - x1))


def create_curve(x1, y1):

    # Constants for growth
    k_growth = 1 / x1
    a = y1 / (np.exp(k_growth * x1) - 1)

    # Constants for decay
    k_decay = k_growth  # Using the same k for simplicity
        b = (1 - y1) / np.exp(-k_decay * 0)  # x - x1 is 0 at x = x1

    # Generate x values
    x_values = np.linspace(0, 24, 200)

    # Compute y values for each x
    y_values = [
        growth_curve(x, a, k_growth) if x <= x1 else decay_curve(x, x1, b, k_decay)
        for x in x_values
    ]

    return x_values, y_values, a, k_growth, b, k_decay


def get_y_from_aspirational_curve(x, x1, y1):
    """
    Calculate the probability of admission (y) for a given time since arrival (x) on the curve.

    Parameters:
    x (float): The x value at which to calculate the y value.
    x1 (float): The transition point on the x-axis from growth to decay.
    y1 (float): The transition point on the x-axis from growth to decay

    Returns:
    float: The y value corresponding to the given x.
    """

    x_values, y_values, a, k_growth, b, k_decay = create_curve(x1, y1)

    if x <= x1:
        return growth_curve(x, a, k_growth)
    else:
        return decay_curve(x, x1, b, k_decay)


def calculate_probability(elapsed_los_td_hrs, time_window_hrs, x1, y1):

    # probability of still being in the ED now (a function of elapsed time since arrival)
    prob_still_being_in_now = get_y_from_aspirational_curve(elapsed_los_td_hrs, x1, y1)

    # prob admission when adding the time window added to elapsed time since arrival
    prob_admission_within_elapsed_time_plus_time_window = get_y_from_aspirational_curve(
        elapsed_los_td_hrs + time_window_hrs, x1, y1
    )

    # prob admission within time window given arrival time
    return (
        prob_admission_within_elapsed_time_plus_time_window - prob_still_being_in_now
    ) / (1 - prob_still_being_in_now)


def plot_curve(full_path, x1, y1):
    x_values, y_values, a, k_growth, b, k_decay = create_curve(x1, y1)

    # Plot the curve
    plt.figure()
    plt.plot(x_values, y_values)
    plt.scatter(x1, y1, color="red")  # Mark the point (x1, y1)
    plt.title(
        "Drawing hypothetical curve reflecting a 4 hour target for 76% of patients",
        fontsize=9,
    )
    plt.xlabel("Hours since admission")
    plt.ylabel("Probability of admission by this point")
    plt.axhline(y=y1, color="green", linestyle="--", label="y = 76%")
    plt.axvline(x=x1, color="gray", linestyle="--", label="x = 4 hours")
    plt.legend()
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.show()
