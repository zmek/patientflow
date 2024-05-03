import numpy as np
import matplotlib.pyplot as plt


# Growth component (0 <= x <= x1)
def growth_curve(x, a, gamma):
    return a * (np.exp(x*gamma))

# Decay component (x >= x1)
def decay_curve(x, x1, y1, lamda):
    return y1 + (1-y1)*(1-np.exp(-lamda*(x-x1)))

def create_curve(x1, y1, x2, y2, a = 0.01):
    
    # Constants for growth
    gamma = np.log(y1/a)/x1
    
    # Constants for decay
    x_delta = x2 - x1
    lamda = np.log((1-y1)/(1-y2))/x_delta

    # Generate x values
    x_values = np.linspace(0, 20, 200)

    # Compute y values for each x
    y_values = [growth_curve(x, a, gamma) if x <= x1 else decay_curve(x, x1, y1, lamda) for x in x_values]
    return gamma, lamda, a, x_values, y_values

def get_y_from_aspirational_curve(x, x1, y1, x2, y2):
    """
    Calculate the probability of admission (y) for a given time since arrival (x) on the curve.

    Parameters:
    x (float): The x value at which to calculate the y value.
    x1 (float): The transition point on the x-axis from growth to decay.
    y1 (float): The transition point on the x-axis from growth to decay

    Returns:
    float: The y value corresponding to the given x.
    """

    gamma, lamda, a, x_values, y_values = create_curve(x1, y1, x2, y2)

    if x < x1:
        return growth_curve(x, a, gamma)
    else:
        return decay_curve(x, x1, y1, lamda)


def calculate_probability(elapsed_los_td_hrs, time_window_hrs, x1, y1, x2, y2):

    # probability of still being in the ED now (a function of elapsed time since arrival)
    prob_admission_prior_to_now = get_y_from_aspirational_curve(elapsed_los_td_hrs, x1, y1, x2, y2)

    # prob admission when adding the time window added to elapsed time since arrival
    prob_admission_by_end_of_window = get_y_from_aspirational_curve(
        elapsed_los_td_hrs + time_window_hrs, x1, y1, x2, y2
    )
    
    # when elapsed_los_td_hrs is extremely high (> 94 hours when x1=4, x2=12, y1=0.76, y2=0.99), 
    # get_y_from_aspirational_curve returns 1.0 for prob_admission_prior_to_now
    # despite the curve being asymptotic
    # causing a divide by zero error
    # also, when elapsed_los_td_hrs is extremely high, prob_admission_by_end_of_window approaches prob_admission_prior_to_now in value
    # in this case, return 1
    if prob_admission_prior_to_now == 1:
        return(1.0)

    # prob admission within time window given arrival time
    return (
        prob_admission_by_end_of_window - prob_admission_prior_to_now
    ) / (1 - prob_admission_prior_to_now)


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
