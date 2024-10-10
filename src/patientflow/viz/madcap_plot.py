from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.prepare import prepare_for_inference

exclude_from_training_data = [
    "visit_number",
    "snapshot_date",
    "prediction_time",
    "specialty",
    "consultation_sequence",
     "final_sequence"]


# Create a function to classify age groups
def classify_age_group(age_group):
    if age_group in ['0-17']:
        return 'children'
    elif age_group in ['18-24', '25-34', '35-44', '45-54', '55-64']:
        return 'adults'
    elif age_group in ['65-74', '75-102']:
        return 'over 65'
    else:
        return 'unknown' # Handle potential NaN or unexpected values

def classify_age(age_on_arrival):
    if age_on_arrival < 18:
        return 'children'
    elif age_on_arrival < 65:
        return 'adults'
    elif age_on_arrival >= 65:
        return '65 or over'
    else:
        return 'unknown' # Handle potential NaN or unexpected values
    
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

def generate_madcap_plots(prediction_times, model_file_path, media_file_path, visits_csv_path):
    num_plots = len(prediction_times)
    
    # Calculate the number of rows and columns for the subplots
    num_cols = min(num_plots, 5)  # Maximum 5 columns
    num_rows = math.ceil(num_plots / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_plots * 5, 4))

    # Ensure axes is always a 2D array
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, _prediction_time in enumerate(prediction_times):
        
        X_test, y_test, pipeline = prepare_for_inference(
            model_file_path, 
            'admissions', 
            prediction_time = _prediction_time, 
            data_path = visits_csv_path, 
            single_snapshot_per_visit = False)

        predict_proba = pipeline.predict_proba(X_test)[:,1]

        row = i // num_cols
        col = i % num_cols
        plot_madcap_subplot(predict_proba, y_test, _prediction_time, axes[row, col])

    # Hide any unused subplots
    for j in range(i+1, num_rows*num_cols):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    
    if media_file_path:
        plot_name = "madcap_plot"
        madcap_plot_path = Path(media_file_path) / plot_name
        plt.savefig(madcap_plot_path)

    plt.show()
    plt.close(fig)

def plot_madcap_subplot(predict_proba, label, _prediction_time, ax):

    hour, minutes = _prediction_time
    # Ensure inputs are numpy arrays
    predict_proba = np.array(predict_proba)
    label = np.array(label)

    # Sort by predict_proba
    sorted_indices = np.argsort(predict_proba)
    sorted_proba = predict_proba[sorted_indices]
    sorted_label = label[sorted_indices]

    # Compute unique probabilities and their mean labels
    unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
    mean_labels = np.zeros_like(unique_probs)

    np.add.at(mean_labels, inverse_indices, sorted_label)
    counts = np.bincount(inverse_indices)
    mean_labels = mean_labels / counts

    # Cumulative sums for model and observed
    model = np.cumsum(sorted_proba)
    observed = np.cumsum(mean_labels[inverse_indices])

    x = np.arange(len(sorted_proba))

    # Plot
    ax.plot(x, model, label="model")
    ax.plot(x, observed, label="observed")
    ax.legend(loc="upper left", fontsize='x-small')
    ax.set_xlabel("Test set visits ordered by predicted probability", fontsize=12)
    ax.set_ylabel("Number of admissions", fontsize=12)
    ax.set_title(f"MADCAP Plot for {hour}:{minutes:02}", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize='x-small')



def plot_madcap_by_group(predict_proba, label, group, _prediction_time, group_name, media_path=None, plot_difference=True):

    hour, minutes = _prediction_time

    predict_proba, label, group = map(np.array, (predict_proba, label, group))
    unique_groups = [grp for grp in np.unique(group) if grp != 'unknown']

    fig_size = (10, 8) if plot_difference else (10, 4)
    fig, ax = plt.subplots(2 if plot_difference else 1, len(unique_groups), figsize=fig_size)
    ax = ax.reshape(-1, len(unique_groups)) if plot_difference else ax.reshape(1, -1)

    for i, grp in enumerate(unique_groups):
        mask = group == grp
        sorted_indices = np.argsort(predict_proba[mask])
        sorted_proba = predict_proba[mask][sorted_indices]
        sorted_label = label[mask][sorted_indices]

        unique_probs, inverse_indices = np.unique(sorted_proba, return_inverse=True)
        mean_labels = np.bincount(inverse_indices, weights=sorted_label) / np.bincount(inverse_indices)

        model = np.cumsum(sorted_proba)
        observed = np.cumsum(mean_labels[inverse_indices])
        x = np.arange(len(sorted_proba))

        ax[0, i].plot(x, model, label="model")
        ax[0, i].plot(x, observed, label="observed")
        ax[0, i].legend(loc="upper left", fontsize=8)
        ax[0, i].set_xlabel("Test set visits ordered by predicted probability", fontsize=8)
        ax[0, i].set_ylabel("Number of admissions", fontsize=8)
        ax[0, i].set_title(f"{group_name}: {grp!s}", fontsize=10) 
        ax[0, i].tick_params(axis='both', which='major', labelsize=8)

        if plot_difference:
            ax[1, i].plot(x, model - observed)
            ax[1, i].set_xlabel("Test set visits ordered by predicted probability", fontsize=8)
            ax[1, i].set_ylabel("Expected number of admissions - observed", fontsize=8)
            ax[1, i].set_title(f"{group_name}: {grp!s}", fontsize=10)
            ax[1, i].tick_params(axis='both', which='major', labelsize=8)

    fig.suptitle(f"MADCAP Plots for {hour}:{minutes:02}", fontsize=14)
    fig.tight_layout(pad=1.08, rect=[0, 0.03, 1, 0.95])

    if media_path:
        plot_name = f"madcap_plot_by_{group_name}_{hour}{minutes:02}.png"
        madcap_plot_path = Path(media_path) / plot_name
        plt.savefig(madcap_plot_path, dpi=300, bbox_inches='tight')
    plt.show()




def generate_madcap_plots_by_group(prediction_times, model_file_path, media_file_path, visits_csv_path, grouping_var, grouping_var_name ):
            
    for i, _prediction_time in enumerate(prediction_times):
        hour, minutes = _prediction_time
        dataset = f'{hour}{minutes:02}'
        
        X_test, y_test, pipeline = prepare_for_inference(
            model_file_path, 
            'admissions', 
            prediction_time = _prediction_time, 
            data_path = visits_csv_path, 
            single_snapshot_per_visit = False)
        
        predict_proba = pipeline.predict_proba(X_test)[:,1]

        if 'age_group' in X_test.columns:

            group = X_test['age_group'].apply(classify_age_group)
        else:
            group = X_test['age_on_arrival'].apply(classify_age)
            
        plot_madcap_by_group(predict_proba, y_test , group, 
                                _prediction_time, grouping_var_name, media_file_path, plot_difference = False)
        

            
