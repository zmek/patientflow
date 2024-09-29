from patientflow.prepare import prepare_for_inference


# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#aec7e8"


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


def plot_calibration(prediction_times, media_file_path, model_file_path, visits_csv_path, strategy = 'uniform'):
    num_plots = len(prediction_times)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))

    for i, _prediction_time in enumerate(prediction_times):
        X_test, y_test, pipeline = prepare_for_inference(model_file_path = model_file_path, model_name = 'ed_admission', prediction_time = _prediction_time, data_path = visits_csv_path, single_snapshot_per_visit = False, model_only = False)

        prob_true, prob_pred = calibration_curve(y_test, pipeline.predict_proba(X_test)[:, 1], n_bins=10, strategy=strategy)

        ax = axs[i]

        hour, minutes = _prediction_time

        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Predictions', color=primary_color)
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color=secondary_color)
        ax.set_title(f'Calibration Plot for {hour}:{minutes:02}', fontsize = 14)
        ax.set_xlabel('Mean Predicted Probability', fontsize = 12)
        ax.set_ylabel('Fraction of Positives', fontsize = 12)
        ax.legend()

    plt.tight_layout()

    calib_plot_path = media_file_path / 'calibration_plot'
    calib_plot_path = calib_plot_path.with_suffix('.png')
    
    plt.savefig(calib_plot_path)
    plt.show()

    
