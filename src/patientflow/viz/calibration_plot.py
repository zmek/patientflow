import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.load import get_model_name

# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#aec7e8"


def plot_calibration(
    prediction_times,
    media_file_path,
    trained_models,
    test_visits,
    exclude_from_training_data,
    strategy="uniform",
    model_group_name="admssions",
    model_name_suffix=None
):
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60
        + x[1],  # Convert (hour, minute) to minutes since midnight
    )
    num_plots = len(prediction_times_sorted)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    for i, prediction_time in enumerate(prediction_times_sorted):
        # Get model name and pipeline for this prediction time
        model_name = get_model_name(model_group_name, prediction_time)
        if model_name_suffix:
            model_name = f"{model_name}_{model_name_suffix}"
        pipeline = trained_models[model_name]

        # Get test data for this prediction time
        X_test, y_test = get_snapshots_at_prediction_time(
            df=test_visits,
            prediction_time=prediction_time,
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
        )

        X_test = add_missing_columns(pipeline, X_test)

        prob_true, prob_pred = calibration_curve(
            y_test, pipeline.predict_proba(X_test)[:, 1], n_bins=10, strategy=strategy
        )

        ax = axs[i]
        hour, minutes = prediction_time

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1,
            label="Predictions",
            color=primary_color,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfectly calibrated",
            color=secondary_color,
        )
        ax.set_title(f"Calibration Plot for {hour}:{minutes:02}", fontsize=14)
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.legend()

    plt.tight_layout()

    calib_plot_path = media_file_path / "calibration_plot"
    calib_plot_path = calib_plot_path.with_suffix(".png")

    plt.savefig(calib_plot_path)
    plt.show()
