import matplotlib.pyplot as plt
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.load import get_model_name

# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"


def plot_prediction_distributions(
    prediction_times,
    media_file_path,
    trained_models,
    test_visits,
    exclude_from_training_data,
    model_group_name="admissions",
    model_name_suffix=None,
    bins=30,
):
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60 + x[1],
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

        # Use calibrated pipeline if available, otherwise use regular pipeline
        if (
            hasattr(trained_models[model_name], "calibrated_pipeline")
            and trained_models[model_name].calibrated_pipeline is not None
        ):
            pipeline = trained_models[model_name].calibrated_pipeline
        else:
            pipeline = trained_models[model_name].pipeline

        # Get test data for this prediction time
        X_test, y_test = get_snapshots_at_prediction_time(
            df=test_visits,
            prediction_time=prediction_time,
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
        )

        X_test = add_missing_columns(pipeline, X_test)

        # Get predictions
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Separate predictions for positive and negative cases
        pos_preds = y_pred_proba[y_test == 1]
        neg_preds = y_pred_proba[y_test == 0]

        ax = axs[i]
        hour, minutes = prediction_time

        # Plot distributions
        ax.hist(
            neg_preds,
            bins=bins,
            alpha=0.5,
            color=primary_color,
            density=True,
            label="Negative Cases",
            histtype="step",
            linewidth=2,
        )
        ax.hist(
            pos_preds,
            bins=bins,
            alpha=0.5,
            color=secondary_color,
            density=True,
            label="Positive Cases",
            histtype="step",
            linewidth=2,
        )

        # Optional: Fill with lower opacity
        ax.hist(neg_preds, bins=bins, alpha=0.2, color=primary_color, density=True)
        ax.hist(pos_preds, bins=bins, alpha=0.2, color=secondary_color, density=True)

        ax.set_title(f"Prediction Distribution at {hour}:{minutes:02}", fontsize=14)
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()

    plt.tight_layout()

    dist_plot_path = media_file_path / "distribution_plot"
    dist_plot_path = dist_plot_path.with_suffix(".png")

    plt.savefig(dist_plot_path)
    plt.show()
