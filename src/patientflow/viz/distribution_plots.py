import matplotlib.pyplot as plt
from patientflow.predict.emergency_demand import add_missing_columns
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.load import get_model_key, load_saved_model
from patientflow.model_artifacts import TrainedClassifier

# Define the color scheme
primary_color = "#1f77b4"
secondary_color = "#ff7f0e"


def plot_prediction_distributions(
    trained_models: list[TrainedClassifier],
    test_visits,
    exclude_from_training_data,
    bins=30,
    media_file_path: str= None

):
    """
    Plot prediction distributions for multiple models.

    Args:
        trained_models: List of TrainedClassifier objects
        test_visits: DataFrame containing test visit data
        exclude_from_training_data: Columns to exclude from the test data
        bins: Number of bins for the histogram (default: 30)
        media_file_path: Path to save the plot (default: None)
    """
    if media_file_path is None:
        raise ValueError("media_file_path must be provided")

    # Sort trained_models by prediction time
    trained_models_sorted = sorted(
        trained_models,
        key=lambda x: x.training_results.prediction_time[0] * 60 + x.training_results.prediction_time[1],
    )
    num_plots = len(trained_models_sorted)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 4))

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    for i, trained_model in enumerate(trained_models_sorted):
        # Use calibrated pipeline if available, otherwise use regular pipeline
        if (
            hasattr(trained_model, "calibrated_pipeline")
            and trained_model.calibrated_pipeline is not None
        ):
            pipeline = trained_model.calibrated_pipeline
        else:
            pipeline = trained_model.pipeline

        prediction_time = trained_model.training_results.prediction_time

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
