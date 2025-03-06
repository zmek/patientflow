import numpy as np
import matplotlib.pyplot as plt
from patientflow.load import get_model_name


def plot_features(
    trained_models, media_file_path, prediction_times, model_group_name="admissions"
):
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60
        + x[1],  # Convert (hour, minute) to minutes since midnight
    )

    num_plots = len(prediction_times_sorted)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 6, 12))

    # Handle case of single prediction time
    if num_plots == 1:
        axs = [axs]

    for i, prediction_time in enumerate(prediction_times_sorted):
        # Get model name and pipeline for this prediction time
        model_name = get_model_name(model_group_name, prediction_time)
        pipeline = trained_models[model_name]

        # Get feature names from the pipeline
        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:25] for col in transformed_cols]

        # Get feature importances
        feature_importances = pipeline.named_steps["classifier"].feature_importances_
        indices = np.argsort(feature_importances)[
            -20:
        ]  # Get indices of the top 20 features

        # Plot for this prediction time
        ax = axs[i]
        hour, minutes = prediction_time
        ax.barh(range(len(indices)), feature_importances[indices], align="center")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(np.array(truncated_cols)[indices])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title(f"Feature Importances for {hour}:{minutes:02}")

    plt.tight_layout()

    # Save and display plot
    feature_plot_path = media_file_path / "feature_importance_plots.png"
    plt.savefig(feature_plot_path)
    plt.show()
    plt.close(fig)
