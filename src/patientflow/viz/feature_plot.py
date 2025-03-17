import numpy as np
import matplotlib.pyplot as plt
from patientflow.load import get_model_key, load_saved_model


def plot_features(
    trained_models,
    media_file_path,
    prediction_times,
    model_group_name="admissions",
    model_name_suffix=None,
    model_file_path=None,
):
    # Load models if not provided
    if trained_models is None:
        if model_file_path is None:
            raise ValueError(
                "model_file_path must be provided if trained_models is None"
            )
        trained_models = {}
        for prediction_time in prediction_times:
            model_name = get_model_key(model_group_name, prediction_time)
            if model_name_suffix:
                model_name = f"{model_name}_{model_name_suffix}"
            trained_models[model_name] = load_saved_model(
                model_file_path, model_group_name, prediction_time
            )

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
        model_name = get_model_key(model_group_name, prediction_time)

        # Always use regular pipeline
        pipeline = trained_models[model_name].pipeline

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
