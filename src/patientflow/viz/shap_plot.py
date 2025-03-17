from matplotlib import pyplot as plt
from patientflow.load import get_model_name
from patientflow.prepare import get_snapshots_at_prediction_time
from patientflow.predict.emergency_demand import add_missing_columns
import shap
import scipy.sparse
import numpy as np


def plot_shap(
    trained_models,
    media_file_path,
    test_visits,
    prediction_times,
    exclude_from_training_data,
    model_group_name,
):
    # Sort prediction times by converting to minutes since midnight
    prediction_times_sorted = sorted(
        prediction_times,
        key=lambda x: x[0] * 60
        + x[1],  # Convert (hour, minute) to minutes since midnight
    )

    for i, prediction_time in enumerate(prediction_times_sorted):
        fig, ax = plt.subplots(figsize=(8, 12))

        # Get model name and pipeline for this prediction time
        model_name = get_model_name(model_group_name, prediction_time)
        pipeline = trained_models[model_name].pipeline

        # Get test data for this prediction time
        X_test, y_test = get_snapshots_at_prediction_time(
            df=test_visits,
            prediction_time=prediction_time,
            exclude_columns=exclude_from_training_data,
            single_snapshot_per_visit=False,
        )

        X_test = add_missing_columns(pipeline, X_test)
        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:45] for col in transformed_cols]

        # Transform features
        X_test = pipeline.named_steps["feature_transformer"].transform(X_test)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])

        # Convert sparse matrix to dense if necessary
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()

        shap_values = explainer.shap_values(X_test)

        # Print prediction distribution
        predictions = pipeline.named_steps["classifier"].predict(X_test)
        print(
            "Predicted classification (not admitted, admitted): ",
            np.bincount(predictions),
        )

        # Print mean SHAP values for each class
        if isinstance(shap_values, list):
            print("SHAP values shape:", [arr.shape for arr in shap_values])
            print("Mean SHAP values (class 0):", np.abs(shap_values[0]).mean(0))
            print("Mean SHAP values (class 1):", np.abs(shap_values[1]).mean(0))

        # Create SHAP summary plot
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=truncated_cols,
            show=False,
        )

        hour, minutes = prediction_time
        ax.set_title(f"SHAP Values for Time of Day: {hour}:{minutes:02}")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()

        # Save plot
        model_name = get_model_name("admissions_minimal", prediction_time)
        shap_plot_path = str(media_file_path / "shap_plot_") + model_name + ".png"

        plt.savefig(shap_plot_path)
        plt.show()
