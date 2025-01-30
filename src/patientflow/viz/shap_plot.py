from matplotlib import pyplot as plt
from patientflow.load import get_model_name
from patientflow.prepare import prepare_for_inference

import shap
import scipy.sparse


def plot_shap_individually(
    model_file_path, media_file_path, data_file_path, prediction_times, model_name
):
    for i, _prediction_time in enumerate(prediction_times):
        fig, ax = plt.subplots(figsize=(8, 12))
        X_test, y_test, pipeline = prepare_for_inference(
            model_file_path=model_file_path,
            model_name=model_name,
            prediction_time=_prediction_time,
            data_path=data_file_path,
            single_snapshot_per_visit=False,
            model_only=False,
        )
        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:45] for col in transformed_cols]

        explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
        X_test = pipeline.named_steps["feature_transformer"].transform(X_test)

        # Convert sparse matrix to dense if necessary
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()

        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=truncated_cols,
            #   color=primary_color,
            show=False,
        )

        ax.set_title(f"SHAP Values for Time of Day: {_prediction_time}")
        ax.set_xlabel("SHAP Value")
        plt.tight_layout()

        _model_name = get_model_name(model_name, _prediction_time)
        shap_plot_path = str(media_file_path / "shap_plot_") + _model_name + ".png"

        plt.savefig(shap_plot_path)
        plt.show()
