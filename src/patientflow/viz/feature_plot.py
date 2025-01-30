import numpy as np
import matplotlib.pyplot as plt
from patientflow.load import get_model_name
from patientflow.prepare import prepare_for_inference


def plot_feature_importances_individually(
    model_file_path, media_file_path, data_file_path, prediction_times, model_name
):
    for _prediction_time in prediction_times:
        pipeline = prepare_for_inference(
            model_file_path=model_file_path,
            model_name=model_name,
            prediction_time=_prediction_time,
            data_path=data_file_path,
            single_snapshot_per_visit=False,
            model_only=True,
        )

        transformed_cols = pipeline.named_steps[
            "feature_transformer"
        ].get_feature_names_out()
        transformed_cols = [col.split("__")[-1] for col in transformed_cols]
        truncated_cols = [col[:25] for col in transformed_cols]

        feature_importances = pipeline.named_steps["classifier"].feature_importances_
        indices = np.argsort(feature_importances)[
            -20:
        ]  # Get indices of the top 20 features

        plt.figure(figsize=(6, 12))
        plt.title(f"Feature Importances for Time of Day: {_prediction_time}")
        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), np.array(truncated_cols)[indices])
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()

        _model_name = get_model_name("ed_admission", _prediction_time)
        feature_plot_path = str(media_file_path / "feature_plot_") + _model_name + ".png"

        plt.savefig(feature_plot_path)
        plt.show()
