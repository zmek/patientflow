from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from patientflow.load import get_model_key

from patientflow.predict.admission_in_prediction_window import (
    calculate_probability,
)

from patientflow.predict.specialty_of_admission import get_specialty_probs

from patientflow.aggregate import (
    model_input_to_pred_proba,
    pred_proba_to_agg_predicted,
)


from patientflow.errors import ModelLoadError
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def add_missing_columns(pipeline, df):
    # check input data for missing columns
    column_transformer = pipeline.named_steps["feature_transformer"]

    # Function to get feature names before one-hot encoding
    def get_feature_names_before_encoding(column_transformer):
        feature_names = []
        for name, transformer, columns in column_transformer.transformers:
            if isinstance(transformer, OneHotEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, OrdinalEncoder):
                feature_names.extend(columns)
            elif isinstance(transformer, StandardScaler):
                feature_names.extend(columns)
            else:
                feature_names.extend(columns)
        return feature_names

    feature_names_before_encoding = get_feature_names_before_encoding(
        column_transformer
    )

    added_columns = []
    for missing_col in set(feature_names_before_encoding).difference(set(df.columns)):
        if missing_col.startswith(("lab_orders_", "visited_", "has_")):
            df[missing_col] = False
        elif missing_col.startswith(("num_", "total_")):
            df[missing_col] = 0
        elif missing_col.startswith("latest_"):
            df[missing_col] = pd.NA
        elif missing_col == "arrival_method":
            df[missing_col] = "None"
        else:
            df[missing_col] = pd.NA
        added_columns.append(missing_col)

    if added_columns:
        print(
            f"Warning: The following columns were used in training, but not found in the real-time data. These have been added to the dataframe: {', '.join(added_columns)}"
        )

    return df


def index_of_sum(sequence: List[float], max_sum: float) -> int:
    """Returns the index where the cumulative sum of a sequence of probabilities exceeds max_sum."""
    cumulative_sum = 0.0
    for i, value in enumerate(sequence):
        cumulative_sum += value
        if cumulative_sum >= 1 - max_sum:
            return i
    return len(sequence) - 1  # Return the last index if the sum doesn't exceed max_sum


def validate_model_names(models: Dict[str, Any], model_names: Dict[str, str]) -> None:
    """
    Validates that all model types specified in model_names exist in models.

    Args:
        models: Dictionary containing all required models
        model_names: Dictionary mapping model types to their names

    Raises:
        ModelLoadError: If a required model name is not found in models
    """
    missing_models = [name for name in model_names.values() if name not in models]
    if missing_models:
        raise ModelLoadError(f"Missing required models: {missing_models}")


def create_predictions(
    models: Dict[str, Any],
    model_names: Dict[str, str],
    prediction_time: Tuple,
    prediction_snapshots: pd.DataFrame,
    specialties: List[str],
    prediction_window_hrs: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cdf_cut_points: List[float],
) -> Dict[str, Dict[str, List[int]]]:
    """
    Create predictions for emergency demand for a single prediction moment.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary containing all required models with the following structure:
        {
            'admissions': {
                'admissions_<prediction_time>': ModelResults,
                # ... one ModelResults object per prediction time
            },
            'specialty': SequencePredictor,
            'yet_to_arrive': WeightedPoissonPredictor
        }
        where ModelResults objects contain either a 'pipeline' or 'calibrated_pipeline' attribute
    model_names : Dict[str, str]
        Dictionary mapping model types to their names, e.g.:
        {
            'admissions': 'admissions',
            'specialty': 'ed_specialty',
            'yet_to_arrive': 'yet_to_arrive_4_hours'
        }
    prediction_time : Tuple
        Hour and minute of time for model inference
    prediction_snapshots : pd.DataFrame
        DataFrame containing prediction snapshots
    specialties : List[str]
        List of specialty names for predictions (e.g., ['surgical', 'medical'])
    prediction_window_hrs : float
        Prediction window in hours
    x1 : float
        X-coordinate of first point for probability curve
    y1 : float
        Y-coordinate of first point for probability curve
    x2 : float
        X-coordinate of second point for probability curve
    y2 : float
        Y-coordinate of second point for probability curve
    cdf_cut_points : List[float]
        List of cumulative distribution function cut points (e.g., [0.9, 0.7])
    special_params : Optional[Dict[str, Any]], optional
        Special handling parameters for categories, by default None

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        Nested dictionary containing predictions for each specialty:
        {
            'specialty_name': {
                'in_ed': [pred1, pred2, ...],
                'yet_to_arrive': [pred1, pred2, ...]
            }
        }

    Notes
    -----
    The admissions models in the models dictionary must be ModelResults objects
    that contain either a 'pipeline' or 'calibrated_pipeline' attribute. The pipeline
    will be used for making predictions, with calibrated_pipeline taking precedence
    if both exist.
    """

    validate_model_names(models, model_names)

    special_params = models[model_names["specialty"]].special_params

    if special_params:
        special_category_func = special_params["special_category_func"]
        special_category_dict = special_params["special_category_dict"]
        special_func_map = special_params["special_func_map"]
    else:
        special_category_func = special_category_dict = special_func_map = None

    predictions: Dict[str, Dict[str, List[int]]] = {
        specialty: {"in_ed": [], "yet_to_arrive": []} for specialty in specialties
    }

    # Get appropriate model for prediction time
    model_for_prediction_time = get_model_key(
        model_names["admissions"], prediction_time
    )

    # Use calibrated pipeline if available, otherwise use regular pipeline
    if (
        hasattr(
            models[model_names["admissions"]][model_for_prediction_time],
            "calibrated_pipeline",
        )
        and models[model_names["admissions"]][
            model_for_prediction_time
        ].calibrated_pipeline
        is not None
    ):
        admissions_model = models[model_names["admissions"]][
            model_for_prediction_time
        ].calibrated_pipeline
    else:
        admissions_model = models[model_names["admissions"]][
            model_for_prediction_time
        ].pipeline

    # Add missing columns expected by the model
    prediction_snapshots = add_missing_columns(admissions_model, prediction_snapshots)

    # Get yet to arrive model
    yet_to_arrive_model = models[model_names["yet_to_arrive"]]

    # Get predictions of admissions for ED patients
    prob_admission_after_ed = model_input_to_pred_proba(
        prediction_snapshots, admissions_model
    )

    # Get predictions of admission to specialty
    prediction_snapshots.loc[:, "specialty_prob"] = get_specialty_probs(
        specialties,
        models[model_names["specialty"]],
        prediction_snapshots,
        special_category_func=special_category_func,
        special_category_dict=special_category_dict,
    )

    prediction_snapshots.loc[:, "elapsed_los_hrs"] = prediction_snapshots[
        "elapsed_los"
    ].apply(lambda x: x / 3600)

    # Get probability of admission within prediction window
    prob_admission_in_window = prediction_snapshots.apply(
        lambda row: calculate_probability(
            row["elapsed_los_hrs"], prediction_window_hrs, x1, y1, x2, y2
        ),
        axis=1,
    )

    if special_func_map is None:
        special_func_map = {"default": lambda row: True}

    for specialty in specialties:
        func = special_func_map.get(specialty, special_func_map["default"])
        non_zero_indices = prediction_snapshots[
            prediction_snapshots.apply(func, axis=1)
        ].index

        filtered_prob_admission_after_ed = prob_admission_after_ed.loc[non_zero_indices]
        prob_admission_to_specialty = prediction_snapshots["specialty_prob"].apply(
            lambda x: x[specialty]
        )

        filtered_prob_admission_to_specialty = prob_admission_to_specialty.loc[
            non_zero_indices
        ]
        filtered_prob_admission_in_window = prob_admission_in_window.loc[
            non_zero_indices
        ]

        filtered_weights = (
            filtered_prob_admission_to_specialty * filtered_prob_admission_in_window
        )

        agg_predicted_in_ed = pred_proba_to_agg_predicted(
            filtered_prob_admission_after_ed, weights=filtered_weights
        )

        prediction_context = {specialty: {"prediction_time": prediction_time}}
        agg_predicted_yta = yet_to_arrive_model.predict(
            prediction_context, x1, y1, x2, y2
        )

        predictions[specialty]["in_ed"] = [
            index_of_sum(agg_predicted_in_ed["agg_proba"].values.cumsum(), cut_point)
            for cut_point in cdf_cut_points
        ]
        predictions[specialty]["yet_to_arrive"] = [
            index_of_sum(
                agg_predicted_yta[specialty]["agg_proba"].values.cumsum(), cut_point
            )
            for cut_point in cdf_cut_points
        ]

    return predictions
