import unittest
import pandas as pd
import numpy as np
import os
from scipy.stats import poisson

from pathlib import Path

from patientflow.predict.emergency_demand import create_predictions
from patientflow.load import get_model_key
from patientflow.train.emergency_demand import ModelResults

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from patientflow.errors import ModelLoadError
from patientflow.predictors.sequence_predictor import SequencePredictor


def create_random_df(n=1000, include_consults=False):
    # Generate random data
    np.random.seed(0)
    age_on_arrival = np.random.randint(1, 100, size=n)
    elapsed_los = np.random.randint(0, 3 * 24 * 3600, size=n)
    arrival_method = np.random.choice(
        ["ambulance", "public_transport", "walk-in"], size=n
    )
    sex = np.random.choice(["M", "F"], size=n)
    is_admitted = np.random.choice([0, 1], size=n)

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "age_on_arrival": age_on_arrival,
            "elapsed_los": elapsed_los,
            "arrival_method": arrival_method,
            "sex": sex,
            "is_admitted": is_admitted,
        }
    )

    if include_consults:
        # Generate random consultation sequence
        consultations = ["medical", "surgical", "haem/onc", "paediatric"]
        df["final_sequence"] = [
            [
                str(x)
                for x in np.random.choice(consultations, size=np.random.randint(1, 4))
            ]
            for _ in range(n)
        ]
        # Create consultation_sequence by removing random number of items from final_sequence
        df["consultation_sequence"] = [
            seq[: -np.random.randint(0, len(seq))] if len(seq) > 0 else []
            for seq in df["final_sequence"]
        ]
        # Set specialty as a random item from final_sequence, ensuring we always have a valid string
        df["specialty"] = [
            str(np.random.choice(seq))
            if len(seq) > 0
            else consultations[0]  # default to first consultation type
            for seq in df["final_sequence"]
        ]

    return df


def create_admissions_model(prediction_time):
    """Create a test admissions model with ModelResults structure.

    Parameters
    ----------
    prediction_time : float
        The prediction time point to create the model for

    Returns
    -------
    tuple
        (ModelResults object, model_name string)
    """
    # Define the feature columns and target
    feature_columns = ["elapsed_los", "sex", "age_on_arrival", "arrival_method"]
    target_column = "is_admitted"

    df = create_random_df(include_consults=True, n=10)

    # Split the data into features and target
    X = df[feature_columns]
    y = df[target_column]

    # Define the model
    model = XGBClassifier(eval_metric="logloss")
    column_transformer = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                ["sex", "arrival_method"],
            ),
            ("passthrough", "passthrough", ["elapsed_los", "age_on_arrival"]),
        ]
    )

    # Create a pipeline with the feature transformer and the model
    pipeline = Pipeline(
        [("feature_transformer", column_transformer), ("classifier", model)]
    )

    # Fit the pipeline to the data
    pipeline.fit(X, y)

    # Create ModelResults object
    model_results = ModelResults(
        pipeline=pipeline,
        valid_logloss=0.5,  # Mock value for testing
        feature_names=feature_columns,
        feature_importances=[0.25] * len(feature_columns),  # Mock values
        metrics={
            "params": "test_params",
            "train_valid_set_results": {},
            "test_set_results": {},
        },
        calibrated_pipeline=None,  # No calibration for test
    )

    model_name = get_model_key("admissions", prediction_time)
    return (model_results, model_name, df)


def create_spec_model(df, apply_special_category_filtering):
    model = SequencePredictor(
        input_var="consultation_sequence",  # Column containing input sequences
        grouping_var="final_sequence",  # Column containing grouping sequences
        outcome_var="specialty",  # Column containing outcome categories
        apply_special_category_filtering=apply_special_category_filtering,
        admit_col="is_admitted",
    )
    model.fit(df)
    # full_path = self.model_file_path / str("ed_specialty.joblib")
    # joblib.dump(model, full_path)

    return model


def create_yta_model(prediction_window_hrs):
    if prediction_window_hrs is None:
        prediction_window_hrs = prediction_window_hrs

    lambdas = {"medical": 5, "paediatric": 3, "surgical": 2, "haem/onc": 1}
    model = PoissonModel(lambdas)

    # full_path = (
    #     self.model_file_path
    #     / f"ed_yet_to_arrive_by_spec_{str(int(self.prediction_window_hrs))}_hours.joblib"
    # )
    # joblib.dump(model, full_path)
    model_name = f"ed_yet_to_arrive_by_spec_{str(int(prediction_window_hrs))}_hours"
    return (model, model_name)


class PoissonModel:
    def __init__(self, lambdas):
        self.lambdas = lambdas

    def predict(self, prediction_context=None, x1=None, y1=None, x2=None, y2=None):
        result = {}
        for spec, lam in self.lambdas.items():
            # Generate Poisson distribution
            x = np.arange(0, 20)
            poisson_dist = poisson.pmf(x, lam)

            # Create DataFrame
            df = pd.DataFrame(poisson_dist, columns=["agg_proba"])
            df["sum"] = df.index

            # Set 'sum' as the index
            df.set_index("sum", inplace=True)

            result[spec] = df

        return result


class TestCreatePredictions(unittest.TestCase):
    def setUp(self):
        self.model_file_path = Path("tmp")
        os.makedirs(self.model_file_path, exist_ok=True)
        self.prediction_time = (7, 0)
        self.prediction_window_hrs = 8.0
        self.x1, self.y1, self.x2, self.y2 = 4.0, 0.76, 12.0, 0.99
        self.cdf_cut_points = [0.7, 0.9]
        self.specialties = ["paediatric", "surgical", "haem/onc", "medical"]
        self.model_names = {
            "admissions": "admissions",
            "specialty": "ed_specialty",
            "yet_to_arrive": "ed_yet_to_arrive_by_spec_",
        }

        # Create and save models
        admissions_model, admissions_name, self.df = create_admissions_model(
            self.prediction_time
        )
        # full_path = self.model_file_path / f"{admissions_name}.joblib"
        # joblib.dump(admissions_model, full_path)
        self.models = {
            self.model_names["admissions"]: {admissions_name: admissions_model},
            self.model_names["specialty"]: create_spec_model(
                self.df, apply_special_category_filtering=False
            ),
        }
        yta_model, yta_name = create_yta_model(self.prediction_window_hrs)
        # full_path = self.model_file_path / f"{yta_name}.joblib"
        # joblib.dump(yta_model, full_path)
        self.models[self.model_names["yet_to_arrive"]] = yta_model

    def test_basic_functionality(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)

        predictions = create_predictions(
            models=self.models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        self.assertEqual(predictions["paediatric"]["in_ed"], [7, 6])
        self.assertEqual(predictions["medical"]["in_ed"], [7, 6])

    def test_basic_functionality_with_special_category(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)

        # print("\nWithout special category")
        # print(self.df[self.df.is_admitted == 1])
        # print(self.models[self.model_names["specialty"]].weights)

        predictions_without_special_category = create_predictions(
            models=self.models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        # print(predictions_without_special_category)

        # print("\nWith special category")
        models = self.models.copy()
        models[self.model_names["specialty"]] = create_spec_model(
            self.df, apply_special_category_filtering=True
        )
        # print(models[self.model_names["specialty"]].weights)

        predictions_with_special_category = create_predictions(
            models=models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        # print(predictions_with_special_category)
        self.assertIn("paediatric", predictions_without_special_category)
        self.assertIn("paediatric", predictions_with_special_category)

        self.assertEqual(
            predictions_without_special_category["paediatric"]["in_ed"], [1, 0]
        )
        self.assertEqual(
            predictions_with_special_category["paediatric"]["in_ed"], [0, 0]
        )

    def test_single_row_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=1, include_consults=True)

        predictions = create_predictions(
            models=self.models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(predictions, dict)
        for specialty in self.specialties:
            self.assertEqual(predictions[specialty]["in_ed"], [0, 0])

    def test_model_not_found(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        non_existing_window_hrs = 10.0

        # save yta model
        yta_model = self.models[self.model_names["yet_to_arrive"]]

        # copy models and remove the yta to arrive entry
        models = self.models.copy()
        del models[self.model_names["yet_to_arrive"]]
        models["ed_yet_to_arrive_"] = yta_model

        with self.assertRaises(ModelLoadError):
            create_predictions(
                models=models,
                model_names=self.model_names,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window_hrs=non_existing_window_hrs,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
            )

    def test_prediction_window_extremes(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        models = self.models.copy()  # Create copy

        short_window_hrs = 0.1
        long_window_hrs = 100.0

        short_yta_model, _ = create_yta_model(short_window_hrs)
        models[self.model_names["yet_to_arrive"]] = short_yta_model

        short_window_predictions = create_predictions(
            models=models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=short_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        long_yta_model, _ = create_yta_model(long_window_hrs)
        models[self.model_names["yet_to_arrive"]] = long_yta_model

        long_window_predictions = create_predictions(
            models=models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=long_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIsInstance(short_window_predictions, dict)
        self.assertIsInstance(long_window_predictions, dict)

    def test_missing_key_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=5, include_consults=True)
        models = self.models.copy()  # Create copy

        models[self.model_names["specialty"]] = create_spec_model(
            self.df[~self.df.final_sequence.apply(lambda x: "paediatric" in x)],
            apply_special_category_filtering=False,
        )

        predictions = create_predictions(
            models=models,
            model_names=self.model_names,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )

        self.assertIn("paediatric", predictions)
        self.assertNotIn(
            "paediatric", models[self.model_names["specialty"]].weights.keys()
        )


#     # def test_large_dataset_performance(self):
#     #     prediction_snapshots = create_random_df(n = 10000, include_consults = True)

#     #     predictions = create_predictions(
#     #         models=self.models,
#     #         model_names=self.model_names,
#     #         prediction_time=self.prediction_time,
#     #         prediction_snapshots=prediction_snapshots,
#     #         specialties=self.specialties,
#     #         prediction_window_hrs=self.prediction_window_hrs,
#     #         cdf_cut_points=self.cdf_cut_points,
#     #         x1=self.x1,
#     #         y1=self.y1,
#     #         x2=self.x2,
#     #         y2=self.y2,
#     #     )

#     #     self.assertIsInstance(predictions, dict)


if __name__ == "__main__":
    unittest.main()
