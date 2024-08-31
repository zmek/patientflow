import unittest
import pandas as pd
import numpy as np
import sys
import os
from scipy.stats import poisson


from pathlib import Path
import joblib

# PROJECT_ROOT = Path().home()
# USER_ROOT = Path().home() / 'work'

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../functions"))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/patientflow"))
)

# sys.path.append(str(USER_ROOT / 'patientflow' / 'src' / 'patientflow'))
# sys.path.append(str(USER_ROOT / 'patientflow' / 'functions'))

from predict.realtime_demand import create_predictions
from load import get_model_name
from prepare import create_special_category_objects

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os

from errors import ModelLoadError, MissingKeysError


# Example usage:
# Assuming you have a dataframe `df` with the necessary columns
# df = pd.read_csv('your_data.csv')
# pipeline = create_pipeline(df)

# class AdmissionModel:
#     def __init__(self, df):
#         self.df = df

#     def fit(self, df, weights = None):
#         pass

#     def predict(self, df):
#         return [0.7] * len(df)


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
        df["consultation_sequence"] = [
            np.random.choice(consultations, size=np.random.randint(1, 4)).tolist()
            for _ in range(n)
        ]

    return df


class ProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, weights=None):
        return self.probabilities


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
        self.create_admissions_model()
        self.create_yta_model()
        self.create_spec_model()

    def create_admissions_model(self):
        # Define the feature columns and target
        feature_columns = ["elapsed_los", "sex", "age_on_arrival", "arrival_method"]
        target_column = "is_admitted"

        df = create_random_df()

        # Split the data into features and target
        X = df[feature_columns]
        y = df[target_column]

        # Define the model
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        # column_transformer = create_column_transformer(X)
        column_transformer = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(), ["sex", "arrival_method"]),
                ("passthrough", "passthrough", ["elapsed_los", "age_on_arrival"]),
            ]
        )

        # create a pipeline with the feature transformer and the model
        pipeline = Pipeline(
            [("feature_transformer", column_transformer), ("classifier", model)]
        )

        # Fit the pipeline to the data
        pipeline.fit(X, y)
        # transformed_X = pipeline.named_steps['feature_transformer'].transform(X)

        model_name = get_model_name("ed_admission", self.prediction_time)
        full_path = self.model_file_path / str(model_name + ".joblib")
        joblib.dump(pipeline, full_path)

    def create_spec_model(self):
        probabilities = {
            "surgical": 0.3,
            "haem/onc": 0.1,
            "medical": 0.6,
            "paediatric": 0.0,
        }

        model = ProbabilityModel(probabilities)

        full_path = self.model_file_path / str("ed_specialty.joblib")
        joblib.dump(model, full_path)

    def create_yta_model(self, prediction_window_hrs=None):
        if prediction_window_hrs is None:
            prediction_window_hrs = self.prediction_window_hrs

        lambdas = {"medical": 5, "paediatric": 3, "surgical": 2, "haem/onc": 1}
        model = PoissonModel(lambdas)

        full_path = (
            self.model_file_path
            / f"ed_yet_to_arrive_by_spec_{str(int(prediction_window_hrs))}_hours.joblib"
        )
        joblib.dump(model, full_path)
        lambdas = {"medical": 5, "paediatric": 3, "surgical": 2, "haem/onc": 1}
        model = PoissonModel(lambdas)

        full_path = (
            self.model_file_path
            / f"ed_yet_to_arrive_by_spec_{str(int(self.prediction_window_hrs))}_hours.joblib"
        )
        joblib.dump(model, full_path)

    def test_basic_functionality(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            special_params=None,
        )

        # print(predictions)

        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        self.assertEqual(predictions["paediatric"]["in_ed"], [0, 0])
        self.assertEqual(predictions["medical"]["in_ed"], [11, 10])

    def test_basic_functionality_with_special_category(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        special_params = create_special_category_objects(uclh=True)

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            special_params=special_params,
        )

        # print(predictions)

        self.assertIsInstance(predictions, dict)
        self.assertIn("paediatric", predictions)
        self.assertIn("medical", predictions)
        self.assertIn("in_ed", predictions["paediatric"])
        self.assertIn("yet_to_arrive", predictions["paediatric"])

        self.assertEqual(predictions["paediatric"]["in_ed"], [1, 1])
        self.assertEqual(predictions["medical"]["in_ed"], [10, 9])

    def test_incorrect_special_params(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)

        with self.assertRaises(MissingKeysError):
            create_predictions(
                model_file_path=self.model_file_path,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window_hrs=self.prediction_window_hrs,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
                special_params={"dict"},
            )

    def test_empty_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=0, include_consults=True)

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            special_params=None,
        )

        self.assertIsInstance(predictions, dict)
        for specialty in self.specialties:
            self.assertEqual(predictions[specialty]["in_ed"], [0, 0])

    def test_single_row_prediction_snapshots(self):
        prediction_snapshots = create_random_df(n=1, include_consults=True)

        predictions = create_predictions(
            model_file_path=self.model_file_path,
            prediction_time=self.prediction_time,
            prediction_snapshots=prediction_snapshots,
            specialties=self.specialties,
            prediction_window_hrs=self.prediction_window_hrs,
            cdf_cut_points=self.cdf_cut_points,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            special_params=None,
        )

        self.assertIsInstance(predictions, dict)
        for specialty in self.specialties:
            self.assertEqual(predictions[specialty]["in_ed"], [0, 0])

    def test_model_not_found(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)
        non_existing_window_hrs = 10.0

        with self.assertRaises(ModelLoadError):
            create_predictions(
                model_file_path=self.model_file_path,
                prediction_time=self.prediction_time,
                prediction_snapshots=prediction_snapshots,
                specialties=self.specialties,
                prediction_window_hrs=non_existing_window_hrs,
                cdf_cut_points=self.cdf_cut_points,
                x1=self.x1,
                y1=self.y1,
                x2=self.x2,
                y2=self.y2,
                special_params=None,
            )

    def test_prediction_window_extremes(self):
        prediction_snapshots = create_random_df(n=50, include_consults=True)

        short_window_hrs = 0.1
        long_window_hrs = 100.0

        self.create_yta_model(prediction_window_hrs=short_window_hrs)
        self.create_yta_model(prediction_window_hrs=long_window_hrs)

        short_window_predictions = create_predictions(
            model_file_path=self.model_file_path,
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

        long_window_predictions = create_predictions(
            model_file_path=self.model_file_path,
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

    # def test_large_dataset_performance(self):
    # prediction_snapshots = create_random_df(n = 10000, include_consults = True)

    #     predictions = create_predictions(
    #         model_file_path=self.model_file_path,
    #         prediction_time=self.prediction_time,
    #         prediction_snapshots=prediction_snapshots,
    #         specialties=self.specialties,
    #         prediction_window_hrs=self.prediction_window_hrs,
    #         cdf_cut_points=self.cdf_cut_points,
    #         x1=self.x1,
    #         y1=self.y1,
    #         x2=self.x2,
    #         y2=self.y2,
    #     )

    #     self.assertIsInstance(predictions, dict)


if __name__ == "__main__":
    unittest.main()
