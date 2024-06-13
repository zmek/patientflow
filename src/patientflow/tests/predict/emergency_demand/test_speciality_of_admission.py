import unittest

import numpy as np
import pandas as pd
from predict.emergency_demand.specialty_of_admission import (
    SpecialityPredictor,
)
from sklearn.model_selection import train_test_split


class TestSpecialityPredictor(unittest.TestCase):
    def setUp(self):
        # Sample data generation as provided
        np.random.seed(42)  # For reproducibility
        data = {
            "training_validation_test": np.random.choice(
                ["train", "valid", "test"], size=100, p=[0.8, 0.1, 0.1]
            ),
            "episode_slice_id": range(1, 101),
            "visit_number": np.random.randint(10000, 99999, size=100),
            "consultation_sequence": np.random.choice(
                [
                    tuple(["haem_onc"]),
                    tuple(["medical", "medical"]),
                    tuple(["medical"]),
                    tuple(["mental_health", "medical"]),
                    tuple(["mental_health"]),
                    tuple(["neuro"]),
                    tuple(["obs_gyn"]),
                    tuple(["surgical", "medical"]),
                    tuple(["surgical", "surgical"]),
                    tuple(["surgical"]),
                    tuple(["urology"]),
                    None,
                ],
                size=100,
                p=[
                    0.05 - 0.0195,
                    0.02,
                    0.4,
                    0.005,
                    0.0075,
                    0.005,
                    0.04,
                    0.005,
                    0.007,
                    0.35,
                    0.035,
                    0.095,
                ],
            ),
            "final_sequence": np.random.choice(
                [
                    tuple(["haem_onc"]),
                    tuple(["medical", "elderly"]),
                    tuple(["medical", "medical"]),
                    tuple(["medical", "mental_health"]),
                    tuple(["medical"]),
                    tuple(["mental_health", "medical"]),
                    tuple(["obs_gyn"]),
                    tuple(["surgical", "medical"]),
                    tuple(["surgical", "surgical"]),
                    tuple(["surgical"]),
                    tuple(["urology"]),
                    None,
                ],
                size=100,
                p=[
                    0.05 - 0.04,
                    0.008,
                    0.03,
                    0.006,
                    0.4,
                    0.008,
                    0.04,
                    0.01,
                    0.008,
                    0.35,
                    0.035,
                    0.095,
                ],
            ),
            "observed_category": np.random.choice(
                ["haem_onc", "medical", "surgical"], size=100, p=[0.1, 0.7, 0.2]
            ),
        }

        df = pd.DataFrame(data)
        self.train_df, self.test_df = train_test_split(
            df[df["training_validation_test"] == "train"],
            test_size=0.2,
            random_state=42,
        )
        self.predictor = SpecialityPredictor()

    def test_train_returns_weights_as_dict(self):
        self.predictor.fit(self.train_df)
        self.assertIsInstance(self.predictor.weights, dict)

    def test_weights_for_urology(self):
        urology_pred = self.train_df.loc[
            self.train_df.final_sequence == tuple(["urology"]), "observed_category"
        ].value_counts() / len(
            self.train_df[self.train_df.final_sequence == tuple(["urology"])]
        )
        self.predictor.fit(self.train_df)
        self.assertTrue(
            self.predictor.weights[tuple(["urology"])]["medical"]
            == urology_pred.values[0]
        )

    def test_predict_returns_dict(self):
        # Assuming fit has been run and self.predictor is ready
        self.predictor.fit(self.train_df)
        for _, row in self.test_df.iterrows():
            consult_sequence = row["consultation_sequence"]
            weights = self.predictor.predict(consult_sequence)
            self.assertIsInstance(
                weights, dict
            )  # Each prediction should be a dictionary of probabilities

    def test_probability_distribution(self):
        self.predictor.fit(self.train_df)
        for _, row in self.test_df.iterrows():
            predicted_probabilities = self.predictor.predict(
                row["consultation_sequence"]
            )
            total_probability = sum(predicted_probabilities.values())
            self.assertAlmostEqual(total_probability, 1.0, places=2)

    def test_edge_cases(self):
        self.predictor.fit(self.train_df)
        empty_sequence = tuple()
        rare_sequence = tuple(["neuro", "neuro", "neuro"])  # Assuming 'neuro' is rare

        empty_prediction = self.predictor.predict(empty_sequence)
        rare_prediction = self.predictor.predict(rare_sequence)

        # print(empty_prediction)

        # Ensure that the method handles empty sequences without errors
        self.assertIsInstance(empty_prediction, dict)
        # Ensure probabilities are returned even for rare sequences
        self.assertTrue(all(isinstance(v, float) for v in rare_prediction.values()))


if __name__ == "__main__":
    unittest.main()
