import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from typing import Any, Dict, Tuple


class SpecialityPredictor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        Fits the predictor based on training data by computing the proportion of each
        consult sequence ending in specific observed categories. It also handles null
        sequences and incorporates a default probability for sequences without explicit data.

        Parameters:
        - X: A pandas DataFrame with at least 'final_sequence' and 'observed_category' columns.

        Returns:
        - A dictionary mapping each sequence (including null sequences) to their
          respective probability distribution across different categories.
        """

        # For each sequences count the number of observed categories
        X_grouped = (
            X.groupby("final_sequence")["observed_category"]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Handling null sequences by assigning them to a specific key
        null_counts = (
            X[X["final_sequence"].isnull()]["observed_category"]
            .value_counts()
            .to_frame()
            .T
        )
        null_counts.index = [
            tuple(
                [
                    "null",
                ]
            )
        ]  # Using 'null' as the key for null sequences as this will allow string searches later

        # Concatenate null sequence handling
        X_grouped = pd.concat([X_grouped, null_counts])

        # Calculate the row totals and proportions
        row_totals = X_grouped.sum(axis=1)
        proportions = X_grouped.div(row_totals, axis=0)

        # derive the names of the observed specialties from the data (used later)
        prop_keys = proportions.columns.values

        # Calculate the probability of each final sequence occurring
        proportions["probability_of_final_sequence"] = row_totals / row_totals.sum()

        # Reweight probabilities by the likelihood of each sequence
        for col in proportions.columns[
            :-1
        ]:  # Avoid the last column which is the 'probability_of_final_sequence'
            proportions[col] *= proportions["probability_of_final_sequence"]

        # convert consultation sequence to a string in order to conduct string searches on it
        proportions["final_sequence_to_string"] = (
            proportions.reset_index()["index"]
            .apply(lambda x: "-".join(map(str, x)))
            .values
        )
        # row-wise function to return, for each final sequence, the proportion that end up in each observed category
        proportions["final_cat_props"] = proportions["final_sequence_to_string"].apply(
            lambda x: self._prop_final_cat_by_consult_sequence(
                x, proportions, prop_keys
            )
        )

        sequence_probabilities = proportions.to_dict()["final_cat_props"]

        # replace the null value in the dict key with None
        sequence_probabilities[tuple()] = sequence_probabilities[
            tuple(
                [
                    "null",
                ]
            )
        ]
        del sequence_probabilities[
            tuple(
                [
                    "null",
                ]
            )
        ]

        return sequence_probabilities

    def _prop_final_cat_by_consult_sequence(
        self, consult_sequence, proportions, prop_keys
    ):

        # get the proportions of visits ending in each speciality that partially match the consult sequence
        props = (
            proportions[
                proportions.final_sequence_to_string.str.match(consult_sequence)
            ][prop_keys].sum()
            / proportions[
                proportions.final_sequence_to_string.str.match(consult_sequence)
            ][prop_keys]
            .sum()
            .values.sum()
        )

        # add entries for paediatric
        props_keys = np.array(list(prop_keys) + ["paediatric"])
        props = np.array(list(props) + [0])

        return dict(zip(props_keys, props))
