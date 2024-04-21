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
        - X: A pandas DataFrame with at least 'consult_sequence', 'final_sequence' and 'observed_category' columns.

        Returns:
        - A dictionary mapping each sequence (including null sequences) to their
          respective probability distribution across different categories.
        """

        # derive the names of the observed specialties from the data (used later)
        prop_keys = X.observed_category.unique()

        # For each sequences count the number of observed categories
        X_grouped = (
            X.groupby("final_sequence")["observed_category"]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Handle null sequences by assigning them to a specific key
        null_counts = (
            X[X["final_sequence"].isnull()]["observed_category"]
            .value_counts()
            .to_frame()
            .T
        )
        null_counts.index = [tuple()]

        # Concatenate null sequence handling
        X_grouped = pd.concat([X_grouped, null_counts])

        # Calculate the total number of times each final sequence occurred
        row_totals = X_grouped.sum(axis=1)

        # Calculate for each final sequence, the proportion of ending with each observed specialty
        proportions = X_grouped.div(row_totals, axis=0)

        # Calculate the probability of each final sequence occurring in the original data
        proportions["probability_of_final_sequence"] = row_totals / row_totals.sum()

        # Reweight probabilities of ending with each observed speciality
        # by the likelihood of each final sequence ocurring
        for col in proportions.columns[
            :-1
        ]:  # Avoid the last column which is the 'probability_of_final_sequence'
            proportions[col] *= proportions["probability_of_final_sequence"]

        # convert final sequence to a string in order to conduct string searches on it
        proportions["final_sequence_to_string"] = (
            proportions.reset_index()["index"]
            .apply(lambda x: "-".join(map(str, x)))
            .values
        )
        # row-wise function to return, for each consult sequence,
        # the proportion that end up in each final sequence and thereby
        # the probability of it ending in any observed category
        proportions["prob_consult_sequence_ends_in_observed_category"] = proportions[
            "final_sequence_to_string"
        ].apply(
            lambda x: self._string_match_consult_sequence(x, proportions, prop_keys)
        )

        # return these as weights
        weights = proportions.to_dict()[
            "prob_consult_sequence_ends_in_observed_category"
        ]

        return weights

    def _string_match_consult_sequence(self, consult_sequence, proportions, prop_keys):

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

        # # add entries for paediatric
        # props_keys = np.array(list(prop_keys) + ["paediatric"])
        # props = np.array(list(props) + [0])

        return dict(zip(prop_keys, props))

    def predict(
        self,
        weights: dict[tuple[str, ...], dict[str, float]],
        consult_sequence: tuple[str, ...],
    ) -> dict[str, float]:
        """
        Predicts which specialities an ED patient might be admitted to from their
        previous consults whilst in ED.
        For a specific consult sequence such as ("medical", "surgical") the return
        value will a dict of probabilities of being admitted under certain
        specialities. For example {"medical": 0.3, "surgical": 0.2,
        "haem_onc": 0.1}.
        :param weights: A dictionary of probability weightings for admission
            specialities. See the `_weights` function for further explanation.
        :param consult_sequence: A tuple containing the specialities that have
            reviewed the ED patient in the order they have been requested. An empty
            tuple represents a patient that has not had any reviews.
        :return: A dictionary of specialities and the probabilities that the
            patient will be admitted to them.
        """
        # Return a direct lookup of probabilities if possible.
        if consult_sequence in weights:
            return weights[consult_sequence]

        # Otherwise, if the sequence has multiple elements, work back looking for a match
        while len(consult_sequence) > 1:
            consult_sequence_list = list(consult_sequence)
            consult_sequence = tuple(consult_sequence_list[0:-1])  # remove last element

            if consult_sequence in weights:
                return weights[consult_sequence]

        #   if no consult data:
        return weights[tuple()]
