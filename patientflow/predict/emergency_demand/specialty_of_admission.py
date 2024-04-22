import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from typing import Any, Dict, Tuple


class SpecialtyPredictor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.weights = None  # Initialize the weights attribute

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        Fits the predictor based on training data by computing the proportion of each
        consult sequence ending in specific observed categories. It also handles null
        sequences and incorporates a default probability for sequences without explicit data.

        Parameters:
        - X: A pandas DataFrame with at least 'consult_sequence', 'final_sequence' and 'observed_specialty' columns.

        Returns:
        - A dictionary mapping each sequence (including null sequences) to their
          respective probability distribution across different categories.
        """

        # derive the names of the observed specialties from the data (used later)
        prop_keys = X.observed_specialty.unique()

        # For each sequences count the number of observed categories
        X_grouped = (
            X.groupby("final_sequence")["observed_specialty"]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Handle null sequences by assigning them to a specific key
        null_counts = (
            X[X["final_sequence"].isnull()]["observed_specialty"]
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

        # Reweight probabilities of ending with each observed specialty
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
        proportions["prob_consult_sequence_ends_in_observed_specialty"] = proportions[
            "final_sequence_to_string"
        ].apply(
            lambda x: self._string_match_consult_sequence(x, proportions, prop_keys)
        )

        # return these as weights
        self.weights = proportions.to_dict()[
            "prob_consult_sequence_ends_in_observed_specialty"
        ]

        return self

    def _string_match_consult_sequence(self, consult_sequence, proportions, prop_keys):
        """
        Matches a given consult_sequence with final sequences (expressed as strings) in the dataset and aggregates
        their probabilities for each observed specialty. This function filters the data to
        match only those rows where the *beginning* of the final_sequence string
        matches the given consult_sequence, ie it requires only a partial match.
        That means that sequence 'medical' will match 'medical, elderly' and 'medical, surgical'
        as well as 'medical' on its own. The function
        computes the total probabilities of any consult_sequence ending in each specialty, and normalizes these totals.

        Parameters:
        - consult_sequence (str): The sequence of consults represented as a string,
        used to match against sequences in the proportions DataFrame.
        - proportions (pd.DataFrame): DataFrame containing proportions data with an additional
        column 'final_sequence_to_string' which includes string representations of sequences.
        - prop_keys (np.array): Array of unique observed specialties to consider in calculations.

        Returns:
        - dict: A dictionary where keys are specialty names and values are the aggregated
        and normalized probabilities of ending a consult sequence in those specialties.
        """

        props = proportions[
            proportions.final_sequence_to_string.str.match(consult_sequence)
        ][prop_keys].sum()
        props_total = props.sum()
        return dict(zip(prop_keys, props / props_total))

    def predict(
        self,
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
        if consult_sequence in self.weights:
            return self.weights[consult_sequence]

        # Otherwise, if the sequence has multiple elements, work back looking for a match
        while len(consult_sequence) > 1:
            consult_sequence_list = list(consult_sequence)
            consult_sequence = tuple(consult_sequence_list[0:-1])  # remove last element

            if consult_sequence in self.weights:
                return self.weights[consult_sequence]

        #   if no consult data:
        return self.weights.get(tuple(), {})
