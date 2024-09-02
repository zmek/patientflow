from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SequencePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, input_var, grouping_var, outcome_var):
        self.input_var = input_var  # Column name for the input sequence
        self.grouping_var = grouping_var  # Column name for the grouping sequence
        self.outcome_var = outcome_var  # Column name for the outcome category
        self.weights = None  # Initialize the weights attribute to store model weights

    def fit(self, X: pd.DataFrame) -> Dict:
        """
        Fits the predictor based on training data by computing the proportion of each
        input variable sequence ending in specific outcome variable categories. It also handles null
        sequences and incorporates a default probability for sequences without explicit data.

        Parameters
        - X: A pandas DataFrame with at least the columns specified by input_var, grouping_var, and outcome_var.
        - input_var: The name of the column representing the input sequence.
        - grouping_var: The name of the column representing the grouping sequence.
        - outcome_var: The name of the column representing the outcome variable.

        Returns
        - A dictionary mapping each sequence (including null sequences) to their
        respective probability distribution across different categories.

        """
        # derive the names of the observed specialties from the data (used later)
        prop_keys = X[self.outcome_var].unique()

        # For each sequences count the number of observed categories
        X_grouped = (
            X.groupby(self.grouping_var)[self.outcome_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Handle null sequences by assigning them to a specific key
        null_counts = (
            X[X[self.grouping_var].isnull()][self.outcome_var]
            .value_counts()
            .to_frame()
            .T
        )
        null_counts.index = [tuple()]

        # Concatenate null sequence handling
        X_grouped = pd.concat([X_grouped, null_counts])

        # Calculate the total number of times each grouping sequence occurred
        row_totals = X_grouped.sum(axis=1)

        # Calculate for each grouping sequence, the proportion of ending with each observed specialty
        proportions = X_grouped.div(row_totals, axis=0)

        # Calculate the probability of each grouping sequence occurring in the original data
        proportions["probability_of_grouping_sequence"] = row_totals / row_totals.sum()

        # Reweight probabilities of ending with each observed specialty
        # by the likelihood of each grouping sequence occurring
        for col in proportions.columns[
            :-1
        ]:  # Avoid the last column which is the 'probability_of_grouping_sequence'
            proportions[col] *= proportions["probability_of_grouping_sequence"]

        # Convert final sequence to a string in order to conduct string searches on it
        proportions["grouping_sequence_to_string"] = (
            proportions.reset_index()["index"]
            .apply(lambda x: "-".join(map(str, x)))
            .values
        )
        # Row-wise function to return, for each input sequence,
        # the proportion that end up in each final sequence and thereby
        # the probability of it ending in any observed category
        proportions["prob_input_var_ends_in_observed_specialty"] = proportions[
            "grouping_sequence_to_string"
        ].apply(lambda x: self._string_match_input_var(x, proportions, prop_keys))

        # save these as weights
        self.weights = proportions.to_dict()[
            "prob_input_var_ends_in_observed_specialty"
        ]

        # save the input to grouping probabilities
        self.input_to_grouping_probs = self._probability_of_input_to_grouping_sequence(
            X
        )

        return self

    def _string_match_input_var(self, input_var_string, proportions, prop_keys):
        """
        Matches a given input sequence string with grouped sequences (expressed as strings) in the dataset and aggregates
        their probabilities for each outcome category. This function filters the data to
        match only those rows where the *beginning* of the grouped sequence string
        matches the given input sequence string, allowing for partial matches.
        For instance, the sequence 'medical' will match 'medical, elderly' and 'medical, surgical'
        as well as 'medical' on its own. It computes the total probabilities of any input sequence ending
        in each outcome category, and normalizes these totals if possible.

        Parameters
        - input_var_string (str): The sequence of inputs represented as a string,
        used to match against sequences in the proportions DataFrame.
        - proportions (pd.DataFrame): DataFrame containing proportions data with an additional
        column 'grouping_sequence_to_string' which includes string representations of sequences.
        - prop_keys (np.array): Array of unique outcomes to consider in calculations.

        Returns
        - dict: A dictionary where keys are outcome names and values are the aggregated
        and normalized probabilities of an input sequence ending in those outcomes.

        """
        # Filter rows where the grouped sequence string starts with the input sequence string
        props = proportions[
            proportions["grouping_sequence_to_string"].str.match("^" + input_var_string)
        ][prop_keys].sum()

        # Sum of all probabilities to normalize them
        props_total = props.sum()

        # Handle cases where the total probability is zero to avoid division by zero
        if props_total > 0:
            normalized_props = props / props_total
        else:
            normalized_props = (
                props * 0
            )  # Returns zero probabilities if no matches found

        return dict(zip(prop_keys, normalized_props))

    def _probability_of_input_to_grouping_sequence(self, X):
        # For each input sequence count the number of grouping sequences
        X_grouped = (
            X.groupby(self.input_var)[self.grouping_var]
            .value_counts()
            .unstack(fill_value=0)
        )

        # # Calculate the total number of times each input sequence occurred
        row_totals = X_grouped.sum(axis=1)

        # # Calculate for each grouping sequence, the proportion of ending with each grouping sequence
        proportions = X_grouped.div(row_totals, axis=0)

        # # Calculate the probability of each input sequence occurring in the original data
        proportions["probability_of_grouping_sequence"] = row_totals / row_totals.sum()

        return proportions

    def predict(self, input_sequence: tuple[str, ...]) -> Dict[str, float]:
        """
        Predicts the probabilities of ending in various outcome categories for a given input sequence.
        For example, for an input sequence such as ("cardiology", "orthopedics"), the return
        value will be a dict of probabilities such as {"cardiology": 0.3, "orthopedics": 0.2, "neurology": 0.1}.

        Parameters
        - input_sequence: A tuple containing the categories that have been observed for an entity in the order
        they have been encountered. An empty tuple represents an entity with no observed categories.

        Returns
        - A dictionary of categories and the probabilities that the input sequence will end in them.

        """
        # Check for no tuple
        if input_sequence is None or pd.isna(input_sequence):
            return self.weights.get(tuple(), {})

        # Return a direct lookup of probabilities if possible.
        if input_sequence in self.weights:
            return self.weights[input_sequence]

        # Otherwise, if the sequence has multiple elements, work back looking for a match
        while len(input_sequence) > 1:
            input_sequence_list = list(input_sequence)
            input_sequence = tuple(input_sequence_list[:-1])  # remove last element

            if input_sequence in self.weights:
                return self.weights[input_sequence]

        # If no relevant data is found:
        return self.weights.get(tuple(), {})
