"""
Poisson-Binomial Admission Predictor

This module implements a Poisson-Binomial admission predictor to estimate the number of hospital admissions within a specified prediction window using historical admission data. It applies Poisson and binomial distributions to forecast future admissions, excluding already arrived patients. The predictor accommodates different data filters for tailored predictions across various hospital settings.

Dependencies:
    - pandas: For data manipulation and analysis, essential for handling the dataset used in predictions.
    - datetime: For manipulating date and time objects, crucial for time-based predictions.
    - sklearn: Utilizes BaseEstimator and TransformerMixin from scikit-learn for creating custom, interoperable predictors.
    - Custom modules:
        - predict.emergency_demand.yet_to_arrive: Includes the Poisson-binomial generating function for prediction calculations.
        - predict.emergency_demand.admission_in_prediction_window: Calculates the probability of admission within a specified prediction window.
        - time_varying_arrival_rates: Computes time-varying arrival rates, a core component of the prediction algorithm.

Classes:
    PoissonBinomialPredictor(BaseEstimator, TransformerMixin):
        Predicts the number of admissions within a given prediction window based on historical data and Poisson-binomial distribution.

    Methods within PoissonBinomialPredictor:
        - __init__(self, filters=None): Initializes the predictor with optional data filters.
        - filter_dataframe(self, df, filters): Applies filters to the dataset for targeted predictions.
        - fit(self, train_df, prediction_window, time_interval, prediction_times, json_file_path, reference_year, y=None): Trains the predictor using historical data and various parameters.
        - predict(self, prediction_context): Predicts the number of admissions using the trained model.

This module is designed for flexibility and customization to suit different prediction needs in hospital environments.
Author: Zella King
Date: 06.04.24
Version: 1.0
"""

import warnings
from datetime import datetime, timedelta

import numpy as np

import pandas as pd
from typing import Dict, List, Optional


# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from predict.emergency_demand.admission_in_prediction_window_using_aspirational_curve import (
    get_y_from_aspirational_curve,
)

# from dissemination.patientflow.predict.emergency_demand.admission_in_prediction_window import (
from predict.emergency_demand.time_varying_arrival_rates import (
    calculate_rates,
)

# from dissemination.patientflow.predict.emergency_demand.yet_to_arrive import (
from predict.emergency_demand.yet_to_arrive import (
    poisson_binom_generating_function,
)

# Import utility functions for time adjustment
# from edmodel.utils.time_utils import adjust_for_model_specific_times
# Import sklearn base classes for custom transformer creation
from sklearn.base import BaseEstimator, TransformerMixin


class PoissonBinomialPredictor(BaseEstimator, TransformerMixin):
    """
    A class to predict an aspirational number of admissions within a specified prediction window.
    This prediction does not include patients who have already arrived and is based on historical data.
    The prediction uses a combination of Poisson and binomial distributions.

    Attributes
        None

    Methods
        __init__(self, filters=None): Initializes the predictor with optional filters for data categorization.
        filter_dataframe(self, df, filters): Filters the dataset based on specified criteria for targeted predictions.
        fit(self, train_df, prediction_window, time_interval, prediction_times, json_file_path, reference_year, y=None): Trains the model using historical data and prediction parameters.
        predict(self, prediction_context): Predicts the number of admissions for a given context after the model is trained.
        get_weights(self): Retrieves the model parameters computed during fitting.

    """

    def __init__(self, filters=None):
        """
        Initialize the PoissonBinomialPredictor with optional filters.

        Args:
            filters (dict, optional): A dictionary defining filters for different categories or specialties.
                                      If None or empty, no filtering will be applied.

        """
        self.filters = filters if filters else {}

    def filter_dataframe(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Apply a set of filters to a dataframe.

        Args:
            df (pandas.DataFrame): The DataFrame to filter.
            filters (dict): A dictionary where keys are column names and values are the criteria or function to filter by.

        Returns:
            pandas.DataFrame: A filtered DataFrame.

        """
        filtered_df = df
        for column, criteria in filters.items():
            if callable(criteria):  # If the criteria is a function, apply it directly
                filtered_df = filtered_df[filtered_df[column].apply(criteria)]
            else:  # Otherwise, assume the criteria is a value or list of values for equality check
                filtered_df = filtered_df[filtered_df[column] == criteria]
        return filtered_df

    def _calculate_parameters(
        self, df, prediction_window, time_interval, prediction_times
    ):
        """
        Calculate parameters required for the model.

        Args:
            df (pandas.DataFrame): The data frame to process.
            prediction_window (int): The total prediction window for prediction.
            time_interval (int): The interval for splitting the prediction window.
            prediction_times (list): Times of day at which predictions are made.

        Returns:
            dict: Calculated lambda_t parameters organized by time of day.

        """
        Ntimes = int(prediction_window / time_interval)
        arrival_rates_dict = calculate_rates(df, time_interval)
        prediction_time_dict = {}

        for prediction_time_ in prediction_times:
            prediction_time_hr, prediction_time_min = (
                (prediction_time_, 0)
                if isinstance(prediction_time_, int)
                else prediction_time_
            )
            lambda_t = [
                arrival_rates_dict[
                    (
                        datetime(1970, 1, 1, prediction_time_hr, prediction_time_min)
                        + i * timedelta(minutes=time_interval)
                    ).time()
                ]
                for i in range(Ntimes)
            ]
            prediction_time_dict[(prediction_time_hr, prediction_time_min)] = {
                "lambda_t": lambda_t
            }

        return prediction_time_dict

    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window: int,
        time_interval: int,
        prediction_times: List[float],
        epsilon: float = 10**-7,
        y: Optional[None] = None,
    ) -> 'PoissonBinomialPredictor':
        """
        Fits the model to the training data, computing necessary parameters for future predictions.

        Args:
            train_df (pandas.DataFrame):
                The training dataset with historical admission data.
            prediction_window (int):
                The prediction prediction window in minutes.
            time_interval (int):
                The interval in minutes for splitting the prediction window.
            prediction_times (list):
                Times of day at which predictions are made, in hours.
            epsilon (float, optional):
                A small value representing acceptable error rate to enable calculation of the maximum value of the random variable representing number of beds.
            y (None, optional):
                Ignored, present for compatibility with scikit-learn's fit method.

        Returns:
            PoissonBinomialPredictor: The instance itself, fitted with the training data.

        """
        # Store prediction_window, time_interval, and any other parameters as instance variables
        self.prediction_window = prediction_window
        self.time_interval = time_interval
        self.epsilon = epsilon
        self.prediction_times = prediction_times

        # Initialize yet_to_arrive_dict
        self.weights = {}

        # If there are filters specified, calculate and store the parameters directly with the respective spec keys
        if self.filters:
            for spec, filters in self.filters.items():
                self.weights[spec] = self._calculate_parameters(
                    self.filter_dataframe(train_df, filters),
                    prediction_window,
                    time_interval,
                    prediction_times,
                )
        else:
            # If there are no filters, store the parameters with a generic key, like 'default' or 'unfiltered'
            self.weights["default"] = self._calculate_parameters(
                train_df, prediction_window, time_interval, prediction_times
            )

        print(f"Poisson Binomial Predictor trained for these times: {prediction_times}")
        print(
            f"using prediction window of {prediction_window} minutes after the time of prediction"
        )
        print(
            f"and time interval of {time_interval} minutes within the prediction window."
        )
        print(f"The error value for prediction will be {epsilon}")
        print("To see the weights saved by this model, used the get_weights() method")

    def get_weights(self):
        """
        Returns the weights computed by the fit method.

        Returns
            dict: The weights.

        """
        return self.weights

    def _find_nearest_previous_prediction_time(self, requested_time):
        """
        Finds the nearest previous time of day in 'prediction_times' relative to the requested time.
        If the requested time is earlier than all times in 'prediction_times', the function returns
        the latest time in 'prediction_times'.

        Args:
            requested_time (tuple): The requested time as (hour, minute).

        Returns:
            tuple: The closest previous time of day from 'prediction_times'.

        """
        requested_datetime = datetime.strptime(
            f"{requested_time[0]:02d}:{requested_time[1]:02d}", "%H:%M"
        )
        closest_prediction_time = max(
            self.prediction_times,
            key=lambda prediction_time_time: datetime.strptime(
                f"{prediction_time_time[0]:02d}:{prediction_time_time[1]:02d}", "%H:%M"
            ),
        )
        min_diff = float("inf")

        for prediction_time_time in self.prediction_times:
            prediction_time_datetime = datetime.strptime(
                f"{prediction_time_time[0]:02d}:{prediction_time_time[1]:02d}", "%H:%M"
            )
            diff = (requested_datetime - prediction_time_datetime).total_seconds()

            # If the difference is negative, it means the prediction_time_time is ahead of the requested_time,
            # hence we calculate the difference by considering a day's wrap around.
            if diff < 0:
                diff += 24 * 60 * 60  # Add 24 hours in seconds

            if 0 <= diff < min_diff:
                closest_prediction_time = prediction_time_time
                min_diff = diff

        return closest_prediction_time

    def predict(self, prediction_context: Dict, x1: float, y1: float, x2: float, y2: float) -> Dict:
        """
        Predicts the number of admissions for the given context based on the fitted model.

        Args:
            prediction_context (dict): A dictionary defining the context for which predictions are to be made.
                                       It should specify either a general context or one based on the applied filters.
            x1 : float
                The x-coordinate of the first transition point on the aspirational curve, where the growth phase ends and the decay phase begins.
            y1 : float
                The y-coordinate of the first transition point (x1), representing the target proportion of patients admitted by time x1.
            x2 : float
                The x-coordinate of the second transition point on the curve, beyond which all but a few patients are expected to be admitted.
            y2 : float
                The y-coordinate of the second transition point (x2), representing the target proportion of patients admitted by time x2.

        Returns:
            dict: A dictionary with predictions for each specified context.

        """
        predictions = {}

        # theta = self.weights.get("theta", 1)  # Provide a default value or handle if missing
        NTimes = int(self.prediction_window / self.time_interval)
        # Calculate theta, probability of admission in prediction window

        # for each time interval, calculate time remaining before end of window
        time_remaining_before_end_of_window = self.prediction_window / 60 - np.arange(
            0, self.prediction_window / 60, self.time_interval / 60
        )

        # probability of admission in that time
        theta = get_y_from_aspirational_curve(
            time_remaining_before_end_of_window, x1, y1, x2, y2
        )

        for filter_key, filter_values in prediction_context.items():
            try:
                if filter_key not in self.weights:
                    raise ValueError(
                        f"Filter key '{filter_key}' is not recognized in the model weights."
                    )

                prediction_time = filter_values.get("prediction_time")
                if prediction_time is None:
                    raise ValueError(
                        f"No 'prediction_time' provided for filter '{filter_key}'."
                    )

                if prediction_time not in self.prediction_times:
                    original_prediction_time = prediction_time
                    prediction_time = self._find_nearest_previous_prediction_time(
                        prediction_time
                    )
                    warnings.warn(
                        f"Time of day requested of {original_prediction_time} was not in model training. "
                        f"Reverting to predictions for {prediction_time}."
                    )

                lambda_t = self.weights[filter_key][prediction_time].get("lambda_t")
                if lambda_t is None:
                    raise ValueError(
                        f"No 'lambda_t' found for the time of day '{prediction_time}' under filter '{filter_key}'."
                    )

                predictions[filter_key] = poisson_binom_generating_function(
                    NTimes, lambda_t, theta, self.epsilon
                )

            except KeyError as e:
                raise KeyError(f"Key error occurred: {e!s}")

        return predictions
