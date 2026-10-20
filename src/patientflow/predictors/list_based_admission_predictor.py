"""List-based elective yet-to-arrive predictor.

``ListBasedAdmissionPredictor`` is a ``DirectAdmissionPredictor`` that can
condition on a known list of patients booked for today's admission. It follows
the same pattern as the other yet-to-arrive predictors: fit once, store
interval-level quantities, and apply ``prediction_time`` / ``prediction_window``
only at inference.

The predicted count has two independent parts, which are convolved:

- **Listed patients** (Poisson-binomial). Every pending patient on the list
  arrives in the window with probability ``q_i`` or does not. In this version
  ``q_i`` is estimated per service, so within a service the count is
  ``Binomial(n, q)``.
- **Residual** (Poisson). Arrivals not on the list, modelled as a rate.

What is learned at fit time
---------------------------
These are stored per service, with one value per ``yta_time_interval`` over the
day:

- ``list_interval_probs`` (``p_k``): the probability that a patient listed for
  day *D* arrives at the listed service in interval *k* of day *D*.
- ``list_other_service_probs`` (``r_k``): the probability that a patient listed
  for day *D* at this service arrives at a **different** service in interval *k*
  of day *D*. ``sum_k (p_k + r_k) <= 1``.
- ``list_count``: the number of listed patients in the denominator of ``p_k``
  and ``r_k``.
- ``residual_arrival_rates_dict`` and ``residual_arrival_rates_by_weekday``:
  arrival rates for arrivals that were not on the list.

The usual ``arrival_rates_dict`` and ``arrival_rates_by_weekday`` (the *full
profile*, identical to ``DirectAdmissionPredictor``) are kept and are used when
no ``day_list`` is supplied.

The denominator is every listed patient who had not arrived before 00:00 on day
*D*, whether or not they later arrive. Patients who arrived earlier are dropped,
and count as residual arrivals on the day they arrived.

Inference
---------
For a pending patient at a service (listed, and not yet arrived at *any*
service by the prediction moment):

```text
q = sum(p_k over the part of the window on day D)
    / (1 - sum(p_k + r_k from 00:00 to the prediction time))
```

The listed count is Poisson-binomial over the ``q_i`` values. The residual rate
is the residual profile over the part of the window on day *D*, plus the full
profile for the next weekday over any part of the window after midnight (the next
day's list is not known). The predicted PMF is the convolution of the two.

Services with fewer than ``min_list_count`` listed patients have no list
probabilities and give the Poisson result from the full profile.

Boundary convention: a patient whose arrival is stamped exactly at the prediction
moment counts as already arrived (``arrival_datetime <= moment``), although the
``p_k`` interval starting at that moment is inside the window. The effect is
negligible.
"""

import warnings
from datetime import date, datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson

from patientflow.calculate.arrival_rates import (
    time_varying_arrival_rates,
    time_varying_arrival_rates_by_weekday,
)
from patientflow.predict.distribution import Distribution
from patientflow.predictors.incoming_admission_predictors import (
    DirectAdmissionPredictor,
)

_LIST_KEYS = ("list_interval_probs", "list_other_service_probs", "list_count")
_MIN_DENOMINATOR = 1e-12


class ListBasedAdmissionPredictor(DirectAdmissionPredictor):
    """Elective yet-to-arrive predictor that can condition on a booked list.

    Behaves exactly like ``DirectAdmissionPredictor`` unless a ``day_list``
    is passed to ``predict`` / ``predict_mean`` for a service that has list
    probabilities.

    Parameters
    ----------
    filters : dict, optional
        As for ``DirectAdmissionPredictor``. Keys are service names. The
        columns named in the filter values must also be present in ``list_df``
        at fit and in ``day_list`` at predict time.
    verbose : bool, default=False
        Whether to enable verbose logging.

    Attributes
    ----------
    weights : dict
        Per service: the full profile, the residual profile, and (for services
        with enough listed patients) ``list_interval_probs``,
        ``list_other_service_probs`` and ``list_count``.
    min_list_count : int
        Minimum ``list_count`` for a service to use the list.
    list_fallback_services : list of str
        Services with some list data but fewer than ``min_list_count`` listed
        patients. Fit warns about these.
    no_list_services : list of str
        Services with no listed patients at all. They fall back to Poisson
        silently, so a caller can fit on a list restricted to a few services.
    low_list_count_filter_count : int
        The length of ``list_fallback_services``.
    no_list_filter_count : int
        The length of ``no_list_services``.
    """

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_df: pd.DataFrame,
        prediction_window: Optional[timedelta] = None,
        yta_time_interval: Optional[timedelta] = None,
        prediction_times: Optional[List[float]] = None,
        num_days: Optional[int] = None,
        epsilon: float = 10**-7,
        y: Optional[None] = None,
        stratify_by_weekday: Optional[bool] = None,
        *,
        list_df: Optional[pd.DataFrame] = None,
        min_list_count: int = 50,
        encounter_col: str = "encounter",
        intended_date_col: str = "intended_date",
    ) -> "ListBasedAdmissionPredictor":
        """Fit the full profile and, when ``list_df`` is given, the list quantities.

        The positional parameters are the same as
        ``IncomingAdmissionPredictor.fit``. The list parameters are
        keyword-only.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Elective arrivals at **all** services, indexed by ``arrival_datetime``,
            with ``encounter_col`` and the columns used in ``filters``. It must not
            be a per-service slice, because listed patients who arrived at another
            service feed ``r_k``.
        prediction_window : timedelta, optional
            Deprecated. Pass ``prediction_window`` to ``predict()`` /
            ``predict_mean()`` instead.
        yta_time_interval : timedelta
            The granularity of arrival-rate buckets. Required.
        prediction_times : list, optional
            Deprecated. Retained for backward compatibility only.
        num_days : int, optional
            Divisor for the pooled arrival rates. Inferred from the date span of
            ``train_df`` when omitted. See ``IncomingAdmissionPredictor.fit``.
        epsilon : float, default=1e-7
            Acceptable error used to set the maximum value of the predicted
            distribution.
        y : None, optional
            Ignored, present for compatibility with scikit-learn's fit method.
        stratify_by_weekday : bool or None, default=None
            Whether to fit weekday arrival profiles. ``None`` fits them silently,
            ``True`` also warns at predict time when ``prediction_date`` is
            omitted, and ``False`` opts out. See ``IncomingAdmissionPredictor.fit``.
        list_df : pandas.DataFrame, optional
            One row per listed patient with ``encounter_col``,
            ``intended_date_col`` and the columns used in ``filters``. The caller
            decides which list snapshot qualifies. If omitted, the fit is the same
            as ``DirectAdmissionPredictor.fit``.
        min_list_count : int, default=50
            A service with fewer listed patients (pooled over training days, in
            the denominator of ``p_k``) gets no list probabilities and falls back
            to Poisson. A single ``UserWarning`` names services that had some
            listed patients but too few. Services with none are not warned about
            (see ``no_list_services``).
        encounter_col : str, default="encounter"
            Encounter identifier column in ``train_df`` and ``list_df``.
        intended_date_col : str, default="intended_date"
            Column in ``list_df`` holding the day the patient is booked for.

        Returns
        -------
        ListBasedAdmissionPredictor
            The fitted instance.
        """
        if min_list_count < 0:
            raise ValueError("min_list_count must be non-negative")

        super().fit(
            train_df,
            prediction_window=prediction_window,
            yta_time_interval=yta_time_interval,
            prediction_times=prediction_times,
            num_days=num_days,
            epsilon=epsilon,
            stratify_by_weekday=stratify_by_weekday,
        )
        assert yta_time_interval is not None

        self.min_list_count = int(min_list_count)
        self.encounter_col = encounter_col
        self.intended_date_col = intended_date_col
        self.list_fallback_services: List[str] = []
        self.no_list_services: List[str] = []
        self.low_list_count_filter_count = 0
        self.no_list_filter_count = 0
        self._n_intervals = int(pd.Timedelta(days=1) / yta_time_interval)

        if list_df is None:
            return self

        self._validate_fit_inputs(train_df, list_df)

        service_filters: Dict[str, Dict] = (
            dict(self.filters) if self.filters else {"unfiltered": {}}
        )

        train_encounters = train_df[encounter_col].to_numpy()
        arrival_times = train_df.index

        rows_by_arrival_time = np.argsort(arrival_times.values, kind="stable")
        first_arrival_row: Dict = {}
        for arrival_row in rows_by_arrival_time:
            first_arrival_row.setdefault(
                train_encounters[arrival_row], int(arrival_row)
            )

        listed_dates = pd.to_datetime(list_df[intended_date_col]).dt.date.to_numpy()
        listed_encounters = list_df[encounter_col].to_numpy()
        interval_seconds = int(yta_time_interval.total_seconds())

        for service, filters in service_filters.items():
            service_arrival_mask = self._filter_mask(train_df, filters)
            service_arrivals = train_df[service_arrival_mask]
            service_listed_mask = self._filter_mask(list_df, filters)

            # Numerators of p_k and r_k, and their shared denominator.
            same_service_counts = np.zeros(self._n_intervals)
            other_service_counts = np.zeros(self._n_intervals)
            n_listed = 0
            listed_arrival_encounters = set()

            for encounter, intended_date in zip(
                listed_encounters[service_listed_mask],
                listed_dates[service_listed_mask],
            ):
                arrival_row = first_arrival_row.get(encounter)
                if arrival_row is None:
                    n_listed += 1  # never arrived: missing mass
                    continue
                arrival_time = arrival_times[arrival_row]
                if arrival_time.date() < intended_date:
                    continue  # arrived before the day; dropped from denominator
                n_listed += 1
                if arrival_time.date() > intended_date:
                    continue  # moved to a later date: missing mass
                interval_index = (
                    arrival_time.hour * 3600
                    + arrival_time.minute * 60
                    + arrival_time.second
                ) // interval_seconds
                if service_arrival_mask[arrival_row]:
                    same_service_counts[interval_index] += 1
                    listed_arrival_encounters.add(encounter)
                else:
                    other_service_counts[interval_index] += 1

            residual_arrivals = service_arrivals[
                ~service_arrivals[encounter_col].isin(listed_arrival_encounters)
            ]
            weights = self.weights[service]

            if n_listed >= max(self.min_list_count, 1):
                weights.update(
                    self._residual_parameters(
                        residual_arrivals, service_arrivals, yta_time_interval, num_days
                    )
                )
                weights["list_interval_probs"] = same_service_counts / n_listed
                weights["list_other_service_probs"] = other_service_counts / n_listed
                weights["list_count"] = n_listed
            else:
                # Fall back: residual equals the full profile; no list probabilities.
                weights["residual_arrival_rates_dict"] = weights["arrival_rates_dict"]
                if "arrival_rates_by_weekday" in weights:
                    weights["residual_arrival_rates_by_weekday"] = weights[
                        "arrival_rates_by_weekday"
                    ]
                weights["list_count"] = n_listed
                if n_listed == 0:
                    self.no_list_services.append(service)
                else:
                    self.list_fallback_services.append(service)

        self.low_list_count_filter_count = len(self.list_fallback_services)
        self.no_list_filter_count = len(self.no_list_services)
        if self.list_fallback_services:
            warnings.warn(
                f"{self.low_list_count_filter_count} service(s) have fewer than "
                f"min_list_count={self.min_list_count} listed patients and will use "
                f"the Poisson result: {sorted(self.list_fallback_services)}.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _validate_fit_inputs(self, train_df: pd.DataFrame, list_df: pd.DataFrame):
        if not isinstance(train_df.index, pd.DatetimeIndex):
            raise TypeError("train_df must be indexed by arrival_datetime")
        filter_cols = {
            column
            for service_filter in (self.filters or {}).values()
            for column in service_filter
        }
        for name, df, needed in (
            ("train_df", train_df, {self.encounter_col} | filter_cols),
            (
                "list_df",
                list_df,
                {self.encounter_col, self.intended_date_col} | filter_cols,
            ),
        ):
            missing = needed - set(df.columns)
            if missing:
                raise ValueError(
                    f"{name} is missing required columns: {sorted(missing)}"
                )

    @staticmethod
    def _filter_mask(df: pd.DataFrame, filters: Dict) -> np.ndarray:
        """Boolean mask equivalent to ``filter_dataframe`` (positional)."""
        mask = np.ones(len(df), dtype=bool)
        for column, criteria in filters.items():
            if callable(criteria):
                mask &= df[column].apply(criteria).to_numpy(dtype=bool)
            else:
                mask &= (df[column] == criteria).to_numpy()
        return mask

    def _residual_parameters(
        self,
        residual_arrivals: pd.DataFrame,
        service_arrivals: pd.DataFrame,
        yta_time_interval,
        num_days: Optional[int],
    ) -> Dict:
        """Residual rates using the divisors of the service's full profile."""
        if len(residual_arrivals.index) == 0 or len(service_arrivals.index) == 0:
            params = self._calculate_parameters(
                residual_arrivals.iloc[0:0],
                yta_time_interval,
                num_days,
                stratify_by_weekday=self.stratify_by_weekday,
            )
        else:
            effective_days = self._resolve_num_days(service_arrivals, num_days)
            params = {
                "arrival_rates_dict": time_varying_arrival_rates(
                    residual_arrivals, yta_time_interval, effective_days
                )
            }
            if self.stratify_by_weekday:
                params["arrival_rates_by_weekday"] = (
                    time_varying_arrival_rates_by_weekday(
                        residual_arrivals,
                        yta_time_interval,
                        num_days,
                        date_span=(
                            service_arrivals.index.date.min(),
                            service_arrivals.index.date.max(),
                        ),
                    )
                )
        return {f"residual_{name}": rates for name, rates in params.items()}

    # -------------------------------------------------------------- predict
    def predict(
        self,
        *args,
        prediction_time=None,
        prediction_window=None,
        filter_keys: Optional[Union[str, Sequence[str]]] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        day_list: Optional[pd.DataFrame] = None,
        arrivals: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict:
        """Predict the PMF of elective arrivals per service.

        Without ``day_list`` this is ``DirectAdmissionPredictor.predict``.

        Parameters
        ----------
        prediction_time : tuple or list or int, optional
            Time of day ``(hour, minute)``, snapped to a ``yta_time_interval``
            boundary. Required unless using the deprecated ``prediction_context``.
        prediction_window : timedelta, optional
            The window over which arrivals are counted.
        filter_keys : str or sequence of str, optional
            Service name(s), keys of ``weights``. Required when ``weights`` has
            more than one key.
        prediction_context : dict, optional
            Deprecated nested dict API. Cannot be combined with ``day_list``.
        prediction_date : datetime.date, optional
            Calendar date at ``prediction_time``. Required when ``day_list`` is
            given. See ``DirectAdmissionPredictor.predict``.
        strict_prediction_date : bool, default=False
            If ``True``, raise ``ValueError`` instead of falling back to pooled
            rates.
        day_list : pandas.DataFrame, optional
            The full list for ``prediction_date``: ``encounter`` and the columns
            used in ``filters``, including patients who may already have arrived.
        arrivals : pandas.DataFrame, optional
            Elective arrivals at **all** services, with ``encounter`` and an
            ``arrival_datetime`` (column, or ``DatetimeIndex`` if there is no such
            column). Required when ``day_list`` is given. It must reach back to the
            day the list was produced. Rows after the prediction moment are ignored.
        **kwargs
            ``max_value`` : int, optional, maximum PMF support for the residual.

        Returns
        -------
        dict
            Service key to DataFrame indexed by ``sum`` with column ``agg_proba``.
        """
        if day_list is None:
            return super().predict(
                *args,
                prediction_time=prediction_time,
                prediction_window=prediction_window,
                filter_keys=filter_keys,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
                strict_prediction_date=strict_prediction_date,
                **kwargs,
            )

        self._validate_list_call(args, prediction_context, prediction_date, arrivals)
        assert prediction_date is not None
        window = self._resolve_prediction_window(prediction_window)
        snapped_time, services, _ = self._prepare_prediction_targets_for_predict(
            prediction_time=prediction_time,
            filter_keys=filter_keys,
            prediction_date=prediction_date,
        )
        max_value = kwargs.get("max_value")
        predictions = {}
        for service in services:
            if not self.uses_list(service):
                predictions[service] = super().predict(
                    prediction_time=snapped_time,
                    prediction_window=window,
                    filter_keys=[service],
                    prediction_date=prediction_date,
                    strict_prediction_date=strict_prediction_date,
                    **kwargs,
                )[service]
                continue
            q_values, residual_rate = self._list_components(
                service,
                snapped_time,
                window,
                prediction_date,
                day_list,
                arrivals,
                strict_prediction_date,
            )
            predictions[service] = self._pmf_to_dataframe(
                self._combine(q_values, residual_rate, max_value).probabilities
            )
        return predictions

    def predict_mean(
        self,
        *args,
        prediction_time=None,
        prediction_window=None,
        filter_key: Optional[str] = None,
        prediction_context: Optional[Dict] = None,
        prediction_date: Optional[date] = None,
        strict_prediction_date: bool = False,
        day_list: Optional[pd.DataFrame] = None,
        arrivals: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> float:
        """Expected arrivals for one service.

        Without ``day_list``, or for a service with no list probabilities, this is
        ``DirectAdmissionPredictor.predict_mean``. Otherwise it is the sum of
        ``q_i`` over pending listed patients plus the residual rate.
        Arguments are as for ``predict``.
        """
        if day_list is None:
            return super().predict_mean(
                *args,
                prediction_time=prediction_time,
                prediction_window=prediction_window,
                filter_key=filter_key,
                prediction_context=prediction_context,
                prediction_date=prediction_date,
                strict_prediction_date=strict_prediction_date,
                **kwargs,
            )
        self._validate_list_call(args, prediction_context, prediction_date, arrivals)
        assert prediction_date is not None
        window = self._resolve_prediction_window(prediction_window)
        snapped_time, services, _ = self._prepare_prediction_targets_for_predict_mean(
            prediction_time=prediction_time,
            filter_key=filter_key,
            prediction_date=prediction_date,
        )
        service = services[0]
        if not self.uses_list(service):
            return super().predict_mean(
                prediction_time=snapped_time,
                prediction_window=window,
                filter_key=service,
                prediction_date=prediction_date,
                strict_prediction_date=strict_prediction_date,
                **kwargs,
            )
        q_values, residual_rate = self._list_components(
            service,
            snapped_time,
            window,
            prediction_date,
            day_list,
            arrivals,
            strict_prediction_date,
        )
        return float(np.sum(q_values) + residual_rate)

    # ------------------------------------------------------------- internals
    def uses_list(self, key: Optional[str]) -> bool:
        """Whether ``key`` has list probabilities (enough listed patients at fit)."""
        return (
            key is not None
            and key in self.weights
            and "list_interval_probs" in self.weights[key]
        )

    @staticmethod
    def _validate_list_call(args, prediction_context, prediction_date, arrivals):
        """Check the arguments that a call with ``day_list`` requires."""
        if args or prediction_context is not None:
            raise ValueError(
                "day_list cannot be combined with the deprecated prediction_context API."
            )
        if prediction_date is None:
            raise ValueError("prediction_date is required when day_list is supplied.")
        if arrivals is None:
            raise ValueError(
                "arrivals is required when day_list is supplied (pass an empty "
                "frame if nobody has arrived)."
            )

    def _arrived_encounters(self, arrivals: pd.DataFrame, moment: datetime) -> set:
        if self.encounter_col not in arrivals.columns:
            raise ValueError(f"arrivals is missing column '{self.encounter_col}'")
        if "arrival_datetime" in arrivals.columns:
            arrival_times = pd.to_datetime(arrivals["arrival_datetime"])
        elif isinstance(arrivals.index, pd.DatetimeIndex):
            arrival_times = arrivals.index.to_series(index=arrivals.index)
        else:
            raise ValueError(
                "arrivals needs an 'arrival_datetime' column or a DatetimeIndex"
            )
        arrived_by_moment = (arrival_times <= pd.Timestamp(moment)).to_numpy()
        return set(arrivals[self.encounter_col].to_numpy()[arrived_by_moment])

    def _list_components(
        self,
        service: str,
        snapped_time: Tuple[int, int],
        window,
        prediction_date: date,
        day_list: pd.DataFrame,
        arrivals: pd.DataFrame,
        strict: bool,
    ) -> Tuple[np.ndarray, float]:
        """Return per-patient ``q_i`` for pending listed patients and the residual rate."""
        weights = self.weights[service]
        hour, minute = snapped_time
        moment = datetime.combine(prediction_date, dt_time(hour, minute))
        n_intervals_per_day = self._n_intervals
        n_window_intervals = int(window / self.yta_time_interval)

        # Pending listed patients at this service.
        filters = self.filters.get(service, {}) if self.filters else {}
        needed = {self.encounter_col} | set(filters)
        missing = needed - set(day_list.columns)
        if missing:
            raise ValueError(f"day_list is missing required columns: {sorted(missing)}")
        listed = day_list[self._filter_mask(day_list, filters)]
        listed_encounters = pd.unique(listed[self.encounter_col])
        arrived = self._arrived_encounters(arrivals, moment)
        n_pending = sum(
            1 for encounter in listed_encounters if encounter not in arrived
        )

        # Service-level q (same for every pending patient in this version):
        # q = sum p_k over the window on day D / (1 - sum (p_k + r_k) before t).
        same_service_probs = np.asarray(weights["list_interval_probs"], dtype=float)
        other_service_probs = np.asarray(
            weights["list_other_service_probs"], dtype=float
        )
        start_interval = (hour * 3600 + minute * 60) // int(
            self.yta_time_interval.total_seconds()
        )
        end_interval = min(start_interval + n_window_intervals, n_intervals_per_day)
        p_window = float(same_service_probs[start_interval:end_interval].sum())
        p_still_pending = max(
            1.0
            - float(
                (
                    same_service_probs[:start_interval]
                    + other_service_probs[:start_interval]
                ).sum()
            ),
            _MIN_DENOMINATOR,
        )
        q = float(min(1.0, max(0.0, p_window / p_still_pending)))
        q_values = np.full(n_pending, q)

        # Residual rate: residual profile on day D, full profile after midnight.
        residual_rates_by_weekday = weights.get("residual_arrival_rates_by_weekday")
        full_rates_by_weekday = weights.get("arrival_rates_by_weekday")
        if residual_rates_by_weekday is None or full_rates_by_weekday is None:
            message = (
                "prediction_date was provided but no weekday-stratified rates were "
                f"found for filter '{service}'. Fit with stratify_by_weekday=True to "
                "enable weekday-stratified arrival rates."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(
                message + " Falling back to pooled arrival rates.",
                UserWarning,
                stacklevel=4,
            )
        residual_rate = 0.0
        for i in range(n_window_intervals):
            interval_start = moment + i * self.yta_time_interval
            interval_time = interval_start.time()
            if interval_start.date() == prediction_date:
                profile = (
                    residual_rates_by_weekday[interval_start.weekday()]
                    if residual_rates_by_weekday is not None
                    else weights["residual_arrival_rates_dict"]
                )
            else:
                profile = (
                    full_rates_by_weekday[interval_start.weekday()]
                    if full_rates_by_weekday is not None
                    else weights["arrival_rates_dict"]
                )
            residual_rate += float(profile[interval_time])
        return q_values, residual_rate

    def _combine(
        self, q_values: np.ndarray, residual_rate: float, max_value: Optional[int]
    ) -> Distribution:
        """Convolve the listed Poisson-binomial with the residual Poisson."""
        listed = Distribution.from_bernoulli_probs(q_values)
        if max_value is not None:
            residual_max_count = int(max_value) - 1
        else:
            residual_max_count = (
                int(poisson.ppf(1 - self.epsilon, residual_rate))
                if residual_rate > 0
                else 0
            )
        residual = Distribution.from_poisson_with_cap(
            residual_rate, max(residual_max_count, 0)
        )
        return listed.convolve(residual)
