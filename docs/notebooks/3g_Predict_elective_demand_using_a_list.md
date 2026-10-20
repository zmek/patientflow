# 3g. Predict elective demand using a list of planned admissions

In notebook 3e I predicted demand from patients yet to arrive using past arrival rates alone. For elective admissions, that approach treats the number of patients who will come in today as uncertain. In practice, hospitals often know a good deal more than that. Planned admissions are recorded on a list some hours before the patients arrive, for example on a 'To Come In' (TCI) list, or as pending admission requests raised the afternoon before.

When a list like this is available, most of the uncertainty about how many electives will arrive is already resolved. What remains is:

- **Attrition.** Some listed patients do not arrive. They cancel, do not attend, are deferred to another day, or arrive at a different service from the one they were listed under.
- **Timing.** A listed patient who does arrive may do so before or after the prediction window.
- **Unlisted electives.** Some elective patients are never on the list.

In this notebook I show how to use the `ListBasedAdmissionPredictor` class to condition the prediction on the list. The predicted count has two independent parts:

- **Listed patients still to arrive.** Each pending patient arrives in the window with probability `q`, or does not, so the number arriving is binomial. (In general this is a Poisson-binomial, because `q` could differ between patients. In this version `q` is estimated per service, so within a service it is a plain binomial.)
- **Off-list arrivals.** Electives who were not on that day's list for that service are modelled with a Poisson distribution, as in notebook 3e. I call these _off-list_ electives. They include patients who were never listed, and also patients who were listed for another day or under another service.

The two distributions are convolved to give the prediction.

A note on identifiers. In notebook 3e I said that the hospital visit for an elective procedure may not be recorded in data before the patient arrives. Where a list of planned admissions exists, the visit identifier (here called `encounter`) is often created when the patient is listed, before they arrive. That is what makes it possible to match the list to later arrivals, and to drop listed patients from the list once they have arrived.

```python
# Reload functions every time
%load_ext autoreload
%autoreload 2
```

## Generate fake data

I use the `create_fake_elective_lists` function to generate about three months of fake data for three services:

- `surgical` has a large list on weekdays and a small one at weekends.
- `paediatric` has a very short list, too short to estimate list probabilities reliably.
- `haem/onc` has no list at all. All its electives are unlisted.

Each listed patient has one outcome. They arrive at the listed service on the day, arrive at a different service on the day, arrive the evening before, are deferred to the next day, or do not arrive. Unlisted electives also arrive at every service.

The function returns two dataframes. `arrivals` has one row per elective arrival, and `list_df` has one row per listed patient with the date they were booked for.

```python
from patientflow.generate import create_fake_elective_lists

arrivals, list_df = create_fake_elective_lists('2023-01-02', '2023-04-03')
arrivals.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>arrival_datetime</th>
      <th>encounter</th>
      <th>service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 17:46:42</td>
      <td>E0000011</td>
      <td>surgical</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 18:25:05</td>
      <td>E0000008</td>
      <td>surgical</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 18:50:08</td>
      <td>E0000026</td>
      <td>surgical</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-02 06:05:08</td>
      <td>E0000025</td>
      <td>surgical</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-02 07:16:06</td>
      <td>E0000028</td>
      <td>surgical</td>
    </tr>
  </tbody>
</table>
</div>

```python
list_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter</th>
      <th>service</th>
      <th>intended_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0000007</td>
      <td>paediatric</td>
      <td>2023-01-02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E0000008</td>
      <td>surgical</td>
      <td>2023-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E0000009</td>
      <td>surgical</td>
      <td>2023-01-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E0000010</td>
      <td>surgical</td>
      <td>2023-01-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E0000011</td>
      <td>surgical</td>
      <td>2023-01-02</td>
    </tr>
  </tbody>
</table>
</div>

The table below shows what happened to the listed patients. Most arrive at the listed service on the day they were booked for, but a substantial minority do not.

```python
import numpy as np
import pandas as pd

listed_outcomes = list_df.merge(arrivals, on='encounter', how='left', suffixes=('_listed', '_arrived'))
arrival_date = listed_outcomes['arrival_datetime'].dt.date

listed_outcomes['outcome'] = np.select(
    [
        listed_outcomes['arrival_datetime'].isna(),
        arrival_date < listed_outcomes['intended_date'],
        arrival_date > listed_outcomes['intended_date'],
        listed_outcomes['service_arrived'] != listed_outcomes['service_listed'],
    ],
    [
        'did not arrive',
        'arrived the evening before',
        'arrived on a later day',
        'arrived at a different service',
    ],
    default='arrived at the listed service on the day',
)

listed_outcomes.groupby('service_listed')['outcome'].value_counts(normalize=True).unstack(fill_value=0).round(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>outcome</th>
      <th>arrived at a different service</th>
      <th>arrived at the listed service on the day</th>
      <th>arrived on a later day</th>
      <th>arrived the evening before</th>
      <th>did not arrive</th>
    </tr>
    <tr>
      <th>service_listed</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>paediatric</th>
      <td>0.04</td>
      <td>0.81</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>surgical</th>
      <td>0.05</td>
      <td>0.73</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.14</td>
    </tr>
  </tbody>
</table>
</div>

## Split into training and test periods

I use the first ten weeks for training and the remaining three weeks for testing.

The list for the first day of the training period is left out. Some patients on that list arrived the evening before, which falls outside the training period, so they would look as if they never arrived. Starting the list one day in avoids that.

```python
from datetime import date

start_training_set = date(2023, 1, 2)
start_test_set = date(2023, 3, 13)
num_days = (start_test_set - start_training_set).days

arrival_dates = arrivals['arrival_datetime'].dt.date
train_arrivals = arrivals[(arrival_dates >= start_training_set) & (arrival_dates < start_test_set)]

train_list = list_df[(list_df['intended_date'] > start_training_set) & (list_df['intended_date'] < start_test_set)]

print(f'Training arrivals: {len(train_arrivals)}; training list rows: {len(train_list)}; days: {num_days}')
```

    Training arrivals: 1447; training list rows: 999; days: 70

## Fit the model

`ListBasedAdmissionPredictor` is a subclass of `DirectAdmissionPredictor`, which predicts arrivals with a Poisson distribution based on past arrival rates, like the first approach in notebook 3e. It is fitted in the same way, with one difference: the list is passed as `list_df`.

Two things to note about the inputs:

- `train_df` must contain elective arrivals at **all** services, not a slice for one service. The model needs to see listed patients who arrived at a different service.
- The columns used in `filters` (here `service`) must be present in both `train_df` and `list_df`, so that listed patients can be assigned to services.

At fit time, the model learns, for each service and each 15-minute interval of the day:

- `p_k`, the probability that a listed patient arrives at the listed service in that interval. It is stored as `list_interval_probs`.
- `r_k`, the probability that a listed patient arrives at a different service in that interval. It is stored as `list_other_service_probs`.
- Arrival rates for off-list electives (the _off-list rates_).

It also keeps the usual arrival rates for all electives (the _all-elective rates_), so that it behaves exactly like `DirectAdmissionPredictor` when no list is supplied.

A service needs at least `min_list_count` listed patients (50 by default) for its list probabilities to be used. Below that, it falls back to the Poisson result. The model warns about services that had some listed patients but too few.

```python
from datetime import timedelta
from patientflow.predictors.list_based_admission_predictor import ListBasedAdmissionPredictor

services = ['surgical', 'paediatric', 'haem/onc']
filters = {service: {'service': service} for service in services}

yta_time_interval = timedelta(minutes=15)

import warnings

list_model = ListBasedAdmissionPredictor(filters=filters)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    list_model.fit(
        train_arrivals.set_index('arrival_datetime'),
        list_df=train_list,
        yta_time_interval=yta_time_interval,
        num_days=num_days,
    )
for warning in caught:
    print(f'Warning: {warning.message}')
```

    Warning: 1 service(s) have fewer than min_list_count=50 listed patients and will use the Poisson result: ['paediatric'].

The table below summarises what was learned for each service. `list_count` is the number of listed patients in the denominator of `p_k` and `r_k`. Summing `p_k` over the day gives the share of listed patients who arrive at the listed service on the day. Summing `r_k` gives the share who arrive at a different service.

```python
rows = []
for service in services:
    weights = list_model.weights[service]
    rows.append({
        'service': service,
        'list_count': weights['list_count'],
        'uses list': list_model.uses_list(service),
        'arrive at listed service (sum of p_k)': weights['list_interval_probs'].sum() if list_model.uses_list(service) else None,
        'arrive at other service (sum of r_k)': weights['list_other_service_probs'].sum() if list_model.uses_list(service) else None,
    })
pd.DataFrame(rows).set_index('service').round(3)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>list_count</th>
      <th>uses list</th>
      <th>arrive at listed service (sum of p_k)</th>
      <th>arrive at other service (sum of r_k)</th>
    </tr>
    <tr>
      <th>service</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>surgical</th>
      <td>933</td>
      <td>True</td>
      <td>0.75</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>paediatric</th>
      <td>21</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>haem/onc</th>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

`haem/onc` had no listed patients, so it falls back to Poisson without a warning. `paediatric` had some, but fewer than `min_list_count`, so it falls back with the warning shown above.

### When do listed patients arrive?

The chart below shows `p_k` and `r_k` for the surgical list, summed to hourly values. Listed patients arrive mainly in the morning.

```python
import matplotlib.pyplot as plt

intervals_per_hour = int(timedelta(hours=1) / yta_time_interval)
surgical = list_model.weights['surgical']
p_by_hour = surgical['list_interval_probs'].reshape(24, intervals_per_hour).sum(axis=1)
r_by_hour = surgical['list_other_service_probs'].reshape(24, intervals_per_hour).sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(24), p_by_hour, label='Arrives at the listed service (p_k)')
ax.bar(range(24), r_by_hour, bottom=p_by_hour, label='Arrives at a different service (r_k)')
ax.set_xticks(range(24))
ax.set_xlabel('Hour of day')
ax.set_ylabel('Probability per listed patient')
ax.set_title('When listed surgical patients arrive on the day they were booked for')
ax.legend()
plt.show()
```

![png](3g_Predict_elective_demand_using_a_list_files/3g_Predict_elective_demand_using_a_list_14_0.png)

### All-elective and off-list arrival rates

The all-elective rates include every elective arrival. The off-list rates leave out patients who arrived on the day they were listed for, at the service they were listed under. The gap between the two is the listed patients. The off-list rates describe what remains uncertain once the list is known.

```python
def hourly(rates_by_time):
    return np.array(list(rates_by_time.values())).reshape(24, intervals_per_hour).sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(24), hourly(surgical['arrival_rates_dict']), marker='o', label='All electives')
ax.plot(range(24), hourly(surgical['off_list_arrival_rates_dict']), marker='s', label='Off-list electives')
ax.set_xticks(range(24))
ax.set_xlabel('Hour of day')
ax.set_ylabel('Mean arrivals per hour')
ax.set_title('Surgical elective arrival rates, averaged over all days of the week')
ax.legend()
plt.show()
```

![png](3g_Predict_elective_demand_using_a_list_files/3g_Predict_elective_demand_using_a_list_16_0.png)

## Predict for one day

To make a prediction, the model needs three things in addition to the usual `prediction_time` and `prediction_window`:

- `day_list`: the list for the prediction date, including patients who may already have arrived.
- `arrivals`: elective arrivals at all services. The model uses them to drop listed patients who have already arrived by the prediction time, at any service. Arrivals after the prediction time are ignored, so the same frame can be used to replay past days.
- `prediction_date`: the date, which anchors the prediction time and selects the arrival rates for that weekday.

I use Thursday 16 March 2023 from the test period.

```python
prediction_date = date(2023, 3, 16)
prediction_window = timedelta(hours=8)
day_list = list_df[list_df['intended_date'] == prediction_date]

print(f'Patients on the list for {prediction_date}: {len(day_list)}')
day_list['service'].value_counts()
```

    Patients on the list for 2023-03-16: 12





    service
    surgical    12
    Name: count, dtype: int64

### Conditioning on not having arrived yet

At the prediction time, some listed patients have already arrived. The model drops them, and predicts only for those still pending. For each pending patient, the probability of arriving in the window is

```
q = sum of p_k over the part of the window on the day
    / (1 - sum of (p_k + r_k) from 00:00 to the prediction time)
```

The numerator alone is the probability that _any_ listed patient arrives in the window. But by the prediction time, many of the patients who were always going to arrive early have already done so, and have been dropped. The patients still pending are therefore more likely than average to arrive later, so the numerator understates their chance of arriving in the window. Dividing by the probability of not having arrived yet corrects for this. At 00:00 the denominator is 1, so the two are the same.

The code below works through this calculation for the surgical list at 06:00 and at 09:30, and compares the result with the mean from the model. At 06:00 the conditioning makes no difference, because in this fake data no listed patient arrives before 06:00. At 09:30 it nearly doubles the probability for each pending patient.

```python
from datetime import datetime

interval_seconds = int(yta_time_interval.total_seconds())
n_window_intervals = int(prediction_window / yta_time_interval)
p = surgical['list_interval_probs']
r = surgical['list_other_service_probs']
surgical_list = day_list[day_list['service'] == 'surgical']
empty_list = day_list.iloc[0:0]

for prediction_time in [(6, 0), (9, 30)]:
    moment = datetime.combine(prediction_date, datetime.min.time()).replace(hour=prediction_time[0], minute=prediction_time[1])
    arrived = set(arrivals.loc[arrivals['arrival_datetime'] <= moment, 'encounter'])
    n_pending = (~surgical_list['encounter'].isin(arrived)).sum()

    start = (prediction_time[0] * 3600 + prediction_time[1] * 60) // interval_seconds
    end = min(start + n_window_intervals, len(p))
    unconditioned = p[start:end].sum()
    q = unconditioned / (1 - (p[:start] + r[:start]).sum())

    common = dict(prediction_time=prediction_time, prediction_window=prediction_window,
                  filter_key='surgical', prediction_date=prediction_date, arrivals=arrivals)
    off_list_rate = list_model.predict_mean(day_list=empty_list, **common)
    mean = list_model.predict_mean(day_list=day_list, **common)

    print(f'At {prediction_time[0]:02d}:{prediction_time[1]:02d}: {len(surgical_list)} listed, {n_pending} still pending, '
          f'unconditioned probability = {unconditioned:.3f}, q = {q:.3f}, off-list rate = {off_list_rate:.2f}')
    print(f'    expected arrivals = {n_pending} x {q:.3f} + {off_list_rate:.2f} = {n_pending * q + off_list_rate:.2f} '
          f'(model: {mean:.2f})')
```

    At 06:00: 12 listed, 12 still pending, unconditioned probability = 0.746, q = 0.746, off-list rate = 3.00
        expected arrivals = 12 x 0.746 + 3.00 = 11.95 (model: 11.95)
    At 09:30: 12 listed, 6 still pending, unconditioned probability = 0.333, q = 0.604, off-list rate = 2.70
        expected arrivals = 6 x 0.604 + 2.70 = 6.32 (model: 6.32)

The `predict()` method returns the full probability distribution. Below are the distributions for the surgical service at the two prediction times.

```python
from patientflow.viz.probability_distribution import plot_prob_dist
from patientflow.viz.pipeline_plots import create_colour_dict

colour_dict = create_colour_dict()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, prediction_time in zip(axes, [(6, 0), (9, 30)]):
    pmf = list_model.predict(
        prediction_time=prediction_time,
        prediction_window=prediction_window,
        filter_keys='surgical',
        prediction_date=prediction_date,
        day_list=day_list,
        arrivals=arrivals,
    )['surgical']
    plot_prob_dist(
        pmf,
        f'Surgical electives, 8 hours from {prediction_time[0]:02d}:{prediction_time[1]:02d}',
        include_titles=True,
        truncate_at_beds=25,
        bar_colour=colour_dict['single']['surgical'],
        xlabel='Number of elective arrivals',
        ax=ax,
    )
plt.tight_layout()
plt.show()
```

![png](3g_Predict_elective_demand_using_a_list_files/3g_Predict_elective_demand_using_a_list_22_0.png)

## Compare with the Poisson-only prediction

If no `day_list` is passed, the same fitted object gives exactly the result of `DirectAdmissionPredictor`, a Poisson distribution based on past arrival rates for that weekday. This is similar to the first approach in notebook 3e.

Below I compare the two predictions at 06:00, with the number of surgical electives who actually arrived in the window marked in red. On this day, the list was shorter than usual for a Thursday. The list-based prediction reflects that, whereas the Poisson-only prediction expects a typical Thursday.

```python
prediction_time = (6, 0)
# max_value sets the largest count in each distribution, so that neither is cut short
common = dict(prediction_time=prediction_time, prediction_window=prediction_window,
              filter_keys='surgical', prediction_date=prediction_date, max_value=60)

with_list = list_model.predict(day_list=day_list, arrivals=arrivals, **common)['surgical']
without_list = list_model.predict(**common)['surgical']

moment = datetime.combine(prediction_date, datetime.min.time()).replace(hour=6)
observed = ((arrivals['service'] == 'surgical')
            & (arrivals['arrival_datetime'] > moment)
            & (arrivals['arrival_datetime'] <= moment + prediction_window)).sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for ax, pmf, label in zip(axes, [without_list, with_list], ['Poisson only (no list)', 'Using the list']):
    plot_prob_dist(pmf, label, include_titles=True, truncate_at_beds=30,
                   bar_colour=colour_dict['single']['surgical'],
                   xlabel='Number of elective arrivals', ax=ax)
    ax.axvline(observed, color='red', linestyle='--', label=f'Observed ({observed})')
    ax.legend()
plt.tight_layout()
plt.show()
```

![png](3g_Predict_elective_demand_using_a_list_files/3g_Predict_elective_demand_using_a_list_24_0.png)

The table compares the mean and the range covering the middle 80% of each distribution. Using the list moves the prediction towards what happened on this day, and narrows the range, because the size of the list is known rather than estimated.

```python
def summarise(pmf):
    probabilities = pmf['agg_proba'].to_numpy()
    counts = pmf.index.to_numpy()
    cumulative = np.cumsum(probabilities)
    return {
        'mean': float((counts * probabilities).sum()),
        '10th percentile': int(counts[np.searchsorted(cumulative, 0.10)]),
        '90th percentile': int(counts[np.searchsorted(cumulative, 0.90)]),
    }

pd.DataFrame({
    'Poisson only (no list)': summarise(without_list),
    'Using the list': summarise(with_list),
}).T.round(2).assign(observed=observed)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>10th percentile</th>
      <th>90th percentile</th>
      <th>observed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Poisson only (no list)</th>
      <td>17.70</td>
      <td>12.0</td>
      <td>23.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Using the list</th>
      <td>11.95</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

This is one day, and the data is fake, so it is an illustration rather than an evaluation. Notebook 3f shows how to evaluate yet-to-arrive predictions systematically.

## Prediction windows that run past midnight

The list only covers the day it was made for. If the prediction window runs past midnight, the model assumes that no patient on today's list arrives after 24:00, so the list contributes nothing after midnight. Those hours are predicted from the all-elective rates for the next weekday, because tomorrow's list is not yet known.

Below I predict from 20:00 for 16 hours, to 12:00 the next day. In this fake data no listed patient arrives at their listed service after 20:00, so the list contributes nothing, and the off-list and all-elective rates are the same between 20:00 and midnight. After midnight, both predictions use the all-elective rates for Friday. The two means are therefore the same.

```python
common = dict(prediction_time=(20, 0), prediction_window=timedelta(hours=16),
              filter_key='surgical', prediction_date=prediction_date)
print(f"Using the list: {list_model.predict_mean(day_list=day_list, arrivals=arrivals, **common):.2f}")
print(f"Poisson only:   {list_model.predict_mean(**common):.2f}")
```

    Using the list: 14.30
    Poisson only:   14.30

## Services without enough list data

`haem/onc` has no list, and `paediatric` had fewer than `min_list_count` listed patients in training. Both give exactly the Poisson-only result, even when a `day_list` is passed. This means the same `day_list` can be passed for all services, and only services with enough list data use it.

```python
for service in ['paediatric', 'haem/onc']:
    common = dict(prediction_time=(6, 0), prediction_window=prediction_window,
                  filter_key=service, prediction_date=prediction_date)
    with_list_mean = list_model.predict_mean(day_list=day_list, arrivals=arrivals, **common)
    without_list_mean = list_model.predict_mean(**common)
    print(f'{service}: uses list = {list_model.uses_list(service)}; '
          f'mean with day_list = {with_list_mean:.3f}; mean without = {without_list_mean:.3f}')
```

    paediatric: uses list = False; mean with day_list = 2.300; mean without = 2.300
    haem/onc: uses list = False; mean with day_list = 2.400; mean without = 2.400

## Summary

Here I have shown how to use a list of planned admissions to predict elective demand with `ListBasedAdmissionPredictor`:

- At fit time, the model learns when listed patients arrive (`p_k`), how often they arrive at a different service (`r_k`), and the arrival rates of off-list electives.
- At prediction time, it drops listed patients who have already arrived, at any service, and predicts for those still pending. The probability for each pending patient is conditioned on them not having arrived yet.
- The listed and off-list parts are combined into a single distribution, which is narrower than a Poisson-only prediction when the list is a good guide to demand.
- Without a list, or for services with too few listed patients, the model gives exactly the Poisson-only result of `DirectAdmissionPredictor`.

The model fills the same place in the production pipeline as `DirectAdmissionPredictor`. The `build_service_data` function, used in notebook 4c, accepts `elective_day_list` and `elective_arrivals`. Services with list data then get a list-based elective flow, while the rest stay on Poisson.
