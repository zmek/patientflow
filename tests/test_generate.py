"""Tests for patientflow.generate."""

from __future__ import annotations

import pandas as pd

from patientflow.generate import create_fake_elective_lists, synthesise_departure_times


def test_synthesise_departure_times_ed_visits():
    df = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2024-01-01").date()],
            "prediction_time": [(10, 0)],
            "is_admitted": [True],
        }
    )
    out = synthesise_departure_times(df, kind="ed_visits", seed=0)
    assert "departure_datetime" in out.columns
    assert out["departure_datetime"].notna().all()


def test_synthesise_departure_times_inpatient_arrivals():
    df = pd.DataFrame(
        {
            "arrival_datetime": pd.to_datetime(["2024-01-01 08:00:00"], utc=True),
        }
    )
    out = synthesise_departure_times(df, kind="inpatient_arrivals", seed=0)
    assert "departure_datetime" in out.columns
    assert out["departure_datetime"].notna().all()


def test_create_fake_elective_lists_shapes_and_outcomes():
    arrivals, list_df = create_fake_elective_lists("2024-01-01", "2024-03-01", seed=1)
    assert list(arrivals.columns) == ["arrival_datetime", "encounter", "service"]
    assert list(list_df.columns) == ["encounter", "service", "intended_date"]
    assert arrivals["arrival_datetime"].is_monotonic_increasing
    assert set(list_df["service"]) == {"surgical", "paediatric"}
    assert "haem/onc" in set(arrivals["service"])

    merged = list_df.merge(
        arrivals, on="encounter", how="left", suffixes=("_listed", "")
    )
    arrived = merged.dropna(subset=["arrival_datetime"])
    arrival_dates = arrived["arrival_datetime"].dt.date
    same_day = arrival_dates == arrived["intended_date"]
    other_service = same_day & (arrived["service"] != arrived["service_listed"])
    assert (merged["arrival_datetime"].isna()).any()
    assert (arrival_dates < arrived["intended_date"]).any()
    assert (arrival_dates > arrived["intended_date"]).any()
    assert other_service.any()


def test_create_fake_elective_lists_rejects_bad_outcome_probs():
    import pytest

    with pytest.raises(ValueError):
        create_fake_elective_lists(
            "2024-01-01",
            "2024-01-08",
            outcome_probs={
                "arrives_at_listed_service": 0.5,
                "arrives_at_other_service": 0.1,
                "arrives_evening_before": 0.1,
                "deferred_to_next_day": 0.1,
                "does_not_arrive": 0.1,
            },
        )
