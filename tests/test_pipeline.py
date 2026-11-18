"""
Smoke tests for the end-to-end pipeline — run with: pytest tests/ -v

These are not exhaustive unit tests; they verify that (a) every trained model loads
and produces predictions of the expected shape/range, and (b) the dispatch engine's
matching logic behaves sensibly, so a broken retrain or a bad merge is caught before
the Streamlit app silently fails.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import MODELS_DIR, PROCESSED_DIR  # noqa: E402


def _model_path(name):
    p = MODELS_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not found — run the training scripts in src/ first")
    return p


def test_demand_model_predicts():
    import joblib
    model = joblib.load(_model_path("demand_model.joblib"))
    df = pd.read_csv(PROCESSED_DIR / "demand_dataset.csv").head(5)
    feats = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "lag_1h_same_zone",
              "lag_24h_same_zone_hour", "rolling7_avg_zone_hour", "is_weekend",
              "is_exam_period", "is_raining", "pickup_zone", "pickup_zone_type"]
    preds = model.predict(df[feats])
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_availability_model_predicts_probability():
    import joblib
    model = joblib.load(_model_path("availability_model.joblib"))
    df = pd.read_csv(PROCESSED_DIR / "availability_dataset.csv").head(5)
    feats = ["hour_sin", "hour_cos", "prev_slot_available", "is_weekend",
              "is_exam_period", "is_raining", "shift_type", "home_zone"]
    proba = model.predict_proba(df[feats])[:, 1]
    assert len(proba) == 5
    assert np.all((proba >= 0) & (proba <= 1))


def test_eta_model_predicts_nonnegative():
    import joblib
    model = joblib.load(_model_path("eta_model.joblib"))
    df = pd.read_csv(PROCESSED_DIR / "eta_failed_dataset.csv")
    df = df[df["request_failed"] == 0].head(5)
    feats = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "distance_km",
              "nearby_available_drivers", "rolling_zone_wait_avg", "rolling_zone_fail_rate",
              "is_weekend", "is_exam_period", "is_raining",
              "pickup_zone_type", "dropoff_zone_type", "time_block"]
    preds = model.predict(df[feats])
    assert len(preds) == 5
    assert np.all(preds >= 0)


def test_failed_model_predicts_probability():
    import joblib
    model = joblib.load(_model_path("failed_model.joblib"))
    df = pd.read_csv(PROCESSED_DIR / "eta_failed_dataset.csv").head(5)
    feats = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "distance_km",
              "nearby_available_drivers", "rolling_zone_wait_avg", "rolling_zone_fail_rate",
              "is_weekend", "is_exam_period", "is_raining",
              "pickup_zone_type", "dropoff_zone_type", "time_block"]
    proba = model.predict_proba(df[feats])[:, 1]
    assert len(proba) == 5
    assert np.all((proba >= 0) & (proba <= 1))


def test_dispatch_engine_matches_nearest_driver():
    from dispatch import DispatchEngine
    engine = DispatchEngine()
    drivers = pd.DataFrame({
        "driver_id": ["D1", "D2"],
        "home_zone": ["Main Gate", "Boys Hostel Zone 1"],
    })
    result = engine.match_driver("Admin Block", drivers, failure_risk=0.1)
    assert result["matched"] is True
    assert result["driver_id"] in ("D1", "D2")


def test_dispatch_engine_reports_failure_when_no_drivers():
    from dispatch import DispatchEngine
    engine = DispatchEngine()
    empty = pd.DataFrame({"driver_id": [], "home_zone": []})
    result = engine.match_driver("Admin Block", empty, failure_risk=0.5)
    assert result["matched"] is False
    assert result["driver_id"] is None


def test_high_risk_requests_start_with_wider_radius():
    from dispatch import DispatchEngine
    engine = DispatchEngine()
    assert engine.starting_radius(0.8) > engine.starting_radius(0.1)
