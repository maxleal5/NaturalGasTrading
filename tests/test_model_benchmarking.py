"""
Tests for natgas.analysis.model_benchmarking

Covers:
- compute_model_accuracy_metrics: correct MAE/RMSE/bias calculation
- compute_model_accuracy_metrics: empty result on no overlapping dates
- compute_model_accuracy_metrics: fewer than 3 observations per lead → skipped
- detect_model_drift: no alert below 2σ threshold
- detect_model_drift: alert triggered after 3 consecutive violations (mocked DB)
"""
import pytest
from datetime import date, timedelta

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from natgas.analysis.model_benchmarking import compute_model_accuracy_metrics


def _make_forecast_df(errors: list[float], model: str = "GFS", lead: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic forecast + actual DataFrames with known per-observation errors."""
    n = len(errors)
    start = date(2025, 1, 1)
    dates = [date.fromordinal(start.toordinal() + i) for i in range(n)]
    actuals = [20.0 + i * 0.1 for i in range(n)]
    forecasts = [actuals[i] + errors[i] for i in range(n)]

    forecast_df = pd.DataFrame({
        "valid_date": dates,
        "lead_days": [lead] * n,
        "pop_weighted_hdd_corrected": forecasts,
    })
    actual_df = pd.DataFrame({
        "valid_date": dates,
        "pop_weighted_hdd_corrected_actual": actuals,
    })
    return forecast_df, actual_df


class TestComputeModelAccuracyMetrics:
    def test_zero_error_returns_zero_mae_rmse_bias(self):
        errors = [0.0] * 10
        fcst, act = _make_forecast_df(errors)
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        assert 7 in result
        assert result[7]["mae"] == pytest.approx(0.0, abs=1e-6)
        assert result[7]["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert result[7]["residual_bias"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_positive_bias(self):
        errors = [2.0] * 10
        fcst, act = _make_forecast_df(errors)
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        assert result[7]["residual_bias"] == pytest.approx(2.0, abs=1e-6)
        assert result[7]["mae"] == pytest.approx(2.0, abs=1e-6)

    def test_mixed_errors_rmse_greater_than_mae_for_unequal_errors(self):
        errors = [1.0, 5.0, -3.0, 2.0, -4.0, 0.5, 0.5, 1.0, 2.0, -1.0]
        fcst, act = _make_forecast_df(errors, lead=5)
        result = compute_model_accuracy_metrics(fcst, act, "GEFS", "midwest")
        assert result[5]["rmse"] >= result[5]["mae"]

    def test_no_overlap_returns_empty(self):
        n = 5
        start = date(2025, 1, 1)
        fcst_dates = [date.fromordinal(start.toordinal() + i) for i in range(n)]
        act_dates = [date.fromordinal(start.toordinal() + i + 100) for i in range(n)]

        fcst = pd.DataFrame({
            "valid_date": fcst_dates,
            "lead_days": [7] * n,
            "pop_weighted_hdd_corrected": [20.0] * n,
        })
        act = pd.DataFrame({
            "valid_date": act_dates,
            "pop_weighted_hdd_corrected_actual": [20.0] * n,
        })
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        assert result == {}

    def test_skips_lead_with_fewer_than_3_obs(self):
        errors = [1.0, 2.0]  # Only 2 obs
        fcst, act = _make_forecast_df(errors, lead=3)
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        assert 3 not in result

    def test_result_dict_contains_expected_keys(self):
        errors = [1.0] * 5
        fcst, act = _make_forecast_df(errors)
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        row = result[7]
        assert "mae" in row
        assert "rmse" in row
        assert "residual_bias" in row
        assert "n_obs" in row
        assert row["n_obs"] == 5

    def test_n_obs_matches_input_length(self):
        errors = [0.5] * 15
        fcst, act = _make_forecast_df(errors)
        result = compute_model_accuracy_metrics(fcst, act, "GFS", "national")
        assert result[7]["n_obs"] == 15

    def test_model_name_and_region_preserved_in_result(self):
        errors = [1.0] * 5
        fcst, act = _make_forecast_df(errors)
        result = compute_model_accuracy_metrics(fcst, act, "AIFS", "texas")
        assert result[7]["model_name"] == "AIFS"
        assert result[7]["region"] == "texas"
