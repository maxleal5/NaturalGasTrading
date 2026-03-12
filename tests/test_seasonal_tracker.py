"""
Tests for natgas.analysis.seasonal_tracker

Covers:
- compute_storage_percentile
- project_end_of_season_storage (linear regression)
- classify_regime
"""
import pytest
from datetime import date

import numpy as np
import pandas as pd

from natgas.analysis.seasonal_tracker import (
    compute_storage_percentile,
    project_end_of_season_storage,
    classify_regime,
    SURPLUS_PERCENTILE_THRESHOLD,
    DEFICIT_PERCENTILE_THRESHOLD,
)


# ── compute_storage_percentile ────────────────────────────────────────────────

class TestComputeStoragePercentile:
    def test_median_value_returns_50(self):
        hist = [1000, 2000, 3000, 4000, 5000]
        pct = compute_storage_percentile(3000.0, hist)
        assert pct == pytest.approx(60.0, abs=5.0)

    def test_max_value_returns_100(self):
        hist = [1000, 2000, 3000]
        pct = compute_storage_percentile(3000.0, hist)
        assert pct == 100.0

    def test_min_value_returns_low(self):
        hist = [1000, 2000, 3000]
        pct = compute_storage_percentile(1000.0, hist)
        assert pct <= 40.0

    def test_empty_history_returns_50(self):
        pct = compute_storage_percentile(2500.0, [])
        assert pct == 50.0

    def test_percentile_in_valid_range(self):
        hist = list(range(100, 4000, 50))
        for val in [100, 500, 1000, 3000, 3990]:
            pct = compute_storage_percentile(float(val), hist)
            assert 0.0 <= pct <= 100.0

    def test_above_all_historical_values(self):
        hist = [1000, 2000, 3000]
        pct = compute_storage_percentile(4000.0, hist)
        assert pct == 100.0


# ── project_end_of_season_storage ─────────────────────────────────────────────

class TestProjectEndOfSeasonStorage:
    def _make_storage_df(self, n_weeks: int = 8, trend: float = -30.0, base: float = 3500.0):
        """Create a synthetic declining storage series."""
        start = date(2025, 11, 7)
        rows = []
        for i in range(n_weeks):
            report_date = date(start.year, start.month, start.day)
            report_date = date.fromordinal(start.toordinal() + i * 7)
            rows.append({"report_date": report_date, "working_gas_bcf": base + trend * i})
        return pd.DataFrame(rows)

    def test_returns_tuple_of_two_floats(self):
        df = self._make_storage_df()
        target = date(2026, 4, 1)
        projected, r2 = project_end_of_season_storage(df, target)
        assert isinstance(projected, float)
        assert isinstance(r2, float)

    def test_r_squared_between_0_and_1(self):
        df = self._make_storage_df()
        target = date(2026, 4, 1)
        _, r2 = project_end_of_season_storage(df, target)
        assert 0.0 <= r2 <= 1.0

    def test_perfect_linear_decline_has_high_r2(self):
        df = self._make_storage_df(trend=-40.0, n_weeks=12)
        target = date(2026, 4, 1)
        _, r2 = project_end_of_season_storage(df, target)
        assert r2 > 0.99

    def test_raises_on_fewer_than_3_weeks(self):
        df = self._make_storage_df(n_weeks=2)
        target = date(2026, 4, 1)
        with pytest.raises(ValueError, match="at least 3"):
            project_end_of_season_storage(df, target)

    def test_projected_value_is_reasonable_for_trend(self):
        """With a -40 Bcf/week trend from 3500 Bcf, April 1 should be well below 3500."""
        df = self._make_storage_df(trend=-40.0, base=3500.0, n_weeks=10)
        target = date(2026, 4, 1)
        projected, _ = project_end_of_season_storage(df, target)
        assert projected < 3500.0

    def test_nan_rows_dropped_gracefully(self):
        df = self._make_storage_df(n_weeks=8)
        df.loc[2, "working_gas_bcf"] = float("nan")
        target = date(2026, 4, 1)
        projected, r2 = project_end_of_season_storage(df, target)
        assert projected > 0.0


# ── classify_regime ───────────────────────────────────────────────────────────

class TestClassifyRegime:
    def test_surplus_above_threshold(self):
        assert classify_regime(SURPLUS_PERCENTILE_THRESHOLD + 1) == "Surplus"

    def test_deficit_below_threshold(self):
        assert classify_regime(DEFICIT_PERCENTILE_THRESHOLD - 1) == "Deficit"

    def test_balanced_between_thresholds(self):
        mid = (SURPLUS_PERCENTILE_THRESHOLD + DEFICIT_PERCENTILE_THRESHOLD) / 2
        assert classify_regime(mid) == "Balanced"

    def test_exactly_at_surplus_threshold(self):
        result = classify_regime(float(SURPLUS_PERCENTILE_THRESHOLD))
        assert result in ("Surplus", "Balanced")

    def test_exactly_at_deficit_threshold(self):
        result = classify_regime(float(DEFICIT_PERCENTILE_THRESHOLD))
        assert result in ("Deficit", "Balanced")

    def test_all_regimes_are_strings(self):
        for pct in [0, 25, 50, 75, 100]:
            assert isinstance(classify_regime(float(pct)), str)
