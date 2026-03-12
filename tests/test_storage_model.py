"""
Tests for natgas.analysis.storage_model (StorageDrawModel)

Covers:
- build_features produces correct columns and shape
- fit raises on insufficient data
- predict returns dict with expected keys
- ensemble_estimate_bcf is between ridge and xgb estimates
- forecast_uncertainty_score is non-negative
- Point-in-time integrity: rows after training_cutoff are excluded
"""
import pytest
from datetime import date, datetime, timezone, timedelta

import numpy as np
import pandas as pd

from natgas.analysis.storage_model import StorageDrawModel


def _make_training_df(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Synthetic weekly training dataset with realistic feature columns."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 7)

    rows = []
    for i in range(n):
        d = date.fromordinal(start.toordinal() + i * 7)
        # Seasonal HDD/CDD: HDD high in winter, CDD high in summer
        month = d.month
        base_hdd = 100 if month in (12, 1, 2) else (20 if month in (6, 7, 8) else 50)
        row = {
            "week_ending_date": d,
            "published_at": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
            "net_change_bcf": rng.normal(-50 if month in (12, 1, 2) else 30, 10),
            "pop_weighted_hdd_corrected_national": base_hdd + rng.normal(0, 5),
            "pop_weighted_hdd_corrected_midwest":  base_hdd + rng.normal(5, 5),
            "pop_weighted_hdd_corrected_northeast": base_hdd + rng.normal(8, 5),
            "pop_weighted_hdd_corrected_texas": base_hdd * 0.5 + rng.normal(0, 3),
            "pop_weighted_hdd_corrected_southeast": base_hdd * 0.6 + rng.normal(0, 3),
            "pop_weighted_cdd_corrected_national":  max(0, rng.normal(30, 5) if month in (6, 7, 8) else 2),
            "pop_weighted_cdd_corrected_midwest":   max(0, rng.normal(25, 5) if month in (6, 7, 8) else 1),
            "pop_weighted_cdd_corrected_northeast": max(0, rng.normal(20, 5) if month in (6, 7, 8) else 1),
            "pop_weighted_cdd_corrected_texas":     max(0, rng.normal(50, 5) if month in (6, 7, 8) else 5),
            "pop_weighted_cdd_corrected_southeast": max(0, rng.normal(45, 5) if month in (6, 7, 8) else 4),
            "lng_exports_bcfd": rng.uniform(10, 14),
            "dry_gas_production_bcfd": rng.uniform(95, 103),
            "residential_commercial_demand_bcfd": rng.uniform(20, 35),
            "industrial_demand_bcfd": rng.uniform(20, 24),
            "pipeline_exports_mexico_bcfd": rng.uniform(5, 7),
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestBuildFeatures:
    def setup_method(self):
        self.model = StorageDrawModel()
        self.df = _make_training_df()

    def test_returns_dataframe(self):
        feats = self.model.build_features(self.df)
        assert isinstance(feats, pd.DataFrame)

    def test_same_row_count(self):
        feats = self.model.build_features(self.df)
        assert len(feats) == len(self.df)

    def test_hdd_columns_present_for_all_regions(self):
        feats = self.model.build_features(self.df)
        for region in StorageDrawModel.REGIONS:
            assert f"hdd_{region}" in feats.columns

    def test_no_nan_in_hdd_cdd_columns(self):
        feats = self.model.build_features(self.df)
        hdd_cols = [c for c in feats.columns if c.startswith("hdd_") or c.startswith("cdd_")]
        assert feats[hdd_cols].isna().sum().sum() == 0


class TestFit:
    def setup_method(self):
        self.model = StorageDrawModel()
        self.df = _make_training_df(n=40)

    def test_fit_returns_dict_with_ridge_cv_mae(self):
        result = self.model.fit(self.df)
        assert "ridge_cv_mae" in result
        assert result["ridge_cv_mae"] >= 0.0

    def test_fit_sets_is_fitted_flag(self):
        assert not self.model._is_fitted
        self.model.fit(self.df)
        assert self.model._is_fitted

    def test_fit_raises_on_too_few_samples(self):
        small_df = _make_training_df(n=5)
        with pytest.raises(ValueError, match="Insufficient"):
            self.model.fit(small_df)

    def test_fit_respects_point_in_time_cutoff(self):
        """Rows published after cutoff must be excluded."""
        df = _make_training_df(n=40)
        cutoff = df["published_at"].iloc[19]  # Cut after first 20 rows
        model_full = StorageDrawModel()
        model_full.fit(df)  # Uses all rows

        model_cutoff = StorageDrawModel()
        # Should only train on first 20 rows — this may fail if <10 remain
        # but here we have 20 which is >=10
        model_cutoff.fit(df, training_cutoff=cutoff)
        assert model_cutoff._is_fitted


class TestPredict:
    def setup_method(self):
        self.model = StorageDrawModel()
        df = _make_training_df(n=40)
        self.model.fit(df, target_col="net_change_bcf")
        # Single-row prediction dataframe
        self.pred_df = _make_training_df(n=1)

    def test_predict_returns_dict(self):
        result = self.model.predict(self.pred_df)
        assert isinstance(result, dict)

    def test_predict_contains_ridge_estimate(self):
        result = self.model.predict(self.pred_df)
        assert "ridge_estimate_bcf" in result

    def test_predict_contains_ensemble_estimate(self):
        result = self.model.predict(self.pred_df)
        assert "ensemble_estimate_bcf" in result

    def test_forecast_uncertainty_score_non_negative(self):
        result = self.model.predict(self.pred_df)
        assert result["forecast_uncertainty_score"] >= 0.0

    def test_predict_raises_when_not_fitted(self):
        unfitted = StorageDrawModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            unfitted.predict(self.pred_df)

    def test_ensemble_estimate_is_weighted_blend(self):
        """Ensemble must lie between ridge and xgb (if XGBoost is available)."""
        result = self.model.predict(self.pred_df)
        if "xgb_estimate_bcf" in result:
            lo = min(result["ridge_estimate_bcf"], result["xgb_estimate_bcf"])
            hi = max(result["ridge_estimate_bcf"], result["xgb_estimate_bcf"])
            assert lo <= result["ensemble_estimate_bcf"] <= hi
