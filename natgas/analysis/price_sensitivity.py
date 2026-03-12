"""
Module 5: Price Sensitivity Analysis

Panel regression: Δprice ~ storage_surprise × storage_percentile × season × pre_release_momentum

Core hypothesis: Same 10 Bcf miss moves prices significantly more in deficit regime
than surplus regime — interaction term captures this.
Retrain quarterly as new observations accumulate.
"""
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import joblib

logger = logging.getLogger(__name__)

# Price sensitivity calibration: ~$0.03-0.05/MMBtu per 5 Bcf surprise
PRICE_MOVE_PER_BCF = 0.007  # $/MMBtu per 1 Bcf surprise (calibrated: $0.035/5Bcf)


class PriceSensitivityModel:
    """
    Panel regression model for price sensitivity analysis.
    
    Features:
    - storage_surprise_bcf: actual - consensus (signed)
    - storage_percentile: 0-100 (low = deficit = bullish)
    - is_winter: seasonal dummy
    - pre_release_momentum: pre_release_price_drift (whisper number proxy)
    - Interaction terms: surprise × percentile, surprise × season, surprise × momentum
    """
    
    def __init__(self):
        self.ols_model = None
        self.ridge_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=0.5)),
        ])
        self._is_fitted = False
        self.summary = None
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix with interaction terms."""
        feats = pd.DataFrame(index=df.index)
        
        # Base features
        feats["storage_surprise"] = df["storage_surprise_bcf"].fillna(0)
        feats["storage_percentile_scaled"] = (df.get("storage_percentile", 50.0).fillna(50) - 50) / 50
        feats["is_winter"] = df.get("is_winter", 0).fillna(0).astype(float)
        feats["pre_release_momentum"] = df.get("pre_release_price_drift", 0.0).fillna(0)
        
        # Interaction terms (core hypothesis)
        feats["surprise_x_percentile"] = feats["storage_surprise"] * feats["storage_percentile_scaled"]
        feats["surprise_x_season"] = feats["storage_surprise"] * feats["is_winter"]
        feats["surprise_x_momentum"] = feats["storage_surprise"] * feats["pre_release_momentum"]
        
        # Second-order interaction
        feats["surprise_x_pct_x_season"] = (
            feats["storage_surprise"] * feats["storage_percentile_scaled"] * feats["is_winter"]
        )
        
        return feats
    
    def fit(
        self,
        training_df: pd.DataFrame,
        price_change_col: str = "price_t_plus_15min_vs_t_minus_5min",
    ) -> dict:
        """
        Fit OLS panel regression on historical release-day data.
        
        Args:
            training_df: DataFrame with features and price change target.
            price_change_col: Column representing price change around release.
        
        Returns dict with model coefficients, R², and interpretation.
        """
        df = training_df.dropna(subset=["storage_surprise_bcf", price_change_col])
        
        if len(df) < 15:
            raise ValueError(
                f"Insufficient training data: got {len(df)} observations, need at least 15. "
                "Collect more historical release-day data before fitting the model."
            )
        
        X = self.build_features(df)
        y = df[price_change_col].values
        
        # OLS with statsmodels for interpretable coefficients + significance
        X_with_const = sm.add_constant(X, has_constant="add")
        self.ols_model = sm.OLS(y, X_with_const).fit(cov_type="HC3")
        self.summary = self.ols_model.summary()
        
        # Also fit Ridge for more stable predictions when X is collinear
        self.ridge_pipeline.fit(X, y)
        self._is_fitted = True
        
        logger.info("Price sensitivity OLS R²=%.3f, Ridge fitted on %d obs",
                    self.ols_model.rsquared, len(df))
        
        return {
            "r_squared": round(float(self.ols_model.rsquared), 4),
            "n_obs": len(df),
            "coefficients": {
                name: round(float(coef), 5)
                for name, coef in zip(X_with_const.columns, self.ols_model.params)
            },
            "p_values": {
                name: round(float(pval), 4)
                for name, pval in zip(X_with_const.columns, self.ols_model.pvalues)
            },
        }
    
    def predict_price_impact(
        self,
        storage_surprise_bcf: float,
        storage_percentile: float = 50.0,
        is_winter: bool = True,
        pre_release_momentum: float = 0.0,
    ) -> dict:
        """
        Predict expected price move given a storage surprise.
        
        Quantifies regime-adjusted price sensitivity:
        - Same surprise moves more in deficit (low percentile) vs surplus (high percentile)
        """
        if not self._is_fitted:
            # Fallback to calibration heuristic: $0.03-0.05/MMBtu per 5 Bcf
            base_move = storage_surprise_bcf * PRICE_MOVE_PER_BCF
            return {"predicted_price_change": round(base_move, 4), "method": "calibration_heuristic"}
        
        feature_df = pd.DataFrame([{
            "storage_surprise_bcf": storage_surprise_bcf,
            "storage_percentile": storage_percentile,
            "is_winter": int(is_winter),
            "pre_release_price_drift": pre_release_momentum,
        }])
        
        X = self.build_features(feature_df)
        ridge_pred = float(self.ridge_pipeline.predict(X)[0])
        
        X_const = sm.add_constant(X, has_constant="add")
        ols_pred = float(self.ols_model.predict(X_const)[0])
        
        # Blend OLS (interpretable) with Ridge (stable)
        blended = 0.5 * ols_pred + 0.5 * ridge_pred
        
        return {
            "predicted_price_change": round(blended, 4),
            "ols_prediction": round(ols_pred, 4),
            "ridge_prediction": round(ridge_pred, 4),
            "regime": "deficit" if storage_percentile < 35 else ("surplus" if storage_percentile > 65 else "balanced"),
            "method": "panel_regression",
        }
    
    def save(self, path: str) -> None:
        joblib.dump({
            "ols_model": self.ols_model,
            "ridge_pipeline": self.ridge_pipeline,
        }, path)
    
    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.ols_model = data["ols_model"]
        self.ridge_pipeline = data["ridge_pipeline"]
        self._is_fitted = True
