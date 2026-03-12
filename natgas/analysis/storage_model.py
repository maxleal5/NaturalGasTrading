"""
Module 1: Storage Draw Estimation Model

Predicts EIA storage number before release using:
- Bias-corrected, seasonally-masked, region-weighted HDD/CDD (prior 7 days)
- LNG exports, dry gas production, industrial demand proxies
- Calendar/seasonal variables
- Ridge Regression (baseline) + XGBoost (primary)

Enforces strict published_at <= training cutoff — no lookahead leakage.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available; using Ridge Regression only.")
    XGBOOST_AVAILABLE = False


GEFS_DIVERGENCE_THRESHOLD_MULTIPLIER = 2.0  # GEFS must diverge >2x Euro/AIFS spread to trigger upweighting


class StorageDrawModel:
    """
    Ensemble storage draw estimation model.
    
    Features: bias-corrected HDD/CDD by region, LNG exports, production,
              seasonal dummies, trend.
    Target: weekly net storage change (Bcf).
    """
    
    REGIONS = ["national", "midwest", "northeast", "texas", "southeast"]
    
    def __init__(self):
        self.ridge_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self.xgb_model = None
        if XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        self._feature_names: list[str] = []
        self._is_fitted = False
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from joined weekly data.
        
        Expected input columns (from weekly_analysis_master or similar join):
        - week_ending_date
        - pop_weighted_hdd_corrected_{region} for each region
        - pop_weighted_cdd_corrected_{region} for each region
        - lng_exports_bcfd
        - dry_gas_production_bcfd
        - residential_commercial_demand_bcfd
        - industrial_demand_bcfd
        - pipeline_exports_mexico_bcfd
        """
        feats = pd.DataFrame(index=df.index)
        
        # HDD/CDD features per region (prior 7-day sum)
        for region in self.REGIONS:
            hdd_col = f"pop_weighted_hdd_corrected_{region}"
            cdd_col = f"pop_weighted_cdd_corrected_{region}"
            if hdd_col in df.columns:
                feats[f"hdd_{region}"] = df[hdd_col].fillna(0)
            if cdd_col in df.columns:
                feats[f"cdd_{region}"] = df[cdd_col].fillna(0)
        
        # Supply/demand fundamentals
        for col in ["lng_exports_bcfd", "dry_gas_production_bcfd",
                    "residential_commercial_demand_bcfd", "industrial_demand_bcfd",
                    "pipeline_exports_mexico_bcfd"]:
            if col in df.columns:
                feats[col] = df[col].fillna(df[col].median() if len(df) > 1 else 0)
        
        # Calendar features
        if "week_ending_date" in df.columns:
            dt = pd.to_datetime(df["week_ending_date"])
            feats["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            feats["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            feats["week_of_year_sin"] = np.sin(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52)
            feats["week_of_year_cos"] = np.cos(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52)
            feats["is_winter"] = dt.dt.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        
        self._feature_names = list(feats.columns)
        return feats
    
    def fit(
        self,
        training_df: pd.DataFrame,
        target_col: str = "net_change_bcf",
        training_cutoff: Optional[date] = None,
    ) -> dict:
        """
        Train both Ridge and XGBoost models on historical data.
        
        Enforces point-in-time integrity: only rows with published_at <= training_cutoff
        are used. This prevents lookahead bias.
        
        Args:
            training_df: DataFrame with features and target. Must have 'published_at' column.
            target_col: Column name for the storage draw target.
            training_cutoff: Maximum published_at date allowed (strict point-in-time).
        
        Returns:
            Dict with cross-validation MAE scores for each model.
        """
        df = training_df.copy()
        
        # Enforce point-in-time cutoff
        if training_cutoff is not None and "published_at" in df.columns:
            cutoff_ts = pd.Timestamp(training_cutoff, tz="UTC")
            df = df[df["published_at"] <= cutoff_ts]
            logger.info("Training on %d rows after point-in-time cutoff %s", len(df), training_cutoff)
        
        if len(df) < 10:
            raise ValueError(f"Insufficient training data: {len(df)} rows (minimum 10)")
        
        X = self.build_features(df)
        y = df[target_col].values
        
        # Remove rows with NaN target
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        
        results = {}
        
        # Ridge Regression
        ridge_scores = cross_val_score(
            self.ridge_pipeline, X, y,
            cv=min(5, len(X)), scoring="neg_mean_absolute_error"
        )
        self.ridge_pipeline.fit(X, y)
        results["ridge_cv_mae"] = float(-ridge_scores.mean())
        logger.info("Ridge CV MAE: %.2f Bcf", results["ridge_cv_mae"])
        
        # XGBoost
        if self.xgb_model is not None:
            xgb_scores = cross_val_score(
                self.xgb_model, X, y,
                cv=min(5, len(X)), scoring="neg_mean_absolute_error"
            )
            self.xgb_model.fit(X, y)
            results["xgb_cv_mae"] = float(-xgb_scores.mean())
            logger.info("XGBoost CV MAE: %.2f Bcf", results["xgb_cv_mae"])
        
        self._is_fitted = True
        return results
    
    def predict(
        self,
        feature_df: pd.DataFrame,
        model_weights: Optional[dict] = None,
    ) -> dict:
        """
        Generate storage draw prediction.
        
        When Euro/AIFS agree but GFS diverges, assign higher weight to Euro/AIFS consensus.
        
        Returns dict with ridge_estimate, xgb_estimate, ensemble_estimate,
                and forecast_uncertainty_score.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self.build_features(feature_df)
        
        ridge_pred = float(self.ridge_pipeline.predict(X)[0])
        results = {"ridge_estimate_bcf": round(ridge_pred, 2)}
        
        if self.xgb_model is not None:
            xgb_pred = float(self.xgb_model.predict(X)[0])
            results["xgb_estimate_bcf"] = round(xgb_pred, 2)
            
            # Default weights: XGBoost primary
            w_xgb = (model_weights or {}).get("xgb", 0.65)
            w_ridge = (model_weights or {}).get("ridge", 0.35)
            ensemble = w_xgb * xgb_pred + w_ridge * ridge_pred
            results["ensemble_estimate_bcf"] = round(ensemble, 2)
            
            # Uncertainty = spread between models
            results["forecast_uncertainty_score"] = round(
                abs(xgb_pred - ridge_pred) / max(abs(ridge_pred), 1.0), 3
            )
        else:
            results["ensemble_estimate_bcf"] = ridge_pred
            results["forecast_uncertainty_score"] = 0.0
        
        return results
    
    def predict_multi_model(
        self,
        euro_df: Optional[pd.DataFrame],
        gefs_df: Optional[pd.DataFrame],
        aifs_df: Optional[pd.DataFrame],
    ) -> dict:
        """
        Generate predictions using each weather model's bias-corrected inputs.
        
        When Euro and AIFS agree but GFS/GEFS diverges (consistent with GFS's known
        volatility per CWG guidance), assign higher weight to Euro/AIFS consensus.
        
        Returns dict with per-model estimates and final weighted ensemble.
        """
        estimates = {}
        
        if euro_df is not None:
            estimates["euro"] = self.predict(euro_df)["ensemble_estimate_bcf"]
        if gefs_df is not None:
            estimates["gefs"] = self.predict(gefs_df)["ensemble_estimate_bcf"]
        if aifs_df is not None:
            estimates["aifs"] = self.predict(aifs_df)["ensemble_estimate_bcf"]
        
        if not estimates:
            return {}
        
        # Consensus weighting: if Euro and AIFS both available and GEFS diverges
        euro_est = estimates.get("euro")
        aifs_est = estimates.get("aifs")
        gefs_est = estimates.get("gefs")
        
        if euro_est is not None and aifs_est is not None and gefs_est is not None:
            euro_aifs_mean = (euro_est + aifs_est) / 2
            gefs_divergence = abs(gefs_est - euro_aifs_mean)
            euro_aifs_spread = abs(euro_est - aifs_est)
            
            if gefs_divergence > GEFS_DIVERGENCE_THRESHOLD_MULTIPLIER * euro_aifs_spread:
                # GEFS diverging — upweight Euro/AIFS consensus
                logger.info(
                    "GFS/GEFS diverging from Euro/AIFS consensus (%.1f Bcf gap). "
                    "Upweighting Euro/AIFS per CWG guidance.",
                    gefs_divergence
                )
                weights = {"euro": 0.40, "aifs": 0.40, "gefs": 0.20}
            else:
                weights = {"euro": 0.35, "aifs": 0.35, "gefs": 0.30}
        elif euro_est is not None:
            weights = {k: 1.0 / len(estimates) for k in estimates}
        else:
            weights = {k: 1.0 / len(estimates) for k in estimates}
        
        weighted_est = sum(estimates[m] * weights.get(m, 0) for m in estimates)
        
        # Uncertainty = std dev across models
        vals = list(estimates.values())
        uncertainty = float(np.std(vals)) if len(vals) > 1 else 0.0
        
        return {
            **{f"{m}_estimate_bcf": v for m, v in estimates.items()},
            "ensemble_estimate_bcf": round(weighted_est, 2),
            "forecast_uncertainty_score": round(uncertainty, 2),
        }
    
    def save(self, path: str) -> None:
        joblib.dump({
            "ridge": self.ridge_pipeline,
            "xgb": self.xgb_model,
            "feature_names": self._feature_names,
        }, path)
        logger.info("StorageDrawModel saved to %s", path)
    
    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.ridge_pipeline = data["ridge"]
        self.xgb_model = data.get("xgb")
        self._feature_names = data.get("feature_names", [])
        self._is_fitted = True
        logger.info("StorageDrawModel loaded from %s", path)
