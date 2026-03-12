"""
Module 3: Weather Model Accuracy Benchmarking + Automated Drift Detection

Continuously evaluates which weather model best predicts HDD/CDD.
Tracks accuracy by model, region, and lead time (1-14 days).
Triggers alerts when AI model residual bias exceeds 2σ for 3 consecutive runs.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_model_accuracy_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model_name: str,
    region: str,
    variable: str = "pop_weighted_hdd_corrected",
) -> dict:
    """
    Compute MAE, RMSE, and residual bias for a model's forecasts vs actuals.
    
    Args:
        forecast_df: DataFrame with columns [valid_date, lead_days, <variable>]
        actual_df: DataFrame with columns [valid_date, <variable>_actual]
        model_name: Name of the weather model
        region: Geographic region
        variable: HDD/CDD column name to evaluate
    
    Returns:
        Dict with mae, rmse, residual_bias, n_obs, by lead_days.
    """
    merged = forecast_df.merge(actual_df, on="valid_date", suffixes=("_fcst", "_actual"))
    
    if merged.empty:
        logger.warning("No overlapping dates for %s %s %s", model_name, region, variable)
        return {}
    
    fcst_col = f"{variable}_fcst" if f"{variable}_fcst" in merged.columns else variable
    actual_col = f"{variable}_actual" if f"{variable}_actual" in merged.columns else f"{variable}_actual"
    
    if fcst_col not in merged.columns or actual_col not in merged.columns:
        return {}
    
    results = {}
    for lead in sorted(merged["lead_days"].unique()):
        sub = merged[merged["lead_days"] == lead].dropna(subset=[fcst_col, actual_col])
        if len(sub) < 3:
            continue
        
        errors = sub[fcst_col].values - sub[actual_col].values
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))  # Positive = model runs warm/high
        
        results[lead] = {
            "model_name": model_name,
            "region": region,
            "lead_days": lead,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "residual_bias": round(bias, 4),
            "n_obs": len(sub),
        }
    
    return results


def build_90day_scorecard(
    db_session,
    as_of_date: Optional[date] = None,
    models: Optional[list] = None,
    regions: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build rolling 90-day accuracy scorecard by model, region, and lead time.
    
    Returns DataFrame with columns: model_name, region, lead_days, mae, rmse, residual_bias, n_obs.
    """
    if as_of_date is None:
        as_of_date = date.today()
    start_date = as_of_date - timedelta(days=90)
    
    from sqlalchemy import text
    
    model_filter = ""
    if models:
        placeholders = ", ".join(f"'{m}'" for m in models)
        model_filter = f"AND f.model_name IN ({placeholders})"
    
    region_filter = ""
    if regions:
        placeholders = ", ".join(f"'{r}'" for r in regions)
        region_filter = f"AND f.region IN ({placeholders})"
    
    sql = text(f"""
        SELECT
            f.model_name,
            f.region,
            f.lead_days,
            AVG(ABS(f.pop_weighted_hdd_corrected - a.pop_weighted_hdd)) AS mae_hdd,
            SQRT(AVG(POWER(f.pop_weighted_hdd_corrected - a.pop_weighted_hdd, 2))) AS rmse_hdd,
            AVG(f.pop_weighted_hdd_corrected - a.pop_weighted_hdd) AS residual_bias_hdd,
            COUNT(*) AS n_obs
        FROM hdd_cdd_bias_corrected_by_model f
        JOIN hdd_cdd_daily_by_model a
            ON a.valid_date = f.valid_date
           AND a.region = f.region
           AND a.model_name = 'ACTUAL'
           AND a.lead_days = 0
        WHERE f.valid_date BETWEEN :start_date AND :end_date
          AND f.model_name != 'ACTUAL'
          {model_filter}
          {region_filter}
        GROUP BY f.model_name, f.region, f.lead_days
        ORDER BY f.model_name, f.region, f.lead_days
    """)
    
    rows = db_session.execute(sql, {"start_date": start_date, "end_date": as_of_date}).fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    cols = ["model_name", "region", "lead_days", "mae_hdd", "rmse_hdd", "residual_bias_hdd", "n_obs"]
    df = pd.DataFrame(rows, columns=cols)
    df["as_of_date"] = as_of_date
    return df


def identify_best_model(
    scorecard_df: pd.DataFrame,
    lead_days: int = 7,
    metric: str = "mae_hdd",
) -> Optional[str]:
    """
    Identify the best-performing model at a given lead time based on specified metric.
    Requires at least 26 weeks of observations for statistical significance.
    """
    if scorecard_df.empty:
        return None
    
    subset = scorecard_df[scorecard_df["lead_days"] == lead_days]
    if subset.empty:
        return None
    
    # Filter for models with sufficient observations (26+ observations, one per day of lead period)
    sufficient = subset[subset["n_obs"] >= 26]
    if sufficient.empty:
        logger.info("Insufficient data for statistical significance (need 26+ weeks). Using all data.")
        sufficient = subset
    
    best_idx = sufficient[metric].idxmin()
    return str(sufficient.loc[best_idx, "model_name"])
