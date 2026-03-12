"""
Module 4: Seasonal Surplus/Deficit Tracker

Monitors storage trajectory relative to historical norms.
Projects end-of-withdrawal (April 1) and end-of-injection (November 1) levels.
Classifies regime: Surplus (bearish) / Balanced / Deficit (bullish).
"""
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

WITHDRAWAL_END = (4, 1)   # April 1 (end-of-winter)
INJECTION_END = (11, 1)   # November 1 (end-of-summer)

SURPLUS_PERCENTILE_THRESHOLD = 65   # Above 65th percentile = Surplus (bearish)
DEFICIT_PERCENTILE_THRESHOLD = 35   # Below 35th percentile = Deficit (bullish)


def compute_storage_percentile(
    current_bcf: float,
    historical_values: list[float],
) -> float:
    """
    Compute current storage as a percentile of historical values for the same week.
    
    Returns percentile (0–100). Higher = more storage = bearish.
    """
    if not historical_values:
        return 50.0
    arr = np.array(historical_values)
    pct = float(stats.percentileofscore(arr, current_bcf, kind="rank"))
    return round(pct, 1)


def project_end_of_season_storage(
    storage_series: pd.DataFrame,
    target_date: date,
) -> tuple[float, float]:
    """
    Fit linear trend to current-season storage and project to target_date.
    
    Args:
        storage_series: DataFrame with columns ['report_date', 'working_gas_bcf']
                       for current season only.
        target_date: Projection target (April 1 or November 1).
    
    Returns:
        (projected_bcf, r_squared) — projected storage level and fit quality.
    """
    if len(storage_series) < 3:
        raise ValueError(
            f"Insufficient data in storage_series for projection: "
            f"need at least 3 weeks, got {len(storage_series)}. "
            "Ensure storage_series contains at least 3 weekly observations."
        )
    
    df = storage_series.sort_values("report_date").dropna(subset=["working_gas_bcf"])
    
    # Convert dates to ordinal for regression
    x = np.array([pd.Timestamp(d).toordinal() for d in df["report_date"]])
    y = df["working_gas_bcf"].values
    
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    
    target_ordinal = pd.Timestamp(target_date).toordinal()
    projected = intercept + slope * target_ordinal
    
    return round(float(projected), 1), round(float(r_value ** 2), 3)


def classify_regime(storage_percentile: float) -> str:
    """
    Classify storage regime based on percentile.
    
    Returns: 'Surplus (Bearish)', 'Balanced', or 'Deficit (Bullish)'
    """
    if storage_percentile > SURPLUS_PERCENTILE_THRESHOLD:
        return "Surplus (Bearish)"
    elif storage_percentile < DEFICIT_PERCENTILE_THRESHOLD:
        return "Deficit (Bullish)"
    else:
        return "Balanced"


def generate_seasonal_report(
    current_date: date,
    current_storage_bcf: float,
    storage_series: pd.DataFrame,
    historical_same_week: list[float],
) -> dict:
    """
    Generate complete seasonal trajectory report.
    
    Returns dict with percentile, regime, projections, and narrative.
    """
    # Storage percentile
    percentile = compute_storage_percentile(current_storage_bcf, historical_same_week)
    regime = classify_regime(percentile)
    
    # Determine target date based on current season
    month = current_date.month
    if month in (10, 11, 12, 1, 2, 3):
        season = "withdrawal"
        target_date = date(current_date.year if month <= 3 else current_date.year + 1, 4, 1)
    else:
        season = "injection"
        target_date = date(current_date.year, 11, 1)
    
    # Project end-of-season
    projection_bcf = None
    projection_r2 = None
    if len(storage_series) >= 3:
        try:
            projection_bcf, projection_r2 = project_end_of_season_storage(
                storage_series, target_date
            )
        except Exception as exc:
            logger.warning("Projection failed: %s", exc)
    
    # Narrative
    regime_arrow = "🐂 Bullish" if "Bullish" in regime else ("🐻 Bearish" if "Bearish" in regime else "⚖️ Balanced")
    narrative = (
        f"Storage at {current_storage_bcf:.1f} Bcf is at the {percentile:.0f}th percentile "
        f"for this week — {regime}. "
    )
    if projection_bcf is not None:
        narrative += (
            f"On track to end {season} season at ~{projection_bcf:.0f} Bcf "
            f"(R²={projection_r2:.2f}). {regime_arrow}."
        )
    
    return {
        "report_date": current_date,
        "current_storage_bcf": current_storage_bcf,
        "storage_percentile": percentile,
        "regime": regime,
        "season": season,
        "projected_end_of_season_bcf": projection_bcf,
        "projection_r2": projection_r2,
        "target_date": target_date,
        "narrative": narrative,
    }
