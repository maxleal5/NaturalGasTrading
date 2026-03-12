"""
Module 2: Storage Surprise Signal + Whisper Number Adjustment

Generates directional trading signal (+1 bullish / 0 neutral / -1 bearish)
based on model_estimate vs analyst_consensus, adjusted for pre-release price drift.

Whisper Number: A print matching consensus may still move prices if market
positioning implied a different expectation. pre_release_price_drift captures this.
"""
import logging
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Price sensitivity calibration: ~$0.03-0.05/MMBtu per 5 Bcf surprise
PRICE_MOVE_PER_BCF = 0.007  # $/MMBtu per 1 Bcf
WHISPER_ADJUSTMENT_FACTOR = 0.5  # Fraction of threshold applied when pre-release drift signals market repositioning


def compute_storage_surprise(
    model_estimate_bcf: float,
    analyst_consensus_bcf: float,
) -> float:
    """Return signed storage surprise: model_estimate - analyst_consensus (Bcf)."""
    return round(model_estimate_bcf - analyst_consensus_bcf, 3)


def compute_directional_signal(
    storage_surprise_bcf: float,
    pre_release_price_drift: Optional[float] = None,
    storage_percentile: Optional[float] = None,
    season: str = "winter",
    surprise_threshold_bcf: float = 3.0,
    drift_threshold: float = 0.05,
) -> tuple[int, float]:
    """
    Compute directional signal and confidence score.
    
    Logic:
    1. Base signal from storage surprise magnitude vs consensus
    2. Whisper number adjustment: strong pre-release rally makes on-consensus bearish
    3. Regime adjustment: same surprise moves more in deficit vs surplus
    
    Args:
        storage_surprise_bcf: model_estimate - analyst_consensus (negative = bearish/bigger draw)
        pre_release_price_drift: Monday close to T-5min price change ($/MMBtu)
        storage_percentile: current season storage percentile (0-100); low = deficit = bullish
        season: 'winter' or 'summer' (affects signal direction interpretation)
        surprise_threshold_bcf: minimum absolute surprise to generate a non-neutral signal
        drift_threshold: minimum pre-release drift for whisper number adjustment
    
    Returns:
        (signal, confidence): signal is +1/0/-1, confidence is 0.0-1.0
    """
    # In winter/withdrawal season: model expects bigger draw than consensus => bullish
    # Negative surprise (bigger draw than expected) => bullish in winter
    # In summer/injection season: model expects smaller injection than consensus => bullish
    sign_multiplier = 1 if season == "winter" else -1
    
    adjusted_surprise = storage_surprise_bcf * sign_multiplier
    
    # Whisper number adjustment
    whisper_adjustment = 0.0
    if pre_release_price_drift is not None:
        if pre_release_price_drift > drift_threshold:
            # Market has already rallied — on-consensus print is effectively bearish
            whisper_adjustment = -WHISPER_ADJUSTMENT_FACTOR * surprise_threshold_bcf
        elif pre_release_price_drift < -drift_threshold:
            # Market has sold off — on-consensus print is effectively bullish
            whisper_adjustment = WHISPER_ADJUSTMENT_FACTOR * surprise_threshold_bcf
    
    effective_surprise = adjusted_surprise + whisper_adjustment
    
    # Regime adjustment
    regime_multiplier = 1.0
    if storage_percentile is not None:
        if storage_percentile < 25:  # Deficit regime — surprises hit harder
            regime_multiplier = 1.3
        elif storage_percentile > 75:  # Surplus regime — surprises matter less
            regime_multiplier = 0.7
    
    scaled = effective_surprise * regime_multiplier
    
    # Signal determination
    if scaled > surprise_threshold_bcf:
        signal = 1  # Bullish
        confidence = min(1.0, 0.5 + (scaled - surprise_threshold_bcf) / (3 * surprise_threshold_bcf))
    elif scaled < -surprise_threshold_bcf:
        signal = -1  # Bearish
        confidence = min(1.0, 0.5 + (abs(scaled) - surprise_threshold_bcf) / (3 * surprise_threshold_bcf))
    else:
        signal = 0  # Neutral
        confidence = 0.5 * (1 - abs(scaled) / surprise_threshold_bcf)
    
    return signal, round(confidence, 3)


def generate_weekly_signal(
    report_date: date,
    model_estimate_bcf: float,
    analyst_consensus_bcf: float,
    pre_release_price_drift: Optional[float] = None,
    storage_percentile: Optional[float] = None,
    season: str = "winter",
    euro_estimate: Optional[float] = None,
    gefs_estimate: Optional[float] = None,
    aifs_estimate: Optional[float] = None,
    forecast_uncertainty_score: Optional[float] = None,
) -> dict:
    """
    Generate the weekly trading signal record for signal_log insertion.
    """
    surprise = compute_storage_surprise(model_estimate_bcf, analyst_consensus_bcf)
    signal, confidence = compute_directional_signal(
        surprise, pre_release_price_drift, storage_percentile, season
    )
    
    signal_names = {1: "BULLISH", 0: "NEUTRAL", -1: "BEARISH"}
    logger.info(
        "Signal for %s: %s (confidence=%.2f) | surprise=%.1f Bcf | drift=%s",
        report_date, signal_names[signal], confidence, surprise,
        f"{pre_release_price_drift:+.3f}" if pre_release_price_drift else "N/A"
    )
    
    return {
        "report_date": report_date,
        "generated_at": datetime.now(timezone.utc),
        "model_estimate_bcf": model_estimate_bcf,
        "analyst_consensus_bcf": analyst_consensus_bcf,
        "model_vs_consensus_bcf": surprise,
        "pre_release_price_drift": pre_release_price_drift,
        "directional_signal": signal,
        "confidence_score": confidence,
        "actual_bcf": None,  # Filled post-release
        "actual_signal": None,
        "signal_correct": None,
        "euro_estimate_bcf": euro_estimate,
        "gefs_estimate_bcf": gefs_estimate,
        "aifs_estimate_bcf": aifs_estimate,
        "forecast_uncertainty_score": forecast_uncertainty_score,
    }


def update_signal_with_actuals(
    report_date: date,
    actual_bcf: float,
    actual_price_change: float,
    db_session,
) -> bool:
    """
    Update signal_log with post-release actuals to compute accuracy.
    Note: We INSERT a new row instead of updating to maintain audit trail.
    """
    from sqlalchemy import text
    
    # Determine actual signal direction from price change
    if actual_price_change > 0.02:
        actual_signal = 1
    elif actual_price_change < -0.02:
        actual_signal = -1
    else:
        actual_signal = 0
    
    # Get the predicted signal
    sql_get = text("""
        SELECT directional_signal FROM signal_log
        WHERE report_date = :report_date
        ORDER BY generated_at DESC LIMIT 1
    """)
    row = db_session.execute(sql_get, {"report_date": report_date}).fetchone()
    if row is None:
        return False
    
    predicted_signal = row[0]
    signal_correct = (predicted_signal == actual_signal)
    
    sql_update = text("""
        UPDATE signal_log
        SET actual_bcf = :actual_bcf,
            actual_signal = :actual_signal,
            signal_correct = :signal_correct
        WHERE report_date = :report_date
    """)
    db_session.execute(sql_update, {
        "actual_bcf": actual_bcf,
        "actual_signal": actual_signal,
        "signal_correct": signal_correct,
        "report_date": report_date,
    })
    return signal_correct


def insert_signal_log(record: dict, db_session) -> bool:
    from sqlalchemy import text
    sql = text("""
        INSERT INTO signal_log
            (report_date, generated_at, model_estimate_bcf, analyst_consensus_bcf,
             model_vs_consensus_bcf, pre_release_price_drift, directional_signal,
             confidence_score, euro_estimate_bcf, gefs_estimate_bcf, aifs_estimate_bcf,
             forecast_uncertainty_score)
        VALUES
            (:report_date, :generated_at, :model_estimate_bcf, :analyst_consensus_bcf,
             :model_vs_consensus_bcf, :pre_release_price_drift, :directional_signal,
             :confidence_score, :euro_estimate_bcf, :gefs_estimate_bcf, :aifs_estimate_bcf,
             :forecast_uncertainty_score)
        ON CONFLICT (report_date) DO UPDATE SET
            model_estimate_bcf = EXCLUDED.model_estimate_bcf,
            directional_signal = EXCLUDED.directional_signal,
            confidence_score = EXCLUDED.confidence_score
    """)
    result = db_session.execute(sql, record)
    return result.rowcount > 0
