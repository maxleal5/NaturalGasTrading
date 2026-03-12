"""
Bias Correction (MOS) Layer — Applied before all analysis modules.

Maintains rolling 30-day bias: bias(model, region, lead_days) = mean(forecast_HDD - actual_HDD)
Detects model drift via 2-sigma alert on residual bias.

AI models (AIFS, GraphCast) can have silent architecture updates — this module
catches them within 3 consecutive anomalous model runs.
"""
import logging
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_SIGMA_THRESHOLD = 2.0      # Z-score threshold for drift alert
DRIFT_CONSECUTIVE_RUNS = 3       # Consecutive violations before alert
NORMAL_MOS_WINDOW_DAYS = 30
FAST_MOS_WINDOW_DAYS = 7         # Narrowed window after drift detected


class BiasCorrector:
    """
    Rolling MOS bias corrector with drift detection.
    
    Usage:
        corrector = BiasCorrector(db_session)
        corrected_hdd = corrector.correct_hdd(raw_hdd, model, region, lead_days, forecast_date)
    """
    
    def __init__(self, db_session=None, mos_window_days: int = NORMAL_MOS_WINDOW_DAYS):
        self.db = db_session
        self.mos_window_days = mos_window_days
        self._bias_cache: dict = {}  # (model, region, lead_days) -> (bias_hdd, bias_cdd)
    
    def compute_rolling_bias(
        self,
        model_name: str,
        region: str,
        lead_days: int,
        as_of_date: Optional[date] = None,
        window_days: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Compute rolling mean bias (forecast - actual) for HDD and CDD.
        
        bias(model, region, lead_days) = mean(forecast_HDD - actual_HDD) over rolling window
        
        Returns (bias_hdd, bias_cdd). Returns (0.0, 0.0) if insufficient history.
        """
        if self.db is None:
            return 0.0, 0.0
        
        window = window_days or self.mos_window_days
        if as_of_date is None:
            as_of_date = date.today()
        
        from sqlalchemy import text
        
        # Join forecast vs actual HDD/CDD (actuals from NOAA GHCN station obs stored as lead_days=0)
        sql = text("""
            SELECT
                AVG(f.pop_weighted_hdd - a.pop_weighted_hdd) AS bias_hdd,
                AVG(f.pop_weighted_cdd - a.pop_weighted_cdd) AS bias_cdd
            FROM hdd_cdd_daily_by_model f
            JOIN hdd_cdd_daily_by_model a
                ON a.valid_date = f.valid_date
               AND a.region = f.region
               AND a.model_name = 'ACTUAL'
               AND a.lead_days = 0
            WHERE f.model_name = :model
              AND f.region = :region
              AND f.lead_days = :lead_days
              AND f.valid_date BETWEEN :start_date AND :end_date
        """)
        
        start_date = as_of_date - pd.Timedelta(days=window)
        result = self.db.execute(sql, {
            "model": model_name,
            "region": region,
            "lead_days": lead_days,
            "start_date": start_date.date() if hasattr(start_date, "date") else start_date,
            "end_date": as_of_date,
        }).fetchone()
        
        if result and result[0] is not None:
            return float(result[0]), float(result[1] or 0.0)
        return 0.0, 0.0
    
    def correct_hdd_cdd(
        self,
        raw_hdd: float,
        raw_cdd: float,
        model_name: str,
        region: str,
        lead_days: int,
        as_of_date: Optional[date] = None,
    ) -> tuple[float, float, float, float]:
        """
        Apply MOS bias correction to raw HDD/CDD values.
        
        Returns (corrected_hdd, corrected_cdd, bias_hdd, bias_cdd).
        Raw values are never overwritten — returned alongside corrected values.
        """
        cache_key = (model_name, region, lead_days)
        if cache_key not in self._bias_cache:
            bias_hdd, bias_cdd = self.compute_rolling_bias(
                model_name, region, lead_days, as_of_date
            )
            self._bias_cache[cache_key] = (bias_hdd, bias_cdd)
        else:
            bias_hdd, bias_cdd = self._bias_cache[cache_key]
        
        corrected_hdd = max(0.0, raw_hdd - bias_hdd)
        corrected_cdd = max(0.0, raw_cdd - bias_cdd)
        return corrected_hdd, corrected_cdd, bias_hdd, bias_cdd
    
    def check_model_stability(
        self,
        model_name: str,
        region: str,
        lead_days: int,
        current_residual_bias: float,
        db_session=None,
    ) -> dict:
        """
        Check for model drift using 2-sigma rule on residual bias.
        
        Computes rolling std of residual bias over 30 prior runs.
        Triggers alert if |residual_bias| > 2σ for 3 consecutive runs.
        
        Returns dict with alert status, z_score, and recommended action.
        """
        sess = db_session or self.db
        if sess is None:
            return {"alert": False, "consecutive": 0, "z_score": 0.0}
        
        from sqlalchemy import text
        
        # Get last 30 residual bias values for this model/region/lead
        sql = text("""
            SELECT residual_bias
            FROM model_stability_log
            WHERE model_name = :model
              AND region = :region
              AND lead_days = :lead_days
            ORDER BY detected_at DESC
            LIMIT 30
        """)
        rows = sess.execute(sql, {
            "model": model_name, "region": region, "lead_days": lead_days
        }).fetchall()
        
        history = [float(r[0]) for r in rows if r[0] is not None]
        
        if len(history) < 5:
            # Not enough history to detect drift
            return {"alert": False, "consecutive": 0, "z_score": 0.0, "std": None}
        
        std = float(np.std(history))
        if std == 0:
            std = 1e-6
        
        z_score = abs(current_residual_bias) / std
        
        # Count consecutive violations (most recent first)
        consecutive = 0
        for i, rb in enumerate(history):
            if abs(rb) / std > DRIFT_SIGMA_THRESHOLD:
                consecutive += 1
            else:
                break
        
        # Add current run
        if z_score > DRIFT_SIGMA_THRESHOLD:
            consecutive += 1
        
        alert = consecutive >= DRIFT_CONSECUTIVE_RUNS
        
        if alert:
            logger.warning(
                "MODEL DRIFT DETECTED: %s region=%s lead=%dd "
                "residual_bias=%.4f z=%.2f consecutive=%d",
                model_name, region, lead_days, current_residual_bias, z_score, consecutive
            )
        
        return {
            "alert": alert,
            "consecutive": consecutive,
            "z_score": z_score,
            "std": std,
            "residual_bias": current_residual_bias,
            "recommended_window": FAST_MOS_WINDOW_DAYS if alert else NORMAL_MOS_WINDOW_DAYS,
        }
    
    def insert_stability_log(
        self,
        model_name: str,
        region: str,
        lead_days: int,
        residual_bias: float,
        bias_std_30d: float,
        z_score: float,
        consecutive: int,
        alert_triggered: bool,
        mos_window_adjusted: bool,
        db_session=None,
        notes: str = None,
    ) -> None:
        """Log a stability check result to model_stability_log."""
        sess = db_session or self.db
        if sess is None:
            return
        
        from sqlalchemy import text
        
        sql = text("""
            INSERT INTO model_stability_log
                (model_name, region, lead_days, residual_bias, bias_std_30d,
                 z_score, consecutive_violations, alert_triggered, mos_window_adjusted, notes)
            VALUES
                (:model_name, :region, :lead_days, :residual_bias, :bias_std_30d,
                 :z_score, :consecutive_violations, :alert_triggered, :mos_window_adjusted, :notes)
        """)
        sess.execute(sql, {
            "model_name": model_name,
            "region": region,
            "lead_days": lead_days,
            "residual_bias": residual_bias,
            "bias_std_30d": bias_std_30d,
            "z_score": z_score,
            "consecutive_violations": consecutive,
            "alert_triggered": alert_triggered,
            "mos_window_adjusted": mos_window_adjusted,
            "notes": notes,
        })
    
    def run_bias_correction_pipeline(
        self,
        model_name: str,
        region: str,
        forecast_date: date,
        lead_days_range: list[int],
        raw_records: list[dict],
        db_session=None,
    ) -> list[dict]:
        """
        Apply bias correction to a batch of raw HDD/CDD records and insert corrected values.
        
        Also runs stability check and logs results.
        Returns list of corrected records for hdd_cdd_bias_corrected_by_model.
        """
        sess = db_session or self.db
        corrected = []
        
        for rec in raw_records:
            lead = rec.get("lead_days", 0)
            raw_hdd = float(rec.get("pop_weighted_hdd", 0) or 0)
            raw_cdd = float(rec.get("pop_weighted_cdd", 0) or 0)
            
            cor_hdd, cor_cdd, bias_h, bias_c = self.correct_hdd_cdd(
                raw_hdd, raw_cdd, model_name, region, lead, forecast_date
            )
            
            corrected.append({
                "forecast_date": forecast_date,
                "valid_date": rec.get("valid_date"),
                "lead_days": lead,
                "model_name": model_name,
                "model_version": rec.get("model_version"),
                "region": region,
                "seasonal_mask": rec.get("seasonal_mask", "winter"),
                "pop_weighted_hdd_raw": raw_hdd,
                "pop_weighted_cdd_raw": raw_cdd,
                "bias_hdd": bias_h,
                "bias_cdd": bias_c,
                "pop_weighted_hdd_corrected": cor_hdd,
                "pop_weighted_cdd_corrected": cor_cdd,
            })
        
        if sess and corrected:
            self._insert_corrected_records(corrected, sess)
        
        return corrected
    
    def _insert_corrected_records(self, records: list[dict], db_session) -> int:
        from sqlalchemy import text
        sql = text("""
            INSERT INTO hdd_cdd_bias_corrected_by_model
                (forecast_date, valid_date, lead_days, model_name, model_version,
                 region, seasonal_mask, pop_weighted_hdd_raw, pop_weighted_cdd_raw,
                 bias_hdd, bias_cdd, pop_weighted_hdd_corrected, pop_weighted_cdd_corrected)
            VALUES
                (:forecast_date, :valid_date, :lead_days, :model_name, :model_version,
                 :region, :seasonal_mask, :pop_weighted_hdd_raw, :pop_weighted_cdd_raw,
                 :bias_hdd, :bias_cdd, :pop_weighted_hdd_corrected, :pop_weighted_cdd_corrected)
            ON CONFLICT DO NOTHING
        """)
        inserted = 0
        for rec in records:
            result = db_session.execute(sql, rec)
            inserted += result.rowcount
        return inserted
