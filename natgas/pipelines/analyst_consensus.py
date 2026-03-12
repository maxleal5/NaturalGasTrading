"""
Pipeline 3: Analyst Consensus Estimates

Sources: Barchart OnDemand, Refinitiv, or manual input form fallback.
NatGasWeather.com public estimate as free directional benchmark.
Note: Bloomberg/Reuters are NOT scraped (bot detection).
"""
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BARCHART_API_KEY = os.getenv("BARCHART_API_KEY")
REFINITIV_API_KEY = os.getenv("REFINITIV_API_KEY")


def fetch_barchart_consensus(report_date: date, api_key: Optional[str] = None) -> Optional[dict]:
    """Fetch analyst consensus from Barchart OnDemand API."""
    key = api_key or BARCHART_API_KEY
    if not key:
        logger.warning("BARCHART_API_KEY not set; cannot fetch Barchart consensus.")
        return None
    
    try:
        url = "https://ondemand.websol.barchart.com/getNaturalGasStorageReport.json"
        params = {"apikey": key, "reportDate": report_date.isoformat()}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "report_date": report_date,
            "published_at": datetime.now(timezone.utc),
            "mean_estimate_bcf": float(r.get("meanEstimate", 0)),
            "high_estimate_bcf": float(r.get("highEstimate", 0)),
            "low_estimate_bcf": float(r.get("lowEstimate", 0)),
            "respondent_count": int(r.get("numEstimates", 0)),
            "source_name": "Barchart",
        }
    except Exception as exc:
        logger.error("Barchart consensus fetch failed: %s", exc)
        return None


def record_manual_consensus(
    report_date: date,
    mean_estimate_bcf: float,
    high_estimate_bcf: Optional[float] = None,
    low_estimate_bcf: Optional[float] = None,
    respondent_count: Optional[int] = None,
    source_name: str = "Manual",
) -> dict:
    """Create a consensus record from manual analyst input (fallback)."""
    return {
        "report_date": report_date,
        "published_at": datetime.now(timezone.utc),
        "mean_estimate_bcf": mean_estimate_bcf,
        "high_estimate_bcf": high_estimate_bcf,
        "low_estimate_bcf": low_estimate_bcf,
        "respondent_count": respondent_count,
        "source_name": source_name,
    }


def insert_consensus_record(record: dict, db_session) -> bool:
    """Insert consensus record into analyst_consensus_weekly (append-only)."""
    from sqlalchemy import text
    sql = text("""
        INSERT INTO analyst_consensus_weekly
            (report_date, published_at, mean_estimate_bcf, high_estimate_bcf,
             low_estimate_bcf, respondent_count, source_name)
        VALUES
            (:report_date, :published_at, :mean_estimate_bcf, :high_estimate_bcf,
             :low_estimate_bcf, :respondent_count, :source_name)
        ON CONFLICT (report_date, published_at, source_name) DO NOTHING
    """)
    result = db_session.execute(sql, record)
    return result.rowcount > 0
