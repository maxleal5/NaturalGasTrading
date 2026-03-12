"""
Pipeline 4: Natural Gas Futures Prices

Sources: CME DataMine, Barchart, or Nasdaq Data Link
Cadence: Daily close + intraday on release days
"""
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BARCHART_API_KEY = os.getenv("BARCHART_API_KEY")
NGAS_FRONT_MONTH_SYMBOL = "NG*1"
NGAS_CONTINUOUS_SYMBOL = "NGZ{year}"


def fetch_daily_settlement(
    trade_date: date,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """
    Fetch natural gas futures daily settlement from Barchart OnDemand.
    Returns front-month settle and 12-month strip average.
    """
    key = api_key or BARCHART_API_KEY
    if not key:
        logger.warning("BARCHART_API_KEY not set; cannot fetch futures prices.")
        return None
    
    try:
        url = "https://ondemand.websol.barchart.com/getQuote.json"
        params = {
            "apikey": key,
            "symbols": NGAS_FRONT_MONTH_SYMBOL,
            "fields": "lastPrice,volume,openInterest",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "trade_date": trade_date,
            "front_month_settle": float(r.get("lastPrice", 0)),
            "twelve_month_strip": None,  # Requires fetching 12 contract months
            "open_interest": int(r.get("openInterest", 0) or 0),
            "volume": int(r.get("volume", 0) or 0),
            "data_source": "Barchart",
        }
    except Exception as exc:
        logger.error("Futures price fetch failed: %s", exc)
        return None


def record_release_day_intraday(
    trade_date: date,
    report_date: date,
    price_t_minus_5min: Optional[float],
    price_t_plus_1min: Optional[float],
    price_t_plus_5min: Optional[float],
    price_t_plus_15min: Optional[float],
    monday_close: Optional[float],
) -> dict:
    """
    Record intraday price snapshots around EIA release.
    pre_release_drift = price at T-5min minus Monday close.
    """
    drift = None
    if price_t_minus_5min is not None and monday_close is not None:
        drift = round(price_t_minus_5min - monday_close, 4)
    
    return {
        "trade_date": trade_date,
        "report_date": report_date,
        "price_t_minus_5min": price_t_minus_5min,
        "price_t_plus_1min": price_t_plus_1min,
        "price_t_plus_5min": price_t_plus_5min,
        "price_t_plus_15min": price_t_plus_15min,
        "pre_release_drift": drift,
        "data_source": "Manual",
    }


def insert_daily_settlement(record: dict, db_session) -> bool:
    from sqlalchemy import text
    sql = text("""
        INSERT INTO ngas_futures_daily
            (trade_date, front_month_settle, twelve_month_strip, open_interest, volume, data_source)
        VALUES
            (:trade_date, :front_month_settle, :twelve_month_strip, :open_interest, :volume, :data_source)
        ON CONFLICT (trade_date) DO NOTHING
    """)
    result = db_session.execute(sql, record)
    return result.rowcount > 0


def insert_intraday_record(record: dict, db_session) -> bool:
    from sqlalchemy import text
    sql = text("""
        INSERT INTO ngas_futures_release_day_intraday
            (trade_date, report_date, price_t_minus_5min, price_t_plus_1min,
             price_t_plus_5min, price_t_plus_15min, pre_release_drift, data_source)
        VALUES
            (:trade_date, :report_date, :price_t_minus_5min, :price_t_plus_1min,
             :price_t_plus_5min, :price_t_plus_15min, :pre_release_drift, :data_source)
        ON CONFLICT (trade_date, report_date) DO NOTHING
    """)
    result = db_session.execute(sql, record)
    return result.rowcount > 0
