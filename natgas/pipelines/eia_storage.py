"""
Pipeline 1: EIA Natural Gas Storage Data Ingestion

- Source: EIA API (api.eia.gov) — Natural Gas Weekly Storage Report
- Retry: Exponential backoff, every 5s for up to 2 minutes
- Point-in-time integrity: append-only, published_at on every row
- Revision detection: logs all changes to data_revision_log
"""
import logging
import os
import time as time_module
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

import requests
from tenacity import retry, stop_after_delay, wait_exponential, before_sleep_log

logger = logging.getLogger(__name__)

EIA_BASE_URL = os.getenv("EIA_BASE_URL", "https://api.eia.gov/v2/")
EIA_SERIES_MAP = {
    "total": "NW2",
    "east": "NW2-DCR-SND-1",
    "midwest": "NW2-DCR-SND-3",
    "mountain": "NW2-DCR-SND-5",
    "pacific": "NW2-DCR-SND-7",
    "south_central": "NW2-DCR-SND-9",
}


@retry(
    stop=stop_after_delay(120),  # 2 minutes
    wait=wait_exponential(multiplier=1, min=5, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_eia_storage_raw(api_key: str, series_id: str, start_date: str, end_date: str) -> dict:
    """Fetch raw EIA storage data with exponential backoff retry."""
    url = f"{EIA_BASE_URL}natural-gas/stor/wkly/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start_date,
        "end": end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 52,
        "offset": 0,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError(f"Unexpected EIA API response structure: {list(data.keys())}")
    return data


def fetch_weekly_storage(
    api_key: str,
    report_date: date,
    regions: Optional[list] = None,
    analyst_consensus: Optional[float] = None,
) -> list[dict]:
    """
    Fetch EIA weekly storage data for all regions for a given report_date.
    
    Returns list of dicts ready for database insertion.
    Uses published_at = now() for point-in-time integrity.
    """
    if regions is None:
        regions = list(EIA_SERIES_MAP.keys())
    
    published_at = datetime.now(timezone.utc)
    start_str = (report_date.replace(day=1)).isoformat()
    end_str = report_date.isoformat()
    
    records = []
    for region in regions:
        series_id = EIA_SERIES_MAP.get(region)
        if not series_id:
            logger.warning("No series ID for region '%s'; skipping.", region)
            continue
        
        try:
            raw = _fetch_eia_storage_raw(api_key, series_id, start_str, end_str)
        except Exception as exc:
            logger.error("Failed to fetch EIA data for region %s: %s", region, exc)
            continue
        
        series_data = raw.get("response", {}).get("data", [])
        if not series_data:
            logger.warning("No EIA data returned for region %s, date %s", region, report_date)
            continue
        
        # Find the row matching report_date
        target = report_date.isoformat()
        row = next((r for r in series_data if r.get("period") == target), None)
        if row is None:
            logger.warning("EIA data not yet available for %s region %s", report_date, region)
            continue
        
        value = float(row.get("value", 0))
        surprise = None
        if analyst_consensus is not None:
            surprise = round(value - analyst_consensus, 3)
        
        records.append({
            "report_date": report_date,
            "published_at": published_at,
            "region": region,
            "working_gas_bcf": value,
            "net_change_bcf": None,  # Computed from prior week
            "five_year_avg_bcf": None,  # Separate series
            "year_ago_bcf": None,  # Separate series
            "analyst_consensus_bcf": analyst_consensus,
            "storage_surprise_bcf": surprise,
            "revision_number": 0,
            "data_source": "EIA_API",
        })
    
    return records


def detect_revisions(
    new_records: list[dict],
    existing_records: list[dict],
) -> list[dict]:
    """
    Compare new EIA data against previously stored values to detect revisions.
    
    Returns list of revision log entries for data_revision_log table.
    Every revision gets a new row in eia_storage_weekly with incremented revision_number.
    """
    revision_logs = []
    existing_map = {
        (r["report_date"], r["region"]): r
        for r in existing_records
    }
    
    for new in new_records:
        key = (new["report_date"], new["region"])
        existing = existing_map.get(key)
        if existing is None:
            continue
        
        for field in ["working_gas_bcf", "net_change_bcf", "five_year_avg_bcf", "year_ago_bcf"]:
            old_val = existing.get(field)
            new_val = new.get(field)
            if old_val is None or new_val is None:
                continue
            if abs(float(old_val) - float(new_val)) > 0.001:  # 0.001 Bcf threshold
                delta = float(new_val) - float(old_val)
                revision_logs.append({
                    "table_name": "eia_storage_weekly",
                    "report_date": new["report_date"],
                    "region": new["region"],
                    "field_name": field,
                    "old_value": old_val,
                    "new_value": new_val,
                    "delta": delta,
                    "published_at": new["published_at"],
                })
                logger.info(
                    "EIA revision detected: %s %s %s: %.3f -> %.3f (Δ%.3f)",
                    new["report_date"], new["region"], field, old_val, new_val, delta
                )
                # Increment revision number
                new["revision_number"] = existing.get("revision_number", 0) + 1
    
    return revision_logs


def insert_storage_records(records: list[dict], db_session) -> int:
    """
    Insert EIA storage records into the database (append-only, never update).
    
    Returns number of rows inserted.
    """
    if not records:
        return 0
    
    from sqlalchemy import text
    
    sql = text("""
        INSERT INTO eia_storage_weekly
            (report_date, published_at, region, working_gas_bcf, net_change_bcf,
             five_year_avg_bcf, year_ago_bcf, analyst_consensus_bcf,
             storage_surprise_bcf, revision_number, data_source)
        VALUES
            (:report_date, :published_at, :region, :working_gas_bcf, :net_change_bcf,
             :five_year_avg_bcf, :year_ago_bcf, :analyst_consensus_bcf,
             :storage_surprise_bcf, :revision_number, :data_source)
        ON CONFLICT (report_date, region, published_at) DO NOTHING
    """)
    
    inserted = 0
    for rec in records:
        result = db_session.execute(sql, rec)
        inserted += result.rowcount
    
    return inserted


def insert_revision_logs(revision_logs: list[dict], db_session) -> int:
    """Insert revision log entries into data_revision_log."""
    if not revision_logs:
        return 0
    
    from sqlalchemy import text
    
    sql = text("""
        INSERT INTO data_revision_log
            (table_name, report_date, region, field_name, old_value, new_value, delta, published_at)
        VALUES
            (:table_name, :report_date, :region, :field_name, :old_value, :new_value, :delta, :published_at)
    """)
    
    for log in revision_logs:
        db_session.execute(sql, log)
    
    return len(revision_logs)
