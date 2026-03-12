"""
Pipeline 5: Supplemental Fundamentals (Weekly)

Sources: EIA, Genscape/Wood Mackenzie (budget permitting)
Fields: LNG exports, dry gas production, pipeline exports to Mexico,
        residential/commercial demand
"""
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

EIA_BASE_URL = os.getenv("EIA_BASE_URL", "https://api.eia.gov/v2/")

# EIA series IDs for supply/demand data
EIA_SERIES = {
    "dry_gas_production": "NG.N9070US2.W",    # Marketed production
    "lng_exports": "NG.N9133US2.W",            # LNG exports
    "pipeline_exports_mexico": "NG.N9130MX2.W", # Pipeline exports to Mexico
    "residential_demand": "NG.N3010US2.W",
    "commercial_demand": "NG.N3020US2.W",
    "industrial_demand": "NG.N3035US2.W",
}


def fetch_supply_demand(
    api_key: str,
    week_ending_date: date,
) -> Optional[dict]:
    """
    Fetch weekly supply/demand fundamentals from EIA API.
    Returns a dict ready for supply_demand_weekly insertion.
    """
    published_at = datetime.now(timezone.utc)
    start_str = week_ending_date.replace(day=1).isoformat()
    end_str = week_ending_date.isoformat()
    
    values = {}
    for field, series_id in EIA_SERIES.items():
        try:
            url = f"{EIA_BASE_URL}natural-gas/sum/lsum/data/"
            params = {
                "api_key": api_key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": series_id,
                "start": start_str,
                "end": end_str,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            series_data = data.get("response", {}).get("data", [])
            if series_data:
                values[field] = float(series_data[0].get("value", 0))
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", field, exc)
            values[field] = None
    
    rc_demand = None
    r = values.get("residential_demand")
    c = values.get("commercial_demand")
    if r is not None and c is not None:
        rc_demand = r + c
    
    return {
        "week_ending_date": week_ending_date,
        "published_at": published_at,
        "lng_exports_bcfd": values.get("lng_exports"),
        "dry_gas_production_bcfd": values.get("dry_gas_production"),
        "pipeline_exports_mexico_bcfd": values.get("pipeline_exports_mexico"),
        "residential_commercial_demand_bcfd": rc_demand,
        "industrial_demand_bcfd": values.get("industrial_demand"),
        "data_source": "EIA_API",
    }


def insert_supply_demand_record(record: dict, db_session) -> bool:
    from sqlalchemy import text
    sql = text("""
        INSERT INTO supply_demand_weekly
            (week_ending_date, published_at, lng_exports_bcfd, dry_gas_production_bcfd,
             pipeline_exports_mexico_bcfd, residential_commercial_demand_bcfd,
             industrial_demand_bcfd, data_source)
        VALUES
            (:week_ending_date, :published_at, :lng_exports_bcfd, :dry_gas_production_bcfd,
             :pipeline_exports_mexico_bcfd, :residential_commercial_demand_bcfd,
             :industrial_demand_bcfd, :data_source)
        ON CONFLICT (week_ending_date, published_at) DO NOTHING
    """)
    result = db_session.execute(sql, record)
    return result.rowcount > 0
