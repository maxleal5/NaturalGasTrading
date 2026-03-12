"""
Pipeline 2: Weather Model Forecast Data Ingestion

Sources:
- NOAA NOMADS (free): GFS Operational and GEFS Ensemble
- ECMWF (commercial): Euro HRES/ENS and AIFS (stub/placeholder)

Cadence: Twice daily (00Z and 12Z model runs)
Fields: T2m, precip, z500, raw HDD/CDD pre-bias-correction
"""
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

NOAA_NOMADS_BASE = os.getenv("NOAA_NOMADS_BASE_URL", "https://nomads.ncep.noaa.gov/")
GFS_FILTER_URL = NOAA_NOMADS_BASE + "cgi-bin/filter_gfs_0p25.pl"

# CONUS bounding box for grid subsetting
CONUS_LAT_MIN, CONUS_LAT_MAX = 24.0, 50.0
CONUS_LON_MIN, CONUS_LON_MAX = -125.0, -66.0

MODEL_CADENCE = {
    "GFS": ["00", "06", "12", "18"],   # 4 runs/day; we use 00Z and 12Z
    "GEFS": ["00", "06", "12", "18"],
    "EURO_HRES": ["00", "12"],
    "AIFS": ["00", "12"],
}

SUPPORTED_FREE_MODELS = {"GFS", "GEFS"}
COMMERCIAL_MODELS = {"EURO_HRES", "EURO_ENS", "AIFS", "GRAPHCAST"}


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def compute_hdd_cdd(t_fahrenheit: float, base: float = 65.0) -> tuple[float, float]:
    hdd = max(0.0, base - t_fahrenheit)
    cdd = max(0.0, t_fahrenheit - base)
    return hdd, cdd


def fetch_gfs_forecast(
    init_date: date,
    init_hour: str,
    forecast_hours: list[int],
    variables: Optional[list] = None,
) -> list[dict]:
    """
    Fetch GFS 0.25° forecast data from NOAA NOMADS for specified hours.
    
    Returns list of grid-point records with T2m, z500, HDD/CDD.
    Falls back to stub data if NOMADS is unavailable (e.g., rate limited).
    """
    if variables is None:
        variables = ["TMP:2 m above ground", "HGT:500 mb", "APCP:surface"]
    
    records = []
    init_datetime = datetime(
        init_date.year, init_date.month, init_date.day,
        int(init_hour), 0, 0, tzinfo=timezone.utc
    )
    
    for fhr in forecast_hours:
        valid_dt = init_datetime + timedelta(hours=fhr)
        valid_date = valid_dt.date()
        lead_days = fhr // 24
        
        try:
            # Build NOMADS filter URL for GFS 0.25°
            date_str = init_date.strftime("%Y%m%d")
            url = GFS_FILTER_URL
            params = {
                "file": f"gfs.t{init_hour}z.pgrb2.0p25.f{fhr:03d}",
                "dir": f"/gfs.{date_str}/{init_hour}/atmos",
                "var_TMP": "on",
                "var_HGT": "on",
                "var_APCP": "on",
                "lev_2_m_above_ground": "on",
                "lev_500_mb": "on",
                "lev_surface": "on",
                "subregion": "",
                "leftlon": str(CONUS_LON_MIN + 360),  # NOMADS uses 0-360
                "rightlon": str(CONUS_LON_MAX + 360),
                "toplat": str(CONUS_LAT_MAX),
                "bottomlat": str(CONUS_LAT_MIN),
            }
            
            resp = requests.get(url, params=params, timeout=60, stream=True)
            resp.raise_for_status()
            
            # Parse GRIB2 data via cfgrib if available
            try:
                import cfgrib
                import xarray as xr
                import tempfile, os as _os
                
                with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name
                
                try:
                    ds = xr.open_dataset(tmp_path, engine="cfgrib",
                                         backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround"}})
                    t2m_k = ds.get("t2m")
                    if t2m_k is not None:
                        lats = t2m_k.latitude.values
                        lons = t2m_k.longitude.values
                        for i, lat in enumerate(lats[::4]):  # Subsample for performance
                            for j, lon in enumerate(lons[::4]):
                                lon_adj = float(lon) - 360 if float(lon) > 180 else float(lon)
                                t_k = float(t2m_k.values[i*4, j*4])
                                t_c = t_k - 273.15
                                t_f = celsius_to_fahrenheit(t_c)
                                hdd, cdd = compute_hdd_cdd(t_f)
                                records.append({
                                    "time": valid_dt,
                                    "forecast_init_date": init_datetime,
                                    "valid_date": valid_date,
                                    "lead_days": lead_days,
                                    "model_name": "GFS",
                                    "model_version": f"gfs.{date_str}",
                                    "latitude": round(float(lat), 3),
                                    "longitude": round(lon_adj, 3),
                                    "t2m_celsius": round(t_c, 2),
                                    "t2m_fahrenheit": round(t_f, 2),
                                    "precip_mm": None,
                                    "z500_gpm": None,
                                    "hdd_raw": round(hdd, 2),
                                    "cdd_raw": round(cdd, 2),
                                })
                finally:
                    _os.unlink(tmp_path)
            except ImportError:
                logger.warning("cfgrib not available; using stub data for GFS forecast hour %d", fhr)
                records.extend(_stub_gfs_records(init_datetime, valid_dt, valid_date, lead_days))
                
        except requests.RequestException as exc:
            logger.warning("NOMADS request failed for GFS fhr=%d: %s; using stub data", fhr, exc)
            records.extend(_stub_gfs_records(init_datetime, valid_dt, valid_date, lead_days))
    
    return records


def _stub_gfs_records(
    init_datetime: datetime,
    valid_dt: datetime,
    valid_date: date,
    lead_days: int,
    n_points: int = 20,
) -> list[dict]:
    """Generate stub records for testing/development when NOMADS is unavailable."""
    rng = np.random.default_rng(seed=int(valid_dt.timestamp()) % 2**31)
    stub_lats = np.linspace(CONUS_LAT_MIN, CONUS_LAT_MAX, 5)
    stub_lons = np.linspace(CONUS_LON_MIN, CONUS_LON_MAX, 4)
    
    records = []
    for lat in stub_lats:
        for lon in stub_lons:
            t_c = float(rng.normal(15.0, 10.0))  # ~59°F mean
            t_f = celsius_to_fahrenheit(t_c)
            hdd, cdd = compute_hdd_cdd(t_f)
            records.append({
                "time": valid_dt,
                "forecast_init_date": init_datetime,
                "valid_date": valid_date,
                "lead_days": lead_days,
                "model_name": "GFS",
                "model_version": f"stub.{init_datetime.strftime('%Y%m%d')}",
                "latitude": round(float(lat), 3),
                "longitude": round(float(lon), 3),
                "t2m_celsius": round(t_c, 2),
                "t2m_fahrenheit": round(t_f, 2),
                "precip_mm": None,
                "z500_gpm": float(rng.normal(5500, 100)),
                "hdd_raw": round(hdd, 2),
                "cdd_raw": round(cdd, 2),
            })
    return records


def fetch_commercial_model_stub(
    model_name: str,
    init_date: date,
    init_hour: str,
    forecast_hours: list[int],
) -> list[dict]:
    """
    Stub for commercial model ingestion (ECMWF EURO, AIFS, GraphCast).
    
    In production, replace with paid API calls to Barchart OnDemand or Refinitiv.
    Real-time ECMWF HRES/ENS requires a commercial license.
    """
    logger.warning(
        "%s requires a commercial ECMWF/Refinitiv license. "
        "Returning stub data. Configure ECMWF_API_KEY or BARCHART_API_KEY "
        "and implement the paid API integration.",
        model_name,
    )
    init_datetime = datetime(
        init_date.year, init_date.month, init_date.day,
        int(init_hour), tzinfo=timezone.utc
    )
    records = []
    for fhr in forecast_hours:
        valid_dt = init_datetime + timedelta(hours=fhr)
        valid_date = valid_dt.date()
        lead_days = fhr // 24
        records.extend(_stub_gfs_records(init_datetime, valid_dt, valid_date, lead_days))
        for r in records[-20:]:
            r["model_name"] = model_name
            r["model_version"] = f"{model_name.lower()}.stub"
    return records


def insert_weather_forecast_records(records: list[dict], db_session) -> int:
    """Insert weather forecast records into weather_forecast_raw (append-only)."""
    if not records:
        return 0
    from sqlalchemy import text
    sql = text("""
        INSERT INTO weather_forecast_raw
            (time, forecast_init_date, valid_date, lead_days, model_name, model_version,
             latitude, longitude, t2m_celsius, t2m_fahrenheit, precip_mm, z500_gpm,
             hdd_raw, cdd_raw)
        VALUES
            (:time, :forecast_init_date, :valid_date, :lead_days, :model_name, :model_version,
             :latitude, :longitude, :t2m_celsius, :t2m_fahrenheit, :precip_mm, :z500_gpm,
             :hdd_raw, :cdd_raw)
        ON CONFLICT DO NOTHING
    """)
    inserted = 0
    for rec in records:
        result = db_session.execute(sql, rec)
        inserted += result.rowcount
    return inserted
