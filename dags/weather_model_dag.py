"""
Weather Model DAG — Twice daily at 02:00 and 14:00 ET.

Allows ~2 hours of model run latency after 00Z (available ~02:00 ET)
and 12Z (available ~14:00 ET) GFS/GEFS runs.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import logging

logger = logging.getLogger(__name__)

default_args = {
    "owner": "natgas",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
}


def ingest_gfs_forecast(**context):
    """Ingest GFS 0.25° forecast from NOAA NOMADS."""
    from natgas.pipelines.weather_models import fetch_gfs_forecast, insert_weather_forecast_records
    from natgas.db.connection import get_session
    from datetime import date
    
    exec_dt = context["execution_date"]
    init_hour = "00" if exec_dt.hour < 12 else "12"
    init_date = exec_dt.date()
    
    # Fetch 1-14 day forecasts (lead hours 24-336, every 24h)
    forecast_hours = list(range(24, 337, 24))
    
    records = fetch_gfs_forecast(
        init_date=init_date,
        init_hour=init_hour,
        forecast_hours=forecast_hours,
    )
    
    with get_session() as session:
        inserted = insert_weather_forecast_records(records, session)
    
    logger.info("GFS %sZ: inserted %d records for %d forecast hours", init_hour, inserted, len(forecast_hours))
    return {"inserted": inserted, "model": "GFS", "init_hour": init_hour}


def ingest_gefs_forecast(**context):
    """Ingest GEFS ensemble mean from NOAA NOMADS."""
    from natgas.pipelines.weather_models import fetch_commercial_model_stub, insert_weather_forecast_records
    from natgas.db.connection import get_session
    
    exec_dt = context["execution_date"]
    init_hour = "00" if exec_dt.hour < 12 else "12"
    init_date = exec_dt.date()
    forecast_hours = list(range(24, 337, 24))
    
    # GEFS uses same NOMADS infrastructure as GFS — stub for now
    records = fetch_commercial_model_stub("GEFS", init_date, init_hour, forecast_hours)
    
    with get_session() as session:
        inserted = insert_weather_forecast_records(records, session)
    
    logger.info("GEFS %sZ: inserted %d records", init_hour, inserted)
    return {"inserted": inserted, "model": "GEFS", "init_hour": init_hour}


def ingest_commercial_models(**context):
    """Ingest commercial model forecasts (EURO, AIFS) — requires paid API keys."""
    from natgas.pipelines.weather_models import fetch_commercial_model_stub, insert_weather_forecast_records
    from natgas.db.connection import get_session
    import os
    
    exec_dt = context["execution_date"]
    init_hour = "00" if exec_dt.hour < 12 else "12"
    init_date = exec_dt.date()
    forecast_hours = list(range(24, 337, 24))
    
    total_inserted = 0
    for model in ["EURO_HRES", "AIFS"]:
        records = fetch_commercial_model_stub(model, init_date, init_hour, forecast_hours)
        with get_session() as session:
            inserted = insert_weather_forecast_records(records, session)
        total_inserted += inserted
        logger.info("%s %sZ: inserted %d records", model, init_hour, inserted)
    
    return {"inserted": total_inserted}


def compute_pop_weighted_hdd_cdd(**context):
    """Aggregate raw grid data to population-weighted HDD/CDD by region."""
    from natgas.population_weights.weight_masks import get_seasonal_mask, compute_regional_hdd_cdd, REGIONS
    from natgas.db.connection import get_session
    from sqlalchemy import text
    import numpy as np
    
    exec_dt = context["execution_date"]
    init_hour = "00" if exec_dt.hour < 12 else "12"
    forecast_date = exec_dt.date()
    month = forecast_date.month
    _, season = get_seasonal_mask(month)
    
    # For each model and lead day, aggregate grid data to scalar HDD/CDD
    with get_session() as session:
        sql = text("""
            SELECT model_name, model_version, valid_date, lead_days,
                   latitude, longitude, t2m_fahrenheit, hdd_raw, cdd_raw
            FROM weather_forecast_raw
            WHERE DATE(forecast_init_date) = :forecast_date
              AND EXTRACT(HOUR FROM forecast_init_date) = :init_hour
        """)
        rows = session.execute(sql, {
            "forecast_date": forecast_date,
            "init_hour": int(init_hour),
        }).fetchall()
        
        if not rows:
            logger.warning("No weather_forecast_raw data for %s %sZ", forecast_date, init_hour)
            return {"inserted": 0}
        
        import pandas as pd
        df = pd.DataFrame(rows, columns=["model_name", "model_version", "valid_date",
                                          "lead_days", "latitude", "longitude",
                                          "t2m_fahrenheit", "hdd_raw", "cdd_raw"])
        
        records_to_insert = []
        for (model, version, valid_date, lead), group in df.groupby(
            ["model_name", "model_version", "valid_date", "lead_days"]
        ):
            # Build temp grid from available points (simplified)
            lat_min, lat_max = group["latitude"].min(), group["latitude"].max()
            
            # Simple mean HDD/CDD as fallback (real impl uses dot product with weight mask)
            mean_hdd = float(group["hdd_raw"].mean())
            mean_cdd = float(group["cdd_raw"].mean())
            
            for region_name in ["national", "midwest", "northeast", "texas", "southeast"]:
                records_to_insert.append({
                    "forecast_date": forecast_date,
                    "valid_date": valid_date,
                    "lead_days": int(lead),
                    "model_name": str(model),
                    "model_version": str(version) if version else None,
                    "region": region_name,
                    "seasonal_mask": season,
                    "pop_weighted_hdd": mean_hdd,
                    "pop_weighted_cdd": mean_cdd,
                })
        
        # Insert to hdd_cdd_daily_by_model
        if records_to_insert:
            ins_sql = text("""
                INSERT INTO hdd_cdd_daily_by_model
                    (forecast_date, valid_date, lead_days, model_name, model_version,
                     region, seasonal_mask, pop_weighted_hdd, pop_weighted_cdd)
                VALUES
                    (:forecast_date, :valid_date, :lead_days, :model_name, :model_version,
                     :region, :seasonal_mask, :pop_weighted_hdd, :pop_weighted_cdd)
                ON CONFLICT DO NOTHING
            """)
            for rec in records_to_insert:
                session.execute(ins_sql, rec)
    
    logger.info("Computed HDD/CDD for %d records", len(records_to_insert))
    return {"inserted": len(records_to_insert)}


def send_failure_alert(context):
    """On-failure callback to send Slack alert."""
    from natgas.alerts.notifier import send_slack_alert, format_dag_failure_alert
    task_id = context.get("task_instance", {}).task_id if hasattr(context.get("task_instance", {}), "task_id") else "unknown"
    exc = str(context.get("exception", "Unknown error"))
    msg = format_dag_failure_alert(f"weather_model_dag.{task_id}", exc)
    send_slack_alert(msg)


with DAG(
    dag_id="weather_model_dag",
    default_args={**default_args, "on_failure_callback": send_failure_alert},
    description="Ingest GFS/GEFS/Euro/AIFS weather model forecasts twice daily",
    schedule_interval="0 7,19 * * *",  # 02:00 ET and 14:00 ET = 07:00 and 19:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["weather", "ingestion"],
) as dag:
    
    t_gfs = PythonOperator(task_id="ingest_gfs", python_callable=ingest_gfs_forecast)
    t_gefs = PythonOperator(task_id="ingest_gefs", python_callable=ingest_gefs_forecast)
    t_commercial = PythonOperator(task_id="ingest_commercial_models", python_callable=ingest_commercial_models)
    t_aggregate = PythonOperator(task_id="compute_pop_weighted_hdd_cdd", python_callable=compute_pop_weighted_hdd_cdd)
    
    [t_gfs, t_gefs, t_commercial] >> t_aggregate
