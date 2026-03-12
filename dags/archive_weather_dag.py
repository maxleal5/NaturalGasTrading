"""
Archive Weather DAG — Every Sunday at 01:00 AM ET.

Purpose: Before the TimescaleDB 90-day retention policy purges raw grid data
from weather_forecast_raw, roll up and persist summary statistics to
hdd_cdd_daily_by_model (cold-tier aggregated table).

This ensures historical HDD/CDD records survive beyond 90 days for:
  - Long-range bias correction training
  - Multi-year seasonal tracker
  - Weather model benchmarking scorecards

Also removes duplicate/stale forecast records to keep the hot tier clean.
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
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
    "email_on_failure": True,
    "email_on_retry": False,
}

MODELS = ["GFS", "GEFS", "EURO_HRES", "AIFS"]
REGIONS = ["national", "midwest", "northeast", "texas", "southeast"]

# Archive window: data older than this many days enters the archive pipeline
ARCHIVE_AFTER_DAYS = 75  # Start archiving 75 days in, before 90-day retention drops it


def identify_archivable_dates(**context) -> dict:
    """
    Find distinct (model_name, forecast_date) pairs in weather_forecast_raw
    that are within the archive window (75-90 days old) and not yet
    fully archived to hdd_cdd_daily_by_model.
    """
    from natgas.db.connection import get_session
    from sqlalchemy import text

    exec_dt = context["execution_date"].date()
    cutoff_start = exec_dt - timedelta(days=90)
    cutoff_end = exec_dt - timedelta(days=ARCHIVE_AFTER_DAYS)

    with get_session() as session:
        sql = text("""
            SELECT DISTINCT
                model_name,
                DATE(forecast_init_date) AS forecast_date
            FROM weather_forecast_raw
            WHERE DATE(forecast_init_date) BETWEEN :start_date AND :end_date
              AND DATE(forecast_init_date) NOT IN (
                  SELECT DISTINCT forecast_date
                  FROM hdd_cdd_daily_by_model
                  WHERE model_name != 'ACTUAL'
              )
            ORDER BY forecast_date DESC
        """)
        rows = session.execute(sql, {
            "start_date": cutoff_start,
            "end_date": cutoff_end,
        }).fetchall()

    dates_to_archive = [(str(r[0]), str(r[1])) for r in rows]
    logger.info(
        "Found %d (model, date) pairs to archive (window: %s to %s)",
        len(dates_to_archive), cutoff_start, cutoff_end,
    )
    return {"dates_to_archive": dates_to_archive, "n_pairs": len(dates_to_archive)}


def aggregate_and_archive_hdd_cdd(**context) -> dict:
    """
    For each archivable (model, forecast_date), aggregate raw grid records to
    region-level pop-weighted HDD/CDD and insert into hdd_cdd_daily_by_model.

    Uses the same population-weighted aggregation logic as the live pipeline.
    """
    from natgas.db.connection import get_session
    from natgas.population_weights.weight_masks import compute_regional_hdd_cdd, get_seasonal_mask, REGIONS as WEIGHT_REGIONS
    from sqlalchemy import text
    import numpy as np
    from datetime import date

    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="identify_archivable_dates")
    dates_to_archive = payload.get("dates_to_archive", [])

    if not dates_to_archive:
        logger.info("No dates to archive.")
        return {"archived": 0}

    total_archived = 0

    with get_session() as session:
        for model_name, forecast_date_str in dates_to_archive:
            forecast_date = date.fromisoformat(forecast_date_str)
            month = forecast_date.month
            _, season = get_seasonal_mask(month)

            # Fetch raw grid records for this model/date
            sql_raw = text("""
                SELECT
                    valid_date,
                    lead_days,
                    model_version,
                    latitude,
                    longitude,
                    hdd_raw,
                    cdd_raw
                FROM weather_forecast_raw
                WHERE model_name = :model
                  AND DATE(forecast_init_date) = :forecast_date
                  AND hdd_raw IS NOT NULL
            """)
            rows = session.execute(sql_raw, {
                "model": model_name,
                "forecast_date": forecast_date,
            }).fetchall()

            if not rows:
                continue

            # Aggregate by valid_date (each valid_date = 1 lead_day record)
            from collections import defaultdict
            by_valid = defaultdict(list)
            for r in rows:
                by_valid[(r[0], r[1], r[2])].append({
                    "lat": float(r[3]),
                    "lon": float(r[4]),
                    "hdd": float(r[5] or 0),
                    "cdd": float(r[6] or 0),
                })

            for (valid_date, lead_days, model_version), grid_points in by_valid.items():
                lats = np.array([p["lat"] for p in grid_points])
                lons = np.array([p["lon"] for p in grid_points])
                hdd_vals = np.array([p["hdd"] for p in grid_points])
                cdd_vals = np.array([p["cdd"] for p in grid_points])

                for region_name in REGIONS:
                    pop_hdd, pop_cdd = compute_regional_hdd_cdd(
                        lats=lats,
                        lons=lons,
                        hdd_values=hdd_vals,
                        cdd_values=cdd_vals,
                        region=region_name,
                        season=season,
                    )

                    sql_insert = text("""
                        INSERT INTO hdd_cdd_daily_by_model
                            (forecast_date, valid_date, lead_days, model_name, model_version,
                             region, seasonal_mask, pop_weighted_hdd, pop_weighted_cdd)
                        VALUES
                            (:forecast_date, :valid_date, :lead_days, :model, :model_version,
                             :region, :season, :hdd, :cdd)
                        ON CONFLICT DO NOTHING
                    """)
                    session.execute(sql_insert, {
                        "forecast_date": forecast_date,
                        "valid_date": valid_date,
                        "lead_days": lead_days,
                        "model": model_name,
                        "model_version": model_version,
                        "region": region_name,
                        "season": season,
                        "hdd": round(float(pop_hdd), 3),
                        "cdd": round(float(pop_cdd), 3),
                    })
                    total_archived += 1

    logger.info("Archived %d region-level HDD/CDD records to hdd_cdd_daily_by_model", total_archived)
    return {"archived": total_archived}


def archive_actual_observations(**context) -> dict:
    """
    Archive station-observed ACTUAL HDD/CDD into hdd_cdd_daily_by_model
    using model_name='ACTUAL', lead_days=0.

    Fetches NOAA GHCN daily station data for the archive window and
    computes population-weighted actuals by region — used for MOS bias
    correction training beyond the 90-day hot tier.
    """
    from natgas.db.connection import get_session
    from natgas.population_weights.weight_masks import compute_regional_hdd_cdd, get_seasonal_mask
    from sqlalchemy import text
    from datetime import date
    import numpy as np

    exec_dt = context["execution_date"].date()
    cutoff_start = exec_dt - timedelta(days=90)
    cutoff_end = exec_dt - timedelta(days=ARCHIVE_AFTER_DAYS)

    with get_session() as session:
        # Check which dates already have archived ACTUAL records
        sql_check = text("""
            SELECT DISTINCT valid_date
            FROM hdd_cdd_daily_by_model
            WHERE model_name = 'ACTUAL'
              AND valid_date BETWEEN :start_date AND :end_date
        """)
        existing = {
            r[0] for r in session.execute(sql_check, {
                "start_date": cutoff_start, "end_date": cutoff_end,
            }).fetchall()
        }

        # Fetch ACTUAL obs from weather_forecast_raw (lead_days=0 = station obs)
        sql_actual = text("""
            SELECT
                valid_date,
                latitude,
                longitude,
                hdd_raw,
                cdd_raw
            FROM weather_forecast_raw
            WHERE model_name = 'ACTUAL'
              AND lead_days = 0
              AND valid_date BETWEEN :start_date AND :end_date
              AND hdd_raw IS NOT NULL
        """)
        obs_rows = session.execute(sql_actual, {
            "start_date": cutoff_start, "end_date": cutoff_end,
        }).fetchall()

    if not obs_rows:
        logger.info("No ACTUAL observation records found for archive window.")
        return {"archived_actuals": 0}

    from collections import defaultdict
    by_date = defaultdict(list)
    for r in obs_rows:
        if r[0] not in existing:
            by_date[r[0]].append({
                "lat": float(r[1]),
                "lon": float(r[2]),
                "hdd": float(r[3] or 0),
                "cdd": float(r[4] or 0),
            })

    total = 0
    with get_session() as session:
        for valid_date, points in by_date.items():
            month = valid_date.month if hasattr(valid_date, "month") else datetime.strptime(str(valid_date), "%Y-%m-%d").month
            _, season = get_seasonal_mask(month)

            lats = np.array([p["lat"] for p in points])
            lons = np.array([p["lon"] for p in points])
            hdd_vals = np.array([p["hdd"] for p in points])
            cdd_vals = np.array([p["cdd"] for p in points])

            for region_name in REGIONS:
                pop_hdd, pop_cdd = compute_regional_hdd_cdd(
                    lats=lats, lons=lons,
                    hdd_values=hdd_vals, cdd_values=cdd_vals,
                    region=region_name, season=season,
                )
                sql_insert = text("""
                    INSERT INTO hdd_cdd_daily_by_model
                        (forecast_date, valid_date, lead_days, model_name, model_version,
                         region, seasonal_mask, pop_weighted_hdd, pop_weighted_cdd)
                    VALUES
                        (:valid_date, :valid_date, 0, 'ACTUAL', 'GHCN',
                         :region, :season, :hdd, :cdd)
                    ON CONFLICT DO NOTHING
                """)
                session.execute(sql_insert, {
                    "valid_date": valid_date,
                    "region": region_name,
                    "season": season,
                    "hdd": round(float(pop_hdd), 3),
                    "cdd": round(float(pop_cdd), 3),
                })
                total += 1

    logger.info("Archived %d ACTUAL observation records", total)
    return {"archived_actuals": total}


def prune_stale_hot_tier_records(**context) -> dict:
    """
    Remove duplicate and redundant records from weather_forecast_raw
    older than ARCHIVE_AFTER_DAYS to reduce hot-tier bloat before
    TimescaleDB's retention policy auto-drops them.

    Keeps only the most recent model_version per (model_name, forecast_init_date,
    valid_date, latitude, longitude) to eliminate redundant re-ingestion rows.
    """
    from natgas.db.connection import get_session
    from sqlalchemy import text

    exec_dt = context["execution_date"].date()
    cutoff_end = exec_dt - timedelta(days=ARCHIVE_AFTER_DAYS)

    with get_session() as session:
        sql_dedup = text("""
            DELETE FROM weather_forecast_raw
            WHERE ctid NOT IN (
                SELECT MAX(ctid)
                FROM weather_forecast_raw
                WHERE DATE(forecast_init_date) <= :cutoff_end
                GROUP BY model_name, DATE(forecast_init_date), valid_date,
                         latitude, longitude
            )
            AND DATE(forecast_init_date) <= :cutoff_end
        """)
        result = session.execute(sql_dedup, {"cutoff_end": cutoff_end})
        deleted = result.rowcount if hasattr(result, "rowcount") else 0

    logger.info("Pruned %d stale/duplicate hot-tier records older than %s", deleted, cutoff_end)
    return {"pruned": deleted}


def send_archive_summary(**context) -> None:
    """Send Slack notification with archive run summary."""
    from natgas.alerts.notifier import send_slack_alert

    ti = context["ti"]
    archived = ti.xcom_pull(task_ids="aggregate_and_archive_hdd_cdd") or {}
    actuals = ti.xcom_pull(task_ids="archive_actual_observations") or {}
    pruned = ti.xcom_pull(task_ids="prune_stale_hot_tier_records") or {}

    n_identified = (ti.xcom_pull(task_ids="identify_archivable_dates") or {}).get("n_pairs", 0)
    msg = (
        f":floppy_disk: *Weather Archive Run Complete*\n"
        f"(Model, date) pairs archived: {n_identified}\n"
        f"HDD/CDD records saved to cold tier: {archived.get('archived', 0)}\n"
        f"ACTUAL observation records archived: {actuals.get('archived_actuals', 0)}\n"
        f"Stale hot-tier rows pruned: {pruned.get('pruned', 0)}"
    )
    send_slack_alert(msg)


with DAG(
    dag_id="archive_weather_dag",
    default_args=default_args,
    description="Weekly weather data archival before 90-day TimescaleDB retention purge",
    schedule_interval="0 6 * * 0",  # 1:00 AM ET Sunday = 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["archive", "weather", "maintenance"],
) as dag:

    t_identify = PythonOperator(
        task_id="identify_archivable_dates",
        python_callable=identify_archivable_dates,
    )

    t_archive_hdd = PythonOperator(
        task_id="aggregate_and_archive_hdd_cdd",
        python_callable=aggregate_and_archive_hdd_cdd,
    )

    t_archive_actual = PythonOperator(
        task_id="archive_actual_observations",
        python_callable=archive_actual_observations,
    )

    t_prune = PythonOperator(
        task_id="prune_stale_hot_tier_records",
        python_callable=prune_stale_hot_tier_records,
    )

    t_summary = PythonOperator(
        task_id="send_archive_summary",
        python_callable=send_archive_summary,
    )

    # Identify → archive HDD/CDD + archive actuals in parallel → prune → summary
    t_identify >> [t_archive_hdd, t_archive_actual] >> t_prune >> t_summary
