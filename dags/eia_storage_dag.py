"""
EIA Storage DAG — Runs on EIA release day at 10:35 AM ET (5 min after report).

Release schedule (dynamic — set by trading_calendar_dag):
  - Standard: Thursday 10:30 AM ET → this DAG polls at 10:35 AM ET
  - Holiday week: Friday 10:30 AM ET → schedule_interval via dynamic timetable

Sequence:
  1. Wait for the release datetime from Airflow Variable.
  2. Fetch EIA storage data (with exponential backoff).
  3. Fetch intraday futures price (pre-release T-5min, T+1min, T+5min, T+15min).
  4. Compute storage surprise = actual - consensus.
  5. Compute directional signal (+ whisper number adjustment).
  6. Insert all records; update signal_log.
  7. Send trade alert to Slack.
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
    "retries": 5,
    "retry_delay": timedelta(minutes=3),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
    "email_on_failure": True,
    "email_on_retry": False,
}


def resolve_release_metadata(**context) -> dict:
    """Load release date, week-ending date, and consensus from DB/Variables."""
    import json
    from datetime import date
    from natgas.db.connection import get_session
    from sqlalchemy import text

    release_info_raw = Variable.get("eia_release_info", default_var=None)
    if release_info_raw:
        info = json.loads(release_info_raw)
    else:
        # Fallback: derive from execution_date
        from natgas.calendar.trading_calendar import get_eia_release_date, get_report_week_ending
        exec_date = context["execution_date"].date()
        release_date = get_eia_release_date(exec_date)
        report_date = get_report_week_ending(release_date)
        info = {
            "release_date": release_date.isoformat(),
            "week_ending_date": report_date.isoformat(),
        }

    report_date = date.fromisoformat(info["week_ending_date"])

    # Fetch latest consensus from DB
    with get_session() as session:
        sql = text("""
            SELECT mean_estimate_bcf
            FROM analyst_consensus_weekly
            WHERE report_date = :report_date
            ORDER BY published_at DESC
            LIMIT 1
        """)
        row = session.execute(sql, {"report_date": report_date}).fetchone()

    consensus_bcf = float(row[0]) if row and row[0] is not None else None

    logger.info(
        "EIA release: %s | week ending: %s | consensus: %s Bcf",
        info["release_date"], report_date, consensus_bcf,
    )
    return {
        "release_date": info["release_date"],
        "report_date": report_date.isoformat(),
        "consensus_bcf": consensus_bcf,
    }


def fetch_intraday_pre_release_price(**context) -> dict:
    """
    Fetch futures price at T-5min (10:25 AM ET) via Barchart intraday endpoint.
    Store as pre-release baseline for surprise/drift computation.
    """
    import os
    import requests
    from datetime import date, datetime, timezone

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="resolve_release_metadata")
    report_date = date.fromisoformat(meta["report_date"])
    release_date = date.fromisoformat(meta["release_date"])

    api_key = os.getenv("BARCHART_API_KEY")
    if not api_key:
        logger.warning("BARCHART_API_KEY not set; skipping intraday pre-release fetch.")
        return {"price_t_minus_5min": None}

    try:
        url = "https://ondemand.websol.barchart.com/getHistory.json"
        params = {
            "apikey": api_key,
            "symbol": "NG*1",
            "type": "minutes",
            "startDate": release_date.strftime("%Y%m%d") + "1025",
            "endDate": release_date.strftime("%Y%m%d") + "1026",
            "maxRecords": 1,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        price = float(results[0].get("close", 0)) if results else None
    except Exception as exc:
        logger.error("Pre-release intraday fetch failed: %s", exc)
        price = None

    return {"price_t_minus_5min": price, "trade_date": str(release_date), "report_date": str(report_date)}


def fetch_and_store_eia_storage(**context) -> dict:
    """
    Fetch EIA weekly storage report and insert all regions.
    Uses exponential backoff (tenacity) in the pipeline layer.
    """
    import os
    from datetime import date
    from natgas.pipelines.eia_storage import fetch_weekly_storage, insert_storage_records
    from natgas.db.connection import get_session

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="resolve_release_metadata")
    report_date = date.fromisoformat(meta["report_date"])
    consensus_bcf = meta.get("consensus_bcf")

    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY environment variable not set")

    records = fetch_weekly_storage(
        api_key=api_key,
        report_date=report_date,
        analyst_consensus=consensus_bcf,
    )

    if not records:
        raise RuntimeError(
            f"EIA returned no storage records for {report_date}. "
            "Report may not be published yet — will retry."
        )

    with get_session() as session:
        inserted_count = insert_storage_records(records, session)

    total_record = next((r for r in records if r.get("region") == "total"), None)
    actual_bcf = total_record["working_gas_bcf"] if total_record else None
    net_change = total_record["net_change_bcf"] if total_record else None

    logger.info(
        "EIA storage for %s: working_gas=%.1f Bcf, "
        "net_change=%s Bcf | %d regions inserted",
        report_date, actual_bcf or 0, net_change, inserted_count,
    )
    return {
        "report_date": str(report_date),
        "actual_bcf": actual_bcf,
        "net_change_bcf": net_change,
        "inserted": inserted_count,
    }


def fetch_intraday_post_release_prices(**context) -> dict:
    """
    Fetch intraday prices at T+1min, T+5min, T+15min post-release.
    Stores the full intraday record in ngas_futures_release_day_intraday.
    """
    import os
    import requests
    from datetime import date
    from natgas.db.connection import get_session
    from sqlalchemy import text

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="resolve_release_metadata")
    pre = ti.xcom_pull(task_ids="fetch_intraday_pre_release_price")

    report_date = date.fromisoformat(meta["report_date"])
    release_date = date.fromisoformat(meta["release_date"])
    price_t_minus_5min = pre.get("price_t_minus_5min")

    api_key = os.getenv("BARCHART_API_KEY")
    prices = {"t_plus_1min": None, "t_plus_5min": None, "t_plus_15min": None}

    if api_key:
        for label, minute_str in [("t_plus_1min", "1031"), ("t_plus_5min", "1035"), ("t_plus_15min", "1045")]:
            try:
                url = "https://ondemand.websol.barchart.com/getHistory.json"
                params = {
                    "apikey": api_key,
                    "symbol": "NG*1",
                    "type": "minutes",
                    "startDate": release_date.strftime("%Y%m%d") + minute_str,
                    "endDate": release_date.strftime("%Y%m%d") + str(int(minute_str) + 1).zfill(4),
                    "maxRecords": 1,
                }
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    prices[label] = float(results[0].get("close", 0))
            except Exception as exc:
                logger.warning("Post-release price fetch (%s) failed: %s", label, exc)

    # Get Monday close for pre_release_drift computation
    try:
        with get_session() as session:
            sql = text("""
                SELECT front_month_settle
                FROM ngas_futures_daily
                WHERE trade_date < :release_date
                ORDER BY trade_date DESC
                LIMIT 1
            """)
            row = session.execute(sql, {"release_date": release_date}).fetchone()
            monday_close = float(row[0]) if row and row[0] else None
    except Exception:
        monday_close = None

    pre_release_drift = None
    if price_t_minus_5min and monday_close:
        pre_release_drift = round(price_t_minus_5min - monday_close, 4)

    with get_session() as session:
        sql = text("""
            INSERT INTO ngas_futures_release_day_intraday
                (trade_date, report_date, price_t_minus_5min, price_t_plus_1min,
                 price_t_plus_5min, price_t_plus_15min, pre_release_drift, data_source)
            VALUES
                (:trade_date, :report_date, :t_minus_5, :t_plus_1,
                 :t_plus_5, :t_plus_15, :drift, 'Barchart')
            ON CONFLICT (trade_date, report_date) DO UPDATE SET
                price_t_plus_1min   = EXCLUDED.price_t_plus_1min,
                price_t_plus_5min   = EXCLUDED.price_t_plus_5min,
                price_t_plus_15min  = EXCLUDED.price_t_plus_15min,
                pre_release_drift   = EXCLUDED.pre_release_drift
        """)
        session.execute(sql, {
            "trade_date": release_date,
            "report_date": report_date,
            "t_minus_5": price_t_minus_5min,
            "t_plus_1": prices["t_plus_1min"],
            "t_plus_5": prices["t_plus_5min"],
            "t_plus_15": prices["t_plus_15min"],
            "drift": pre_release_drift,
        })

    logger.info(
        "Intraday prices stored for %s: T-5=%s, T+1=%s, T+5=%s, T+15=%s | drift=%s",
        release_date, price_t_minus_5min,
        prices["t_plus_1min"], prices["t_plus_5min"], prices["t_plus_15min"],
        pre_release_drift,
    )
    return {
        "price_t_minus_5min": price_t_minus_5min,
        "pre_release_drift": pre_release_drift,
        **prices,
    }


def compute_and_store_signal(**context) -> dict:
    """
    Compute storage surprise + directional signal, store in signal_log.
    
    Pulls model estimate from the most recent signal_log row (pre-populated
    by weekly_analysis_dag run earlier in the week). If no model estimate
    exists yet, falls back to analyst consensus as proxy.
    """
    from datetime import date
    from natgas.analysis.surprise_signal import compute_storage_surprise, compute_directional_signal
    from natgas.analysis.seasonal_tracker import compute_storage_percentile
    from natgas.db.connection import get_session
    from natgas.calendar.trading_calendar import get_report_week_ending
    from sqlalchemy import text

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="resolve_release_metadata")
    storage_result = ti.xcom_pull(task_ids="fetch_and_store_eia_storage")
    intraday = ti.xcom_pull(task_ids="fetch_intraday_post_release_prices")

    report_date = date.fromisoformat(meta["report_date"])
    actual_bcf = storage_result.get("actual_bcf")
    net_change_bcf = storage_result.get("net_change_bcf")
    consensus_bcf = meta.get("consensus_bcf")
    pre_release_drift = intraday.get("pre_release_drift") if intraday else None

    if actual_bcf is None:
        logger.error("No actual_bcf available for signal computation — skipping.")
        return {"signal": None}

    with get_session() as session:
        # Fetch pre-release model estimate (if any)
        sql = text("""
            SELECT model_estimate_bcf, euro_estimate_bcf, gefs_estimate_bcf, aifs_estimate_bcf
            FROM signal_log
            WHERE report_date = :report_date
            LIMIT 1
        """)
        row = session.execute(sql, {"report_date": report_date}).fetchone()
        model_estimate = float(row[0]) if row and row[0] else consensus_bcf

        # Fetch historical storage for percentile computation
        sql_hist = text("""
            SELECT DISTINCT ON (report_date)
                working_gas_bcf
            FROM eia_storage_weekly
            WHERE region = 'total'
              AND EXTRACT(WEEK FROM report_date) = EXTRACT(WEEK FROM :report_date::date)
              AND report_date < :report_date
            ORDER BY report_date, published_at DESC
            LIMIT 10
        """)
        hist_rows = session.execute(sql_hist, {"report_date": report_date}).fetchall()
        historical_vals = [float(r[0]) for r in hist_rows if r[0] is not None]

    storage_percentile = compute_storage_percentile(actual_bcf, historical_vals) if historical_vals else None

    month = report_date.month
    season = "winter" if month in (11, 12, 1, 2, 3) else "summer"

    surprise_vs_consensus = (actual_bcf - consensus_bcf) if consensus_bcf else None
    surprise_vs_model = compute_storage_surprise(model_estimate, consensus_bcf) if (model_estimate and consensus_bcf) else None

    signal, confidence = compute_directional_signal(
        storage_surprise_bcf=surprise_vs_consensus or 0.0,
        pre_release_price_drift=pre_release_drift,
        storage_percentile=storage_percentile,
        season=season,
    )

    price_t_plus_15min = intraday.get("t_plus_15min") if intraday else None
    price_t_minus_5min = intraday.get("price_t_minus_5min") if intraday else None
    actual_price_move = None
    if price_t_plus_15min and price_t_minus_5min:
        actual_price_move = round(price_t_plus_15min - price_t_minus_5min, 4)

    # Determine if signal was correct (bullish = price went up, bearish = price went down)
    signal_correct = None
    if signal != 0 and actual_price_move is not None:
        if signal == 1:
            signal_correct = actual_price_move > 0
        elif signal == -1:
            signal_correct = actual_price_move < 0

    with get_session() as session:
        sql_upsert = text("""
            INSERT INTO signal_log
                (report_date, model_estimate_bcf, analyst_consensus_bcf,
                 model_vs_consensus_bcf, pre_release_price_drift,
                 directional_signal, confidence_score,
                 actual_bcf, actual_signal, signal_correct)
            VALUES
                (:report_date, :model_estimate, :consensus,
                 :model_vs_consensus, :drift,
                 :signal, :confidence,
                 :actual_bcf, :actual_signal, :signal_correct)
            ON CONFLICT (report_date) DO UPDATE SET
                actual_bcf          = EXCLUDED.actual_bcf,
                actual_signal       = EXCLUDED.actual_signal,
                signal_correct      = EXCLUDED.signal_correct,
                directional_signal  = EXCLUDED.directional_signal,
                confidence_score    = EXCLUDED.confidence_score,
                pre_release_price_drift = EXCLUDED.pre_release_price_drift,
                model_vs_consensus_bcf  = EXCLUDED.model_vs_consensus_bcf
        """)
        session.execute(sql_upsert, {
            "report_date": report_date,
            "model_estimate": model_estimate,
            "consensus": consensus_bcf,
            "model_vs_consensus": surprise_vs_model,
            "drift": pre_release_drift,
            "signal": signal,
            "confidence": confidence,
            "actual_bcf": actual_bcf,
            "actual_signal": 1 if (actual_price_move or 0) > 0 else (-1 if (actual_price_move or 0) < 0 else 0),
            "signal_correct": signal_correct,
        })

    logger.info(
        "Signal for %s: %+d (conf=%.2f) | actual=%.1f Bcf | vs_consensus=%s | price_move=%s",
        report_date, signal, confidence, actual_bcf,
        f"{surprise_vs_consensus:+.1f}" if surprise_vs_consensus else "N/A",
        f"{actual_price_move:+.4f}" if actual_price_move else "N/A",
    )
    return {
        "report_date": str(report_date),
        "signal": signal,
        "confidence": confidence,
        "actual_bcf": actual_bcf,
        "surprise_vs_consensus": surprise_vs_consensus,
        "actual_price_move": actual_price_move,
        "signal_correct": signal_correct,
    }


def send_release_alert(**context) -> None:
    """Send Slack alert with EIA result, signal, and price reaction."""
    from natgas.alerts.notifier import send_slack_alert

    ti = context["ti"]
    signal_result = ti.xcom_pull(task_ids="compute_and_store_signal")
    meta = ti.xcom_pull(task_ids="resolve_release_metadata")

    if not signal_result:
        return

    signal = signal_result.get("signal")
    confidence = signal_result.get("confidence", 0)
    actual_bcf = signal_result.get("actual_bcf")
    surprise = signal_result.get("surprise_vs_consensus")
    price_move = signal_result.get("actual_price_move")
    report_date = signal_result.get("report_date")
    consensus = meta.get("consensus_bcf")

    signal_emoji = ":large_green_circle:" if signal == 1 else (":red_circle:" if signal == -1 else ":white_circle:")
    signal_label = "BULLISH" if signal == 1 else ("BEARISH" if signal == -1 else "NEUTRAL")

    surprise_str = f"{surprise:+.1f} Bcf vs consensus" if surprise is not None else "N/A"
    price_str = f"{price_move:+.4f} $/MMBtu (T-5 to T+15)" if price_move is not None else "N/A"
    consensus_str = f"{consensus:.1f} Bcf" if consensus is not None else "N/A"

    msg = (
        f"{signal_emoji} *EIA Storage Report — {report_date}*\n"
        f"Actual: *{actual_bcf:.1f} Bcf* | Consensus: {consensus_str}\n"
        f"Surprise: *{surprise_str}*\n"
        f"Signal: *{signal_label}* (confidence: {confidence:.0%})\n"
        f"Price reaction: {price_str}"
    )
    send_slack_alert(msg)


with DAG(
    dag_id="eia_storage_dag",
    default_args=default_args,
    description="EIA storage report ingestion and signal generation",
    schedule_interval="35 15 * * 4,5",  # 10:35 AM ET Thu+Fri = 15:35 UTC (winter)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["eia", "storage", "signal"],
) as dag:

    t_meta = PythonOperator(
        task_id="resolve_release_metadata",
        python_callable=resolve_release_metadata,
    )

    t_pre_price = PythonOperator(
        task_id="fetch_intraday_pre_release_price",
        python_callable=fetch_intraday_pre_release_price,
    )

    t_eia = PythonOperator(
        task_id="fetch_and_store_eia_storage",
        python_callable=fetch_and_store_eia_storage,
        retries=8,
        retry_delay=timedelta(minutes=2),  # Report may be delayed; retry frequently
    )

    t_post_price = PythonOperator(
        task_id="fetch_intraday_post_release_prices",
        python_callable=fetch_intraday_post_release_prices,
    )

    t_signal = PythonOperator(
        task_id="compute_and_store_signal",
        python_callable=compute_and_store_signal,
    )

    t_alert = PythonOperator(
        task_id="send_release_alert",
        python_callable=send_release_alert,
    )

    # Pre-release price captured before EIA; then EIA + post-price in parallel; then signal + alert
    t_meta >> t_pre_price >> t_eia >> t_post_price >> t_signal >> t_alert
