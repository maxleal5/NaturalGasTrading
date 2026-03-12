"""
Consensus Estimate DAG — Mon/Tue/Wed before EIA release at 8:00 AM ET.

Collects analyst consensus estimates from Barchart (primary) and Refinitiv
(secondary). If neither source is available, fires a Slack alert asking for
manual input. Stores results in analyst_consensus_weekly.

Runs Monday through Wednesday so the estimate is ready well before
Thursday's 10:30 AM ET EIA release.
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
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
    "email_on_failure": True,
    "email_on_retry": False,
}


def resolve_report_date(**context) -> dict:
    """
    Determine the EIA report_date (week-ending Friday) for the current week.

    Reads eia_release_info set by trading_calendar_dag; falls back to
    computing it on-the-fly so this DAG is self-contained.
    """
    import json
    from datetime import date

    release_info_raw = Variable.get("eia_release_info", default_var=None)
    if release_info_raw:
        release_info = json.loads(release_info_raw)
        report_date_str = release_info.get("week_ending_date")
        if report_date_str:
            return {"report_date": report_date_str}

    # Fallback: compute from execution_date
    from natgas.calendar.trading_calendar import get_report_week_ending, get_eia_release_date

    exec_dt = context["execution_date"].date()
    release_date = get_eia_release_date(exec_dt)
    report_date = get_report_week_ending(release_date)
    return {"report_date": report_date.isoformat()}


def fetch_barchart_consensus(**context) -> dict:
    """Fetch analyst consensus from Barchart OnDemand and store in DB."""
    from datetime import date
    from natgas.pipelines.analyst_consensus import fetch_barchart_consensus, insert_consensus_record
    from natgas.db.connection import get_session
    import os

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="resolve_report_date")
    report_date = date.fromisoformat(result["report_date"])

    record = fetch_barchart_consensus(report_date, api_key=os.getenv("BARCHART_API_KEY"))
    if record is None:
        logger.warning("Barchart consensus unavailable for %s", report_date)
        return {"inserted": False, "source": "Barchart"}

    with get_session() as session:
        inserted = insert_consensus_record(record, session)

    logger.info(
        "Barchart consensus for %s: mean=%.1f Bcf (%d respondents) | inserted=%s",
        report_date,
        record.get("mean_estimate_bcf", 0),
        record.get("respondent_count", 0),
        inserted,
    )
    return {"inserted": inserted, "mean_estimate_bcf": record.get("mean_estimate_bcf"), "source": "Barchart"}


def fetch_refinitiv_consensus(**context) -> dict:
    """Fetch analyst consensus from Refinitiv Eikon/Workspace API."""
    from datetime import date
    from natgas.pipelines.analyst_consensus import insert_consensus_record
    from natgas.db.connection import get_session
    import os
    import requests

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="resolve_report_date")
    report_date = date.fromisoformat(result["report_date"])

    api_key = os.getenv("REFINITIV_API_KEY")
    if not api_key:
        logger.warning("REFINITIV_API_KEY not set; skipping Refinitiv consensus.")
        return {"inserted": False, "source": "Refinitiv"}

    try:
        url = "https://api.refinitiv.com/data/news-analytics/v1/headlines"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"query": f"natural gas storage consensus {report_date.isoformat()}", "count": 1}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        # Refinitiv returns analyst consensus in structured data endpoint;
        # this stub parses a minimal expected shape.
        data = resp.json()
        estimates = data.get("estimates", [])
        if not estimates:
            return {"inserted": False, "source": "Refinitiv"}

        e = estimates[0]
        from datetime import datetime, timezone

        record = {
            "report_date": report_date,
            "published_at": datetime.now(timezone.utc),
            "mean_estimate_bcf": float(e.get("mean", 0)),
            "high_estimate_bcf": float(e.get("high", 0)) if e.get("high") else None,
            "low_estimate_bcf": float(e.get("low", 0)) if e.get("low") else None,
            "respondent_count": int(e.get("count", 0)) if e.get("count") else None,
            "source_name": "Refinitiv",
        }

        with get_session() as session:
            inserted = insert_consensus_record(record, session)

        logger.info(
            "Refinitiv consensus for %s: mean=%.1f Bcf | inserted=%s",
            report_date,
            record["mean_estimate_bcf"],
            inserted,
        )
        return {"inserted": inserted, "mean_estimate_bcf": record["mean_estimate_bcf"], "source": "Refinitiv"}

    except Exception as exc:
        logger.error("Refinitiv consensus fetch failed: %s", exc)
        return {"inserted": False, "source": "Refinitiv"}


def check_consensus_coverage(**context) -> dict:
    """
    Verify at least one consensus estimate was inserted this week.
    If none found, fire a Slack alert requesting manual input.
    """
    from datetime import date
    from natgas.db.connection import get_session
    from natgas.alerts.notifier import send_slack_alert
    from sqlalchemy import text

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="resolve_report_date")
    report_date = date.fromisoformat(result["report_date"])

    with get_session() as session:
        sql = text("""
            SELECT COUNT(*) AS cnt, MAX(mean_estimate_bcf) AS latest_mean
            FROM analyst_consensus_weekly
            WHERE report_date = :report_date
        """)
        row = session.execute(sql, {"report_date": report_date}).fetchone()

    count = int(row[0]) if row else 0
    latest_mean = float(row[1]) if row and row[1] is not None else None

    if count == 0:
        msg = (
            f":warning: *No Consensus Estimate Available* — week ending {report_date}\n"
            f"Neither Barchart nor Refinitiv returned data.\n"
            f"Please enter the analyst consensus manually by Wednesday EOD.\n"
            f"Use: `POST /api/consensus/manual` or Airflow Variable `manual_consensus_bcf`."
        )
        send_slack_alert(msg)
        logger.warning("No consensus estimate for %s — manual input alert sent", report_date)
        return {"coverage": False, "report_date": str(report_date)}

    logger.info(
        "Consensus coverage OK for %s: %d source(s), latest mean=%.1f Bcf",
        report_date, count, latest_mean or 0,
    )
    return {"coverage": True, "count": count, "latest_mean": latest_mean}


def ingest_manual_consensus_if_set(**context) -> dict:
    """
    Check Airflow Variable 'manual_consensus_bcf' and insert if set.
    Clears the variable after ingestion to avoid double-insertion.
    """
    import json
    from datetime import date, datetime, timezone
    from natgas.pipelines.analyst_consensus import record_manual_consensus, insert_consensus_record
    from natgas.db.connection import get_session

    ti = context["ti"]
    result = ti.xcom_pull(task_ids="resolve_report_date")
    report_date = date.fromisoformat(result["report_date"])

    manual_raw = Variable.get("manual_consensus_bcf", default_var=None)
    if not manual_raw:
        return {"inserted": False}

    try:
        manual_val = float(manual_raw)
    except ValueError:
        logger.error("Invalid manual_consensus_bcf value: %r", manual_raw)
        return {"inserted": False}

    record = record_manual_consensus(
        report_date=report_date,
        mean_estimate_bcf=manual_val,
        source_name="Manual",
    )

    with get_session() as session:
        inserted = insert_consensus_record(record, session)

    if inserted:
        # Clear the variable so it won't be re-inserted next run
        Variable.set("manual_consensus_bcf", "")
        logger.info("Manual consensus %.1f Bcf inserted for %s", manual_val, report_date)

    return {"inserted": inserted, "mean_estimate_bcf": manual_val}


with DAG(
    dag_id="consensus_estimate_dag",
    default_args=default_args,
    description="Collect analyst consensus estimates Mon–Wed before EIA release",
    schedule_interval="0 13 * * 1,2,3",  # 8:00 AM ET = 13:00 UTC (Mon/Tue/Wed)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["consensus", "eia", "analyst"],
) as dag:

    t_resolve = PythonOperator(
        task_id="resolve_report_date",
        python_callable=resolve_report_date,
    )

    t_barchart = PythonOperator(
        task_id="fetch_barchart_consensus",
        python_callable=fetch_barchart_consensus,
    )

    t_refinitiv = PythonOperator(
        task_id="fetch_refinitiv_consensus",
        python_callable=fetch_refinitiv_consensus,
    )

    t_manual = PythonOperator(
        task_id="ingest_manual_consensus_if_set",
        python_callable=ingest_manual_consensus_if_set,
    )

    t_check = PythonOperator(
        task_id="check_consensus_coverage",
        python_callable=check_consensus_coverage,
    )

    # Resolve date → fetch from both sources in parallel → check manual → verify coverage
    t_resolve >> [t_barchart, t_refinitiv] >> t_manual >> t_check
