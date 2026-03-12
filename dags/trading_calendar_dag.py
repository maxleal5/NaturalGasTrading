"""
Trading Calendar DAG — Every Monday morning.

Computes the correct EIA release date for the week using CME/NYSE holiday calendar.
Sets Airflow Variables used by eia_storage_dag and weekly_analysis_dag.
Runs at 7:00 AM ET every Monday.
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
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
}


def compute_and_store_release_date(**context):
    """Compute EIA release date for current week and store as Airflow Variable."""
    from natgas.calendar.trading_calendar import get_eia_release_date, get_eia_release_datetime, get_report_week_ending
    from datetime import date
    import json
    
    today = context["execution_date"].date()
    release_date = get_eia_release_date(today)
    release_datetime = get_eia_release_datetime(today)
    week_ending = get_report_week_ending(release_date)
    
    release_info = {
        "release_date": release_date.isoformat(),
        "release_datetime_et": release_datetime.isoformat(),
        "week_ending_date": week_ending.isoformat(),
        "computed_at": datetime.utcnow().isoformat(),
    }
    
    Variable.set("eia_release_info", json.dumps(release_info))
    
    logger.info(
        "Week of %s: EIA release on %s (week ending %s)",
        today, release_date, week_ending
    )
    
    return release_info


def send_weekly_schedule_notification(**context):
    """Send Slack notification with weekly EIA release schedule."""
    import json
    from natgas.alerts.notifier import send_slack_alert
    
    release_info = json.loads(Variable.get("eia_release_info", "{}"))
    if not release_info:
        return
    
    msg = (
        f":calendar: *NatGas Weekly Schedule*\n"
        f"EIA storage report release: *{release_info.get('release_date')}* at 10:30 AM ET\n"
        f"Coverage week ending: {release_info.get('week_ending_date')}"
    )
    send_slack_alert(msg)


with DAG(
    dag_id="trading_calendar_dag",
    default_args=default_args,
    description="Compute dynamic EIA release date from trading calendar",
    schedule_interval="0 12 * * 1",  # 7:00 AM ET = 12:00 UTC (winter), adjust for DST
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["calendar", "eia", "scheduling"],
) as dag:
    
    compute_release_date = PythonOperator(
        task_id="compute_eia_release_date",
        python_callable=compute_and_store_release_date,
    )
    
    notify = PythonOperator(
        task_id="send_schedule_notification",
        python_callable=send_weekly_schedule_notification,
    )
    
    compute_release_date >> notify
