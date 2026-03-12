"""
Futures Price DAG — Daily at 5:00 PM ET (after CME close).
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
}


def fetch_and_store_daily_settlement(**context):
    """Fetch daily settlement from Barchart and store."""
    from natgas.pipelines.futures_prices import fetch_daily_settlement, insert_daily_settlement
    from natgas.db.connection import get_session
    
    trade_date = context["execution_date"].date()
    record = fetch_daily_settlement(trade_date)
    
    if record is None:
        logger.warning("No settlement data for %s", trade_date)
        return {"inserted": False}
    
    with get_session() as session:
        inserted = insert_daily_settlement(record, session)
    
    logger.info("Futures settlement for %s: $%.4f", trade_date, record.get("front_month_settle", 0))
    return {"inserted": inserted, "trade_date": str(trade_date)}


with DAG(
    dag_id="futures_price_dag",
    default_args=default_args,
    description="Daily natural gas futures price ingestion",
    schedule_interval="0 22 * * 1-5",  # 5:00 PM ET = 22:00 UTC (Mon-Fri)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["futures", "prices"],
) as dag:
    
    fetch_settlement = PythonOperator(
        task_id="fetch_daily_settlement",
        python_callable=fetch_and_store_daily_settlement,
    )
