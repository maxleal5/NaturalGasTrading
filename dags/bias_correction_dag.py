"""
Bias Correction DAG — Runs after each weather_model_dag run.

Updates rolling 30-day MOS bias correction.
Runs stability check and fires drift alert if needed.
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
    "retry_delay": timedelta(minutes=5),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
}

MODELS = ["GFS", "GEFS", "EURO_HRES", "AIFS"]
REGIONS = ["national", "midwest", "northeast", "texas", "southeast"]
LEAD_DAYS = [1, 3, 5, 7, 10, 14]


def run_bias_correction(**context):
    """Run MOS bias correction for all models/regions/lead days."""
    from natgas.analysis.bias_correction import BiasCorrector
    from natgas.db.connection import get_session
    from natgas.population_weights.weight_masks import get_seasonal_mask
    from sqlalchemy import text
    
    exec_dt = context["execution_date"]
    forecast_date = exec_dt.date()
    month = forecast_date.month
    _, season = get_seasonal_mask(month)
    
    total_corrected = 0
    
    with get_session() as session:
        corrector = BiasCorrector(db_session=session)
        
        for model in MODELS:
            for region in REGIONS:
                # Fetch raw HDD/CDD records for this model/region
                sql = text("""
                    SELECT forecast_date, valid_date, lead_days, model_name, model_version,
                           pop_weighted_hdd, pop_weighted_cdd, seasonal_mask
                    FROM hdd_cdd_daily_by_model
                    WHERE model_name = :model AND region = :region
                      AND forecast_date = :forecast_date
                """)
                rows = session.execute(sql, {
                    "model": model, "region": region, "forecast_date": forecast_date
                }).fetchall()
                
                if not rows:
                    continue
                
                raw_records = [
                    dict(zip(["forecast_date", "valid_date", "lead_days", "model_name",
                               "model_version", "pop_weighted_hdd", "pop_weighted_cdd",
                               "seasonal_mask"], row))
                    for row in rows
                ]
                
                corrected = corrector.run_bias_correction_pipeline(
                    model_name=model,
                    region=region,
                    forecast_date=forecast_date,
                    lead_days_range=LEAD_DAYS,
                    raw_records=raw_records,
                    db_session=session,
                )
                total_corrected += len(corrected)
    
    logger.info("Bias correction complete: %d records corrected", total_corrected)
    return {"corrected": total_corrected}


def check_model_stability(**context):
    """Run stability check for AI models — trigger drift alert if needed."""
    from natgas.analysis.bias_correction import BiasCorrector
    from natgas.alerts.notifier import send_slack_alert, format_model_drift_alert
    from natgas.db.connection import get_session
    from sqlalchemy import text
    
    exec_dt = context["execution_date"]
    forecast_date = exec_dt.date()
    
    ai_models = ["AIFS", "EURO_HRES"]  # AI/commercial models most prone to silent updates
    
    with get_session() as session:
        corrector = BiasCorrector(db_session=session)
        
        for model in ai_models:
            for region in REGIONS:
                for lead in [1, 3, 7]:
                    # Get latest residual bias
                    sql = text("""
                        SELECT AVG(pop_weighted_hdd_corrected - a.pop_weighted_hdd) as residual
                        FROM hdd_cdd_bias_corrected_by_model f
                        JOIN hdd_cdd_daily_by_model a
                            ON a.valid_date = f.valid_date AND a.region = f.region
                           AND a.model_name = 'ACTUAL' AND a.lead_days = 0
                        WHERE f.model_name = :model AND f.region = :region
                          AND f.lead_days = :lead
                          AND f.forecast_date >= :start_date
                    """)
                    row = session.execute(sql, {
                        "model": model, "region": region, "lead": lead,
                        "start_date": forecast_date - timedelta(days=7),
                    }).fetchone()
                    
                    if not row or row[0] is None:
                        continue
                    
                    residual = float(row[0])
                    stability = corrector.check_model_stability(
                        model, region, lead, residual, session
                    )
                    
                    corrector.insert_stability_log(
                        model_name=model, region=region, lead_days=lead,
                        residual_bias=residual,
                        bias_std_30d=stability.get("std", 0.0) or 0.0,
                        z_score=stability.get("z_score", 0.0),
                        consecutive=stability.get("consecutive", 0),
                        alert_triggered=stability.get("alert", False),
                        mos_window_adjusted=stability.get("alert", False),
                        db_session=session,
                    )
                    
                    if stability.get("alert"):
                        msg = format_model_drift_alert(
                            model, region, lead, residual,
                            stability["z_score"], stability["consecutive"]
                        )
                        send_slack_alert(msg)
                        logger.warning("DRIFT ALERT sent for %s/%s/lead=%d", model, region, lead)

from datetime import timedelta  # ensure imported in scope

with DAG(
    dag_id="bias_correction_dag",
    default_args=default_args,
    description="MOS bias correction and model stability monitoring",
    schedule_interval="30 7,19 * * *",  # 30 min after weather_model_dag
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bias_correction", "mos", "drift_detection"],
) as dag:
    
    t_correct = PythonOperator(
        task_id="run_bias_correction",
        python_callable=run_bias_correction,
    )
    
    t_stability = PythonOperator(
        task_id="check_model_stability",
        python_callable=check_model_stability,
    )
    
    t_correct >> t_stability
