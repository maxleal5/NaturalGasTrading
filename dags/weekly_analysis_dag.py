"""
Weekly Analysis DAG — Runs Friday at 3:00 PM ET (post-EIA, post-close).

Sequence:
  1. Build feature matrix from weekly_analysis_master view.
  2. Retrain / update storage draw model (Ridge + XGBoost).
  3. Compute pre-release model estimate for NEXT week and write to signal_log.
  4. Run weather model accuracy benchmarking (90-day scorecard).
  5. Update seasonal surplus/deficit tracker + project end-of-season storage.
  6. Refresh the weekly_analysis_master materialized view.
  7. Retrain price sensitivity model quarterly (if applicable).
  8. Send weekly summary Slack alert.
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
    "retry_delay": timedelta(minutes=10),
    "email": [Variable.get("ALERT_EMAIL", default_var="")],
    "email_on_failure": True,
    "email_on_retry": False,
}

MODELS = ["GFS", "GEFS", "EURO_HRES", "AIFS"]
REGIONS = ["national", "midwest", "northeast", "texas", "southeast"]


def load_training_data(**context) -> dict:
    """
    Load the weekly feature matrix from weekly_analysis_master for model training.
    Returns summary stats for downstream tasks via XCom.
    """
    from natgas.db.connection import get_session
    from sqlalchemy import text
    import pandas as pd

    with get_session() as session:
        sql = text("""
            SELECT
                week_ending_date,
                region,
                working_gas_bcf,
                net_change_bcf,
                analyst_consensus_bcf,
                storage_surprise_bcf,
                model_name,
                seasonal_mask,
                pop_weighted_hdd_corrected,
                pop_weighted_cdd_corrected,
                lng_exports_bcfd,
                dry_gas_production_bcfd,
                pipeline_exports_mexico_bcfd,
                residential_commercial_demand_bcfd,
                front_month_settle,
                directional_signal,
                confidence_score,
                pre_release_price_drift
            FROM weekly_analysis_master
            WHERE week_ending_date >= NOW() - INTERVAL '3 years'
            ORDER BY week_ending_date DESC
        """)
        rows = session.execute(sql).fetchall()
        columns = sql.columns if hasattr(sql, "columns") else [
            "week_ending_date", "region", "working_gas_bcf", "net_change_bcf",
            "analyst_consensus_bcf", "storage_surprise_bcf", "model_name",
            "seasonal_mask", "pop_weighted_hdd_corrected", "pop_weighted_cdd_corrected",
            "lng_exports_bcfd", "dry_gas_production_bcfd", "pipeline_exports_mexico_bcfd",
            "residential_commercial_demand_bcfd", "front_month_settle",
            "directional_signal", "confidence_score", "pre_release_price_drift",
        ]

    n_rows = len(rows)
    logger.info("Loaded %d rows from weekly_analysis_master", n_rows)
    return {"n_rows": n_rows, "loaded": True}


def retrain_storage_model(**context) -> dict:
    """
    Retrain Ridge + XGBoost storage draw model on latest data.
    Saves model weights to data/weights/. Logs cross-validation RMSE.
    """
    import pandas as pd
    from pathlib import Path
    from natgas.analysis.storage_model import StorageDrawModel
    from natgas.db.connection import get_session
    from sqlalchemy import text
    import joblib

    with get_session() as session:
        sql = text("""
            SELECT
                week_ending_date,
                net_change_bcf,
                MAX(CASE WHEN region = 'national' THEN pop_weighted_hdd_corrected END) AS hdd_national,
                MAX(CASE WHEN region = 'midwest'  THEN pop_weighted_hdd_corrected END) AS hdd_midwest,
                MAX(CASE WHEN region = 'northeast' THEN pop_weighted_hdd_corrected END) AS hdd_northeast,
                MAX(CASE WHEN region = 'texas'     THEN pop_weighted_hdd_corrected END) AS hdd_texas,
                MAX(CASE WHEN region = 'southeast' THEN pop_weighted_hdd_corrected END) AS hdd_southeast,
                MAX(CASE WHEN region = 'national' THEN pop_weighted_cdd_corrected END) AS cdd_national,
                MAX(CASE WHEN region = 'midwest'  THEN pop_weighted_cdd_corrected END) AS cdd_midwest,
                MAX(CASE WHEN region = 'northeast' THEN pop_weighted_cdd_corrected END) AS cdd_northeast,
                MAX(CASE WHEN region = 'texas'     THEN pop_weighted_cdd_corrected END) AS cdd_texas,
                MAX(CASE WHEN region = 'southeast' THEN pop_weighted_cdd_corrected END) AS cdd_southeast,
                MAX(lng_exports_bcfd) AS lng_exports_bcfd,
                MAX(dry_gas_production_bcfd) AS dry_gas_production_bcfd,
                MAX(residential_commercial_demand_bcfd) AS residential_commercial_demand_bcfd,
                MAX(pipeline_exports_mexico_bcfd) AS pipeline_exports_mexico_bcfd,
                MAX(analyst_consensus_bcf) AS analyst_consensus_bcf
            FROM weekly_analysis_master
            WHERE net_change_bcf IS NOT NULL
              AND week_ending_date >= NOW() - INTERVAL '5 years'
            GROUP BY week_ending_date
            ORDER BY week_ending_date
        """)
        rows = session.execute(sql).fetchall()

    if not rows:
        logger.warning("No training data available; skipping model retraining.")
        return {"trained": False, "n_samples": 0}

    col_names = [
        "week_ending_date", "net_change_bcf",
        "pop_weighted_hdd_corrected_national", "pop_weighted_hdd_corrected_midwest",
        "pop_weighted_hdd_corrected_northeast", "pop_weighted_hdd_corrected_texas",
        "pop_weighted_hdd_corrected_southeast",
        "pop_weighted_cdd_corrected_national", "pop_weighted_cdd_corrected_midwest",
        "pop_weighted_cdd_corrected_northeast", "pop_weighted_cdd_corrected_texas",
        "pop_weighted_cdd_corrected_southeast",
        "lng_exports_bcfd", "dry_gas_production_bcfd",
        "residential_commercial_demand_bcfd", "pipeline_exports_mexico_bcfd",
        "analyst_consensus_bcf",
    ]
    df = pd.DataFrame(rows, columns=col_names)

    model = StorageDrawModel()
    features = model.build_features(df)
    target = df["net_change_bcf"]

    valid_mask = features.notna().all(axis=1) & target.notna()
    X = features[valid_mask]
    y = target[valid_mask]

    if len(X) < 10:
        logger.warning("Insufficient samples (%d) for model training; need at least 10.", len(X))
        return {"trained": False, "n_samples": len(X)}

    metrics = model.fit(X, y)

    # Persist model weights
    weights_dir = Path("data/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, weights_dir / "storage_draw_model.joblib")

    logger.info(
        "Storage model retrained: n=%d | Ridge CV RMSE=%.2f | XGB CV RMSE=%s",
        len(X),
        metrics.get("ridge_cv_rmse", 0),
        f"{metrics.get('xgb_cv_rmse', 'N/A'):.2f}" if metrics.get("xgb_cv_rmse") else "N/A",
    )
    return {
        "trained": True,
        "n_samples": len(X),
        "ridge_cv_rmse": metrics.get("ridge_cv_rmse"),
        "xgb_cv_rmse": metrics.get("xgb_cv_rmse"),
    }


def generate_next_week_model_estimate(**context) -> dict:
    """
    Use freshly retrained model + latest bias-corrected HDD/CDD forecasts
    to generate the storage draw estimate for the NEXT EIA report.
    Pre-populates signal_log so eia_storage_dag can read it on release day.
    """
    import json
    from datetime import date
    from pathlib import Path
    import pandas as pd
    import joblib
    from natgas.calendar.trading_calendar import get_eia_release_date, get_report_week_ending
    from natgas.db.connection import get_session
    from sqlalchemy import text

    # Determine next report's week-ending date
    today = context["execution_date"].date()
    next_release = get_eia_release_date(today + timedelta(days=7))
    next_report_date = get_report_week_ending(next_release)

    weights_path = Path("data/weights/storage_draw_model.joblib")
    if not weights_path.exists():
        logger.warning("No trained model found; skipping next-week estimate.")
        return {"estimated": False}

    model = joblib.load(weights_path)

    with get_session() as session:
        # Fetch most recent bias-corrected HDD/CDD by region
        sql = text("""
            SELECT
                region,
                AVG(pop_weighted_hdd_corrected) AS hdd,
                AVG(pop_weighted_cdd_corrected) AS cdd
            FROM hdd_cdd_bias_corrected_by_model
            WHERE forecast_date = (
                SELECT MAX(forecast_date)
                FROM hdd_cdd_bias_corrected_by_model
            )
            AND model_name IN ('GFS', 'GEFS')
            AND lead_days BETWEEN 1 AND 7
            GROUP BY region
        """)
        hdd_rows = session.execute(sql).fetchall()

        # Fetch latest supply/demand data
        sql_sd = text("""
            SELECT lng_exports_bcfd, dry_gas_production_bcfd,
                   pipeline_exports_mexico_bcfd, residential_commercial_demand_bcfd
            FROM supply_demand_weekly
            ORDER BY published_at DESC
            LIMIT 1
        """)
        sd_row = session.execute(sql_sd).fetchone()

        sql_consensus = text("""
            SELECT mean_estimate_bcf
            FROM analyst_consensus_weekly
            WHERE report_date = :report_date
            ORDER BY published_at DESC
            LIMIT 1
        """)
        consensus_row = session.execute(sql_consensus, {"report_date": next_report_date}).fetchone()

    # Build feature dict
    feature_dict = {"week_ending_date": next_report_date}
    for row in hdd_rows:
        region = row[0]
        feature_dict[f"pop_weighted_hdd_corrected_{region}"] = float(row[1] or 0)
        feature_dict[f"pop_weighted_cdd_corrected_{region}"] = float(row[2] or 0)

    if sd_row:
        feature_dict["lng_exports_bcfd"] = float(sd_row[0] or 0)
        feature_dict["dry_gas_production_bcfd"] = float(sd_row[1] or 0)
        feature_dict["pipeline_exports_mexico_bcfd"] = float(sd_row[2] or 0)
        feature_dict["residential_commercial_demand_bcfd"] = float(sd_row[3] or 0)

    df_pred = pd.DataFrame([feature_dict])
    features = model.build_features(df_pred)

    try:
        estimate = model.predict(features)[0]
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        return {"estimated": False}

    consensus_bcf = float(consensus_row[0]) if consensus_row and consensus_row[0] else None

    with get_session() as session:
        sql_insert = text("""
            INSERT INTO signal_log
                (report_date, model_estimate_bcf, analyst_consensus_bcf,
                 model_vs_consensus_bcf)
            VALUES
                (:report_date, :model_estimate, :consensus, :model_vs_consensus)
            ON CONFLICT (report_date) DO UPDATE SET
                model_estimate_bcf      = EXCLUDED.model_estimate_bcf,
                model_vs_consensus_bcf  = EXCLUDED.model_vs_consensus_bcf
        """)
        model_vs_consensus = round(estimate - consensus_bcf, 3) if consensus_bcf else None
        session.execute(sql_insert, {
            "report_date": next_report_date,
            "model_estimate": round(float(estimate), 3),
            "consensus": consensus_bcf,
            "model_vs_consensus": model_vs_consensus,
        })

    logger.info(
        "Next week model estimate for %s: %.1f Bcf | consensus=%s | model_vs_consensus=%s",
        next_report_date, estimate,
        f"{consensus_bcf:.1f}" if consensus_bcf else "N/A",
        f"{model_vs_consensus:+.1f}" if model_vs_consensus else "N/A",
    )
    return {
        "estimated": True,
        "next_report_date": str(next_report_date),
        "model_estimate_bcf": round(float(estimate), 3),
        "consensus_bcf": consensus_bcf,
    }


def run_model_benchmarking(**context) -> dict:
    """
    Build 90-day weather model accuracy scorecard and log to model_stability_log.
    """
    from natgas.analysis.model_benchmarking import build_90day_scorecard, detect_model_drift
    from natgas.alerts.notifier import send_slack_alert, format_model_drift_alert
    from natgas.db.connection import get_session
    from sqlalchemy import text

    exec_dt = context["execution_date"].date()

    with get_session() as session:
        scorecard = build_90day_scorecard(
            db_session=session,
            as_of_date=exec_dt,
            models=MODELS,
            regions=REGIONS,
        )

    if scorecard.empty:
        logger.info("No benchmarking data available yet.")
        return {"rows": 0}

    # Check for drift on each model/region
    alerts_sent = 0
    with get_session() as session:
        for _, row in scorecard.iterrows():
            drift_result = detect_model_drift(
                model_name=row["model_name"],
                region=row.get("region", "national"),
                lead_days=int(row.get("lead_days", 7)),
                residual_bias=float(row.get("residual_bias", 0)),
                db_session=session,
            )
            if drift_result.get("alert_triggered"):
                msg = format_model_drift_alert(
                    model_name=row["model_name"],
                    region=row.get("region", "national"),
                    lead_days=int(row.get("lead_days", 7)),
                    residual_bias=float(row.get("residual_bias", 0)),
                    z_score=drift_result.get("z_score", 0),
                    consecutive=drift_result.get("consecutive_violations", 1),
                )
                send_slack_alert(msg)
                alerts_sent += 1

    logger.info("Benchmarking complete: %d model/region/lead combos | %d drift alerts", len(scorecard), alerts_sent)
    return {"rows": len(scorecard), "drift_alerts": alerts_sent}


def update_seasonal_tracker(**context) -> dict:
    """
    Update seasonal surplus/deficit tracker: compute percentile, project EOS storage.
    """
    from datetime import date
    import pandas as pd
    from natgas.analysis.seasonal_tracker import (
        compute_storage_percentile,
        project_end_of_season_storage,
        classify_regime,
        WITHDRAWAL_END,
        INJECTION_END,
    )
    from natgas.db.connection import get_session
    from sqlalchemy import text

    exec_dt = context["execution_date"].date()
    month = exec_dt.month
    season = "winter" if month in (11, 12, 1, 2, 3) else "summer"
    year = exec_dt.year

    with get_session() as session:
        # Fetch current season's storage data
        if season == "winter":
            season_start = date(year - 1 if month < 4 else year, 11, 1)
        else:
            season_start = date(year, 4, 1)

        sql_current = text("""
            SELECT DISTINCT ON (report_date)
                report_date, working_gas_bcf
            FROM eia_storage_weekly
            WHERE region = 'total'
              AND report_date >= :season_start
            ORDER BY report_date, published_at DESC
        """)
        current_rows = session.execute(sql_current, {"season_start": season_start}).fetchall()

        # Fetch same week historical values for percentile
        latest_date = max((r[0] for r in current_rows), default=exec_dt)
        sql_hist = text("""
            SELECT DISTINCT ON (report_date)
                working_gas_bcf
            FROM eia_storage_weekly
            WHERE region = 'total'
              AND EXTRACT(WEEK FROM report_date) = EXTRACT(WEEK FROM :ref_date::date)
              AND report_date < :season_start
            ORDER BY report_date, published_at DESC
            LIMIT 15
        """)
        hist_rows = session.execute(sql_hist, {
            "ref_date": latest_date,
            "season_start": season_start,
        }).fetchall()

    historical_vals = [float(r[0]) for r in hist_rows if r[0] is not None]
    current_bcf = float(current_rows[-1][1]) if current_rows and current_rows[-1][1] else None

    percentile = compute_storage_percentile(current_bcf, historical_vals) if (current_bcf and historical_vals) else 50.0
    regime = classify_regime(percentile)

    # Project end-of-season
    projected_bcf, r_squared = None, None
    if len(current_rows) >= 3:
        import pandas as pd
        season_df = pd.DataFrame(current_rows, columns=["report_date", "working_gas_bcf"])
        target_date = date(*WITHDRAWAL_END) if season == "winter" else date(*INJECTION_END)
        if target_date.year < exec_dt.year:
            target_date = target_date.replace(year=exec_dt.year)
        try:
            projected_bcf, r_squared = project_end_of_season_storage(season_df, target_date)
        except ValueError as exc:
            logger.warning("Projection failed: %s", exc)

    logger.info(
        "Seasonal tracker: %s regime | current=%.1f Bcf | percentile=%.1f | "
        "EOS projection=%.1f Bcf (R²=%.3f)",
        regime, current_bcf or 0, percentile,
        projected_bcf or 0, r_squared or 0,
    )
    return {
        "regime": regime,
        "current_bcf": current_bcf,
        "storage_percentile": percentile,
        "projected_eos_bcf": projected_bcf,
        "r_squared": r_squared,
        "season": season,
    }


def retrain_price_sensitivity_model(**context) -> dict:
    """
    Retrain price sensitivity model. Only runs quarterly (first week of Jan/Apr/Jul/Oct).
    """
    import pandas as pd
    from pathlib import Path
    import joblib
    from natgas.analysis.price_sensitivity import PriceSensitivityModel
    from natgas.db.connection import get_session
    from sqlalchemy import text

    exec_dt = context["execution_date"].date()
    # Only retrain quarterly
    if exec_dt.month not in (1, 4, 7, 10) or exec_dt.day > 7:
        logger.info("Skipping price sensitivity retraining (not quarterly window).")
        return {"retrained": False}

    with get_session() as session:
        sql = text("""
            SELECT
                s.report_date,
                s.model_vs_consensus_bcf AS storage_surprise_bcf,
                s.pre_release_price_drift,
                s.actual_signal,
                i.price_t_minus_5min,
                i.price_t_plus_15min,
                (i.price_t_plus_15min - i.price_t_minus_5min) AS price_t_plus_15min_vs_t_minus_5min
            FROM signal_log s
            LEFT JOIN ngas_futures_release_day_intraday i
                ON i.report_date = s.report_date
            WHERE s.actual_bcf IS NOT NULL
              AND s.report_date >= NOW() - INTERVAL '3 years'
            ORDER BY s.report_date
        """)
        rows = session.execute(sql).fetchall()

    if not rows:
        logger.warning("No price sensitivity training data available.")
        return {"retrained": False, "n_samples": 0}

    df = pd.DataFrame(rows, columns=[
        "report_date", "storage_surprise_bcf", "pre_release_price_drift",
        "actual_signal", "price_t_minus_5min", "price_t_plus_15min",
        "price_t_plus_15min_vs_t_minus_5min",
    ])

    # Add seasonal flag
    df["is_winter"] = pd.to_datetime(df["report_date"]).dt.month.isin([11, 12, 1, 2, 3]).astype(int)

    psm = PriceSensitivityModel()
    metrics = psm.fit(df)

    weights_dir = Path("data/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(psm, weights_dir / "price_sensitivity_model.joblib")

    logger.info(
        "Price sensitivity model retrained: n=%d | R²=%.3f | MAE=%.4f",
        len(df), metrics.get("r_squared", 0), metrics.get("mae", 0),
    )
    return {"retrained": True, "n_samples": len(df), **metrics}


def refresh_materialized_view(**context) -> dict:
    """REFRESH weekly_analysis_master materialized view concurrently."""
    from natgas.db.connection import get_session
    from sqlalchemy import text

    with get_session() as session:
        session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY weekly_analysis_master"))

    logger.info("weekly_analysis_master materialized view refreshed.")
    return {"refreshed": True}


def send_weekly_summary(**context) -> None:
    """Send weekly analysis summary to Slack."""
    from natgas.alerts.notifier import send_slack_alert

    ti = context["ti"]
    seasonal = ti.xcom_pull(task_ids="update_seasonal_tracker") or {}
    model_result = ti.xcom_pull(task_ids="retrain_storage_model") or {}
    next_est = ti.xcom_pull(task_ids="generate_next_week_model_estimate") or {}
    bench = ti.xcom_pull(task_ids="run_model_benchmarking") or {}

    regime = seasonal.get("regime", "Unknown")
    current_bcf = seasonal.get("current_bcf", 0)
    percentile = seasonal.get("storage_percentile", 50)
    projected = seasonal.get("projected_eos_bcf")
    season = seasonal.get("season", "")
    next_estimate = next_est.get("model_estimate_bcf")
    next_report = next_est.get("next_report_date", "N/A")
    drift_alerts = bench.get("drift_alerts", 0)

    regime_emoji = ":chart_with_upwards_trend:" if regime == "Deficit" else (":chart_with_downwards_trend:" if regime == "Surplus" else ":bar_chart:")

    proj_str = f"{projected:.0f} Bcf" if projected else "N/A"
    next_est_str = f"{next_estimate:.1f} Bcf" if next_estimate else "N/A"

    msg = (
        f":newspaper: *NatGas Weekly Analysis Summary*\n"
        f"{regime_emoji} *Storage Regime:* {regime} ({percentile:.0f}th percentile)\n"
        f"Current working gas: {current_bcf:.0f} Bcf\n"
        f"EOS projection ({season}): {proj_str}\n"
        f"---\n"
        f":crystal_ball: *Next week model estimate ({next_report}):* {next_est_str}\n"
        f":warning: Model drift alerts: {drift_alerts}"
    )
    send_slack_alert(msg)


with DAG(
    dag_id="weekly_analysis_dag",
    default_args=default_args,
    description="Weekly analysis: model retraining, benchmarking, seasonal tracker, view refresh",
    schedule_interval="0 20 * * 4,5",  # 3:00 PM ET Thu+Fri = 20:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["analysis", "model", "benchmarking"],
) as dag:

    t_load = PythonOperator(
        task_id="load_training_data",
        python_callable=load_training_data,
    )

    t_retrain = PythonOperator(
        task_id="retrain_storage_model",
        python_callable=retrain_storage_model,
    )

    t_next_est = PythonOperator(
        task_id="generate_next_week_model_estimate",
        python_callable=generate_next_week_model_estimate,
    )

    t_benchmark = PythonOperator(
        task_id="run_model_benchmarking",
        python_callable=run_model_benchmarking,
    )

    t_seasonal = PythonOperator(
        task_id="update_seasonal_tracker",
        python_callable=update_seasonal_tracker,
    )

    t_price_sens = PythonOperator(
        task_id="retrain_price_sensitivity_model",
        python_callable=retrain_price_sensitivity_model,
    )

    t_refresh = PythonOperator(
        task_id="refresh_materialized_view",
        python_callable=refresh_materialized_view,
    )

    t_summary = PythonOperator(
        task_id="send_weekly_summary",
        python_callable=send_weekly_summary,
    )

    # Load data → retrain model → generate next-week estimate (dependent on fresh model)
    #           → benchmarking + seasonal in parallel → price sensitivity → refresh → summary
    t_load >> t_retrain >> t_next_est
    t_retrain >> [t_benchmark, t_seasonal] >> t_price_sens >> t_refresh >> t_summary
    t_next_est >> t_refresh
