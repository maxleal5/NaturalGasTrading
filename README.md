# NatGas Intelligence Platform

A production-grade pipeline for natural gas trading decisions — ingesting EIA storage data, weather model forecasts, and analyst consensus estimates to generate weekly trade signals with full model benchmarking and a real-time Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Pipelines](#data-pipelines)
- [Analysis Modules](#analysis-modules)
- [Airflow DAGs](#airflow-dags)
- [Dashboard](#dashboard)
- [Database](#database)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

Every Thursday at 10:30 AM ET the EIA publishes its weekly natural gas storage report. The platform:

1. Collects analyst consensus estimates Mon–Wed before the release
2. Ingests the EIA number seconds after publication (with exponential backoff)
3. Computes the storage surprise and a directional trade signal adjusted for whisper numbers and seasonal regime
4. Retrains the Ridge + XGBoost storage draw ensemble weekly
5. Archives weather model forecasts before the 90-day TimescaleDB retention window expires
6. Surfaces everything in a 5-page Streamlit dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Apache Airflow                        │
│  trading_calendar_dag  ──►  consensus_estimate_dag           │
│                         ──►  eia_storage_dag                 │
│                         ──►  weekly_analysis_dag             │
│  futures_price_dag                                           │
│  weather_model_dag  ──►  bias_correction_dag                 │
│  archive_weather_dag (Sunday)                                │
└────────────────────────┬────────────────────────────────────┘
                         │
               ┌─────────▼──────────┐
               │   TimescaleDB/PG   │
               │  (11 tables + MV)  │
               └─────────┬──────────┘
                         │
               ┌─────────▼──────────┐
               │  Streamlit Dashboard│
               │  (5 pages, port 8501)│
               └────────────────────┘
```

**Stack:** Python 3.11 · Apache Airflow 2.8 · TimescaleDB (PostgreSQL 15) · scikit-learn · XGBoost · Streamlit · Plotly · Docker Compose

---

## Project Structure

```
NaturalGasTrading/
├── dags/                          # Airflow DAG definitions
│   ├── trading_calendar_dag.py    # Derives EIA release date weekly
│   ├── consensus_estimate_dag.py  # Collects Mon–Wed analyst estimates
│   ├── eia_storage_dag.py         # Release-day ingestion + signal
│   ├── weekly_analysis_dag.py     # Post-release model retraining
│   ├── futures_price_dag.py       # Daily settlement prices
│   ├── weather_model_dag.py       # GFS/GEFS/AIFS forecast ingestion
│   ├── bias_correction_dag.py     # MOS bias correction per model
│   └── archive_weather_dag.py     # Sunday archival before retention purge
│
├── natgas/                        # Core library (importable)
│   ├── pipelines/                 # Data ingestion
│   │   ├── eia_storage.py         # EIA API with tenacity backoff
│   │   ├── analyst_consensus.py   # Barchart + manual consensus
│   │   ├── futures_prices.py      # Daily settlement (CME/EIA)
│   │   ├── weather_models.py      # GFS/commercial model fetchers
│   │   └── supply_demand.py       # EIA supply/demand series
│   ├── analysis/                  # Quantitative models
│   │   ├── storage_model.py       # Ridge + XGBoost ensemble
│   │   ├── surprise_signal.py     # Signal generation + whisper adjust
│   │   ├── seasonal_tracker.py    # Surplus/deficit regime classifier
│   │   ├── bias_correction.py     # MOS bias corrector
│   │   ├── price_sensitivity.py   # Price sensitivity model
│   │   └── model_benchmarking.py  # 90-day accuracy scorecard + drift
│   ├── calendar/
│   │   └── trading_calendar.py    # EIA release date logic (CME holidays)
│   ├── population_weights/
│   │   └── weight_masks.py        # Population-weighted HDD/CDD by region
│   ├── alerts/
│   │   └── notifier.py            # Slack alerts
│   └── db/
│       └── connection.py          # SQLAlchemy session + engine
│
├── dashboard/
│   └── app.py                     # Streamlit app (5 pages)
│
├── db/
│   └── schema.sql                 # Full DDL — tables, indexes, MV
│
├── data/
│   └── weights/                   # Saved model artefacts (.joblib)
│
├── tests/                         # pytest unit tests
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Data Pipelines

| Module | Source | Cadence | Key Output |
|--------|--------|---------|------------|
| `eia_storage.py` | EIA Open Data API | Thursday 10:30 AM ET | `eia_storage_weekly` |
| `analyst_consensus.py` | Barchart / manual | Mon–Wed | `analyst_consensus_weekly` |
| `futures_prices.py` | CME / EIA | Daily EOD | `futures_daily_settlement` |
| `weather_models.py` | GFS, GEFS, AIFS | Daily | `weather_forecast_raw` (hypertable) |
| `supply_demand.py` | EIA API | Weekly | `supply_demand_weekly` |

`weather_forecast_raw` is a TimescaleDB **hypertable** with a **90-day retention policy** — the `archive_weather_dag` aggregates and moves data to `weather_forecast_archive` before expiry.

---

## Analysis Modules

### Storage Draw Model (`storage_model.py`)
Ridge regression + XGBoost ensemble trained on weekly HDD/CDD features, production/LNG exports, and lagged storage. Returns `ridge_estimate_bcf`, `xgb_estimate_bcf`, `ensemble_estimate_bcf`, and `forecast_uncertainty_score`. Model weights persist to `data/weights/storage_draw_model.joblib`.

### Signal Generation (`surprise_signal.py`)
Computes `storage_surprise_bcf = actual − consensus` then applies:
- **Whisper number adjustment** — pre-release futures drift shifts the effective consensus
- **Regime multiplier** — deficit years amplify the signal; surplus years dampen it
- Returns `(signal: int, confidence: float)` where signal ∈ {−1, 0, 1}

### Seasonal Tracker (`seasonal_tracker.py`)
Computes storage percentile vs the 5-year historical range, projects end-of-season storage via linear regression, and classifies the regime as `deficit`, `balanced`, or `surplus`.

### Bias Correction (`bias_correction.py`)
Model Output Statistics (MOS) corrector. Computes rolling mean bias per model/region/lead and returns corrected HDD/CDD. Falls back to raw values when fewer than `mos_window_days` observations exist.

### Model Benchmarking (`model_benchmarking.py`)
Builds a 90-day MAE/RMSE/bias scorecard by model × region × lead time. `detect_model_drift` fires a Slack alert when residual bias exceeds 2σ for 3 consecutive runs.

### Price Sensitivity (`price_sensitivity.py`)
Regresses front-month futures price changes against storage surprises and weather deviations to estimate $/Bcf sensitivity by season.

---

## Airflow DAGs

| DAG | Schedule (UTC) | Purpose |
|-----|----------------|---------|
| `trading_calendar_dag` | `0 12 * * 1` | Sets `eia_release_info` Airflow Variable for the week |
| `consensus_estimate_dag` | `0 13 * * 1,2,3` | Fetches analyst estimates Mon–Wed (8 AM ET) |
| `eia_storage_dag` | `35 15 * * 4,5` | Release-day ingestion + surprise + signal (10:35 AM ET) |
| `weekly_analysis_dag` | `0 20 * * 4,5` | Post-release model retraining + MV refresh (3 PM ET) |
| `futures_price_dag` | `0 22 * * 1-5` | Daily settlement prices (5 PM ET) |
| `weather_model_dag` | `0 6,18 * * *` | GFS/GEFS/AIFS 0z and 12z runs |
| `bias_correction_dag` | `30 7,19 * * *` | MOS bias correction after each model run |
| `archive_weather_dag` | `0 6 * * 0` | Sunday 1 AM ET — archive 75–90 day old forecasts |

The `eia_storage_dag` EIA fetch task has `retries=8, retry_delay=2min` to handle delayed EIA publications.

---

## Dashboard

Five-page Streamlit app at `http://localhost:8501`:

| Page | Content |
|------|---------|
| **Overview** | Signal badge, storage vs 5-year average, seasonal regime, all-time win rate |
| **Storage** | Region selector, lookback slider, storage level + weekly change charts |
| **Weather Models** | Bias-corrected vs raw HDD/CDD by model, 7-day summary table |
| **Model Monitor** | MAE/RMSE/bias pivot heatmaps by model × region × lead, drift log |
| **Signal History** | Win-rate metrics, surprise vs price move scatter (OLS trendline), full log |

All DB queries are cached with `@st.cache_data(ttl=300)`.

---

## Database

11 tables + 1 materialized view (`weekly_analysis_master`) in TimescaleDB/PostgreSQL 15.

| Table | Purpose |
|-------|---------|
| `eia_storage_weekly` | Weekly storage readings with point-in-time integrity |
| `data_revision_log` | Tracks EIA revisions |
| `analyst_consensus_weekly` | Pre-release estimates by source |
| `signal_log` | One row per report week — surprise, signal, outcome |
| `futures_daily_settlement` | CME front-month settlements |
| `weather_forecast_raw` | Hypertable — grid-point HDD/CDD forecasts (90-day retention) |
| `weather_forecast_archive` | Long-term aggregated archive |
| `weather_model_accuracy` | MAE/RMSE scorecard by model × region × lead |
| `supply_demand_weekly` | Production, LNG exports, pipeline flows |
| `price_sensitivity_params` | Quarterly $/Bcf sensitivity estimates |
| `model_benchmarking_log` | Drift detection history |

Apply the schema:

```bash
psql -U natgas -d natgas -f db/schema.sql
```

---

## Getting Started

### Prerequisites

- Docker Desktop
- An `.env` file in the project root (see [Configuration](#configuration))

### Start all services

```bash
docker compose up -d
```

This starts:
- **TimescaleDB** on port `5432`
- **Airflow webserver** on port `8080`
- **Airflow scheduler**
- **Streamlit dashboard** on port `8501`

### First-time Airflow setup

```bash
# Initialise the Airflow metadata DB (runs inside the container)
docker compose run --rm airflow-webserver airflow db migrate
docker compose run --rm airflow-webserver airflow users create \
    --username admin --password admin \
    --firstname Admin --lastname User \
    --role Admin --email admin@example.com
```

### Apply the database schema

```bash
docker compose exec db psql -U natgas -d natgas -f /docker-entrypoint-initdb.d/schema.sql
# Or from the host:
psql -h localhost -U natgas -d natgas -f db/schema.sql
```

### Local development (no Docker)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Start the dashboard only
streamlit run dashboard/app.py
```

---

## Configuration

Create a `.env` file at the project root:

```env
# Database
POSTGRES_PASSWORD=natgas_dev

# Airflow
AIRFLOW_FERNET_KEY=<generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
AIRFLOW_SECRET_KEY=<random string>

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Data sources (set as Airflow Variables or env vars)
EIA_API_KEY=<your EIA API key>
BARCHART_API_KEY=<optional>
ALERT_EMAIL=you@example.com
```

Set Airflow Variables via the UI (`Admin → Variables`) or CLI:

```bash
airflow variables set EIA_API_KEY "<key>"
airflow variables set SLACK_WEBHOOK_URL "<url>"
airflow variables set ALERT_EMAIL "you@example.com"
```

---

## Testing

```bash
# Run all tests with coverage
pytest

# Run a specific module
pytest tests/test_storage_model.py -v

# Coverage report only
pytest --cov=natgas --cov-report=html
```

Test files:

| File | Covers |
|------|--------|
| `test_surprise_signal.py` | Surprise computation, signal direction, whisper/regime adjustments |
| `test_bias_correction.py` | MOS corrector, cache behaviour, zero-bias edge cases |
| `test_seasonal_tracker.py` | Percentile, end-of-season projection, regime classification |
| `test_trading_calendar.py` | EIA release date logic, CME holiday handling |
| `test_storage_model.py` | Ridge+XGBoost feature building, fit/predict contracts |
| `test_pipelines.py` | Consensus, futures, supply/demand pipeline unit tests (mocked) |
| `test_model_benchmarking.py` | MAE/RMSE/bias scorecard, overlap edge cases, min-obs guard |