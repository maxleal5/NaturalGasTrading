-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- EIA Storage Weekly (Pipeline 1)
-- Append-only: never UPDATE, always INSERT new row for revisions
-- ============================================================
CREATE TABLE IF NOT EXISTS eia_storage_weekly (
    id BIGSERIAL PRIMARY KEY,
    report_date DATE NOT NULL,               -- week the storage data refers to
    published_at TIMESTAMPTZ NOT NULL,        -- exact timestamp this number became public
    region VARCHAR(30) NOT NULL DEFAULT 'total',  -- total, east, midwest, mountain, pacific, south_central
    working_gas_bcf NUMERIC(10,3),
    net_change_bcf NUMERIC(10,3),
    five_year_avg_bcf NUMERIC(10,3),
    year_ago_bcf NUMERIC(10,3),
    analyst_consensus_bcf NUMERIC(10,3),
    storage_surprise_bcf NUMERIC(10,3),      -- actual - analyst_consensus (signed)
    revision_number INTEGER NOT NULL DEFAULT 0,
    data_source VARCHAR(50) DEFAULT 'EIA_API',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eia_storage_report_date ON eia_storage_weekly(report_date, region);
CREATE INDEX IF NOT EXISTS idx_eia_storage_published_at ON eia_storage_weekly(published_at);
-- Enforce point-in-time integrity: unique per report_date + region + published_at
CREATE UNIQUE INDEX IF NOT EXISTS uq_eia_storage ON eia_storage_weekly(report_date, region, published_at);

-- Data revision log
CREATE TABLE IF NOT EXISTS data_revision_log (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    report_date DATE NOT NULL,
    region VARCHAR(30),
    field_name VARCHAR(100) NOT NULL,
    old_value NUMERIC,
    new_value NUMERIC,
    delta NUMERIC,
    published_at TIMESTAMPTZ NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Analyst Consensus Weekly (Pipeline 3)
-- ============================================================
CREATE TABLE IF NOT EXISTS analyst_consensus_weekly (
    id BIGSERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    mean_estimate_bcf NUMERIC(10,3),
    high_estimate_bcf NUMERIC(10,3),
    low_estimate_bcf NUMERIC(10,3),
    respondent_count INTEGER,
    source_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_consensus ON analyst_consensus_weekly(report_date, published_at, source_name);

-- ============================================================
-- Weather Forecast Raw (Pipeline 2) — TimescaleDB hypertable
-- Hot tier: 0-90 days full resolution; cold tier: aggregated only
-- ============================================================
CREATE TABLE IF NOT EXISTS weather_forecast_raw (
    time TIMESTAMPTZ NOT NULL,               -- valid_datetime (for TimescaleDB partitioning)
    forecast_init_date TIMESTAMPTZ NOT NULL,  -- model initialization time (00Z or 12Z)
    valid_date DATE NOT NULL,                 -- date this forecast applies to
    lead_days INTEGER NOT NULL,              -- valid_date - init_date in days
    model_name VARCHAR(50) NOT NULL,         -- GFS, GEFS, EURO_HRES, EURO_ENS, AIFS, GRAPHCAST
    model_version VARCHAR(50),
    latitude NUMERIC(6,3) NOT NULL,
    longitude NUMERIC(6,3) NOT NULL,
    t2m_celsius NUMERIC(6,2),                -- 2m temperature in Celsius
    t2m_fahrenheit NUMERIC(6,2),             -- 2m temperature in Fahrenheit
    precip_mm NUMERIC(8,3),                  -- precipitation mm (NULL for AI models that omit it)
    z500_gpm NUMERIC(8,2),                   -- 500mb geopotential height
    hdd_raw NUMERIC(6,2),                    -- HDD vs 65F baseline, pre-bias-correction
    cdd_raw NUMERIC(6,2),                    -- CDD vs 65F baseline, pre-bias-correction
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Convert to TimescaleDB hypertable
SELECT create_hypertable('weather_forecast_raw', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_weather_model ON weather_forecast_raw(model_name, forecast_init_date);
CREATE INDEX IF NOT EXISTS idx_weather_location ON weather_forecast_raw(latitude, longitude);
-- 90-day retention policy
SELECT add_retention_policy('weather_forecast_raw', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================
-- HDD/CDD Daily by Model (pre-correction) — aggregated/cold tier
-- ============================================================
CREATE TABLE IF NOT EXISTS hdd_cdd_daily_by_model (
    id BIGSERIAL PRIMARY KEY,
    forecast_date DATE NOT NULL,
    valid_date DATE NOT NULL,
    lead_days INTEGER NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    region VARCHAR(50) NOT NULL,            -- e.g. 'national', 'midwest', 'northeast', 'texas', 'southeast'
    seasonal_mask VARCHAR(10) NOT NULL,     -- 'winter' or 'summer'
    pop_weighted_hdd NUMERIC(8,3),
    pop_weighted_cdd NUMERIC(8,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_hdd_cdd_model_date ON hdd_cdd_daily_by_model(model_name, forecast_date, valid_date);
CREATE INDEX IF NOT EXISTS idx_hdd_cdd_region ON hdd_cdd_daily_by_model(region, valid_date);

-- ============================================================
-- HDD/CDD Bias Corrected (post-MOS)
-- ============================================================
CREATE TABLE IF NOT EXISTS hdd_cdd_bias_corrected_by_model (
    id BIGSERIAL PRIMARY KEY,
    forecast_date DATE NOT NULL,
    valid_date DATE NOT NULL,
    lead_days INTEGER NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    region VARCHAR(50) NOT NULL,
    seasonal_mask VARCHAR(10) NOT NULL,
    pop_weighted_hdd_raw NUMERIC(8,3),
    pop_weighted_cdd_raw NUMERIC(8,3),
    bias_hdd NUMERIC(8,3),                  -- rolling 30-day mean bias
    bias_cdd NUMERIC(8,3),
    pop_weighted_hdd_corrected NUMERIC(8,3),
    pop_weighted_cdd_corrected NUMERIC(8,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_hdd_bc_model_date ON hdd_cdd_bias_corrected_by_model(model_name, forecast_date, valid_date);

-- ============================================================
-- Natural Gas Futures Daily (Pipeline 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS ngas_futures_daily (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL UNIQUE,
    front_month_settle NUMERIC(8,4),
    twelve_month_strip NUMERIC(8,4),
    open_interest BIGINT,
    volume BIGINT,
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Natural Gas Futures Release Day Intraday
CREATE TABLE IF NOT EXISTS ngas_futures_release_day_intraday (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    report_date DATE NOT NULL,               -- EIA report_date this release corresponds to
    price_t_minus_5min NUMERIC(8,4),
    price_t_plus_1min NUMERIC(8,4),
    price_t_plus_5min NUMERIC(8,4),
    price_t_plus_15min NUMERIC(8,4),
    pre_release_drift NUMERIC(8,4),         -- price change from Monday close to T-5min
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_intraday ON ngas_futures_release_day_intraday(trade_date, report_date);

-- ============================================================
-- Supply/Demand Weekly (Pipeline 5)
-- ============================================================
CREATE TABLE IF NOT EXISTS supply_demand_weekly (
    id BIGSERIAL PRIMARY KEY,
    week_ending_date DATE NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    lng_exports_bcfd NUMERIC(8,3),
    dry_gas_production_bcfd NUMERIC(8,3),
    pipeline_exports_mexico_bcfd NUMERIC(8,3),
    residential_commercial_demand_bcfd NUMERIC(8,3),
    industrial_demand_bcfd NUMERIC(8,3),
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_supply_demand ON supply_demand_weekly(week_ending_date, published_at);

-- ============================================================
-- Model Stability Log (Module 3 drift detection)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_stability_log (
    id BIGSERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(50) NOT NULL,
    region VARCHAR(50),
    lead_days INTEGER,
    residual_bias NUMERIC(10,4),
    bias_std_30d NUMERIC(10,4),
    z_score NUMERIC(8,3),
    consecutive_violations INTEGER NOT NULL DEFAULT 1,
    alert_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    mos_window_adjusted BOOLEAN NOT NULL DEFAULT FALSE,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_stability_model ON model_stability_log(model_name, detected_at);

-- ============================================================
-- Signal Log (Module 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_log (
    id BIGSERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_estimate_bcf NUMERIC(10,3),
    analyst_consensus_bcf NUMERIC(10,3),
    model_vs_consensus_bcf NUMERIC(10,3),   -- model_estimate - analyst_consensus
    pre_release_price_drift NUMERIC(8,4),
    directional_signal SMALLINT,            -- +1 bullish, 0 neutral, -1 bearish
    confidence_score NUMERIC(5,3),          -- 0.0 to 1.0
    actual_bcf NUMERIC(10,3),               -- filled in post-release
    actual_signal SMALLINT,                 -- filled in post-release
    signal_correct BOOLEAN,                 -- filled in post-release
    euro_estimate_bcf NUMERIC(10,3),
    gefs_estimate_bcf NUMERIC(10,3),
    aifs_estimate_bcf NUMERIC(10,3),
    forecast_uncertainty_score NUMERIC(5,3),
    notes TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_signal ON signal_log(report_date);

-- ============================================================
-- Weekly Analysis Master Materialized View
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_analysis_master AS
SELECT
    e.report_date AS week_ending_date,
    e.region,
    e.working_gas_bcf,
    e.net_change_bcf,
    e.five_year_avg_bcf,
    e.year_ago_bcf,
    e.storage_surprise_bcf,
    ac.mean_estimate_bcf AS analyst_consensus_bcf,
    hbc.model_name,
    hbc.seasonal_mask,
    hbc.pop_weighted_hdd_corrected,
    hbc.pop_weighted_cdd_corrected,
    sd.lng_exports_bcfd,
    sd.dry_gas_production_bcfd,
    sd.pipeline_exports_mexico_bcfd,
    sd.residential_commercial_demand_bcfd,
    f.front_month_settle,
    f.twelve_month_strip,
    sl.directional_signal,
    sl.confidence_score,
    sl.pre_release_price_drift
FROM (
    -- Latest published value per report_date + region (no lookahead)
    SELECT DISTINCT ON (report_date, region)
        report_date, region, working_gas_bcf, net_change_bcf,
        five_year_avg_bcf, year_ago_bcf, storage_surprise_bcf
    FROM eia_storage_weekly
    ORDER BY report_date, region, published_at DESC
) e
LEFT JOIN (
    SELECT DISTINCT ON (report_date)
        report_date, mean_estimate_bcf
    FROM analyst_consensus_weekly
    ORDER BY report_date, published_at DESC
) ac ON ac.report_date = e.report_date
LEFT JOIN hdd_cdd_bias_corrected_by_model hbc
    ON hbc.valid_date = e.report_date AND hbc.region = e.region
LEFT JOIN supply_demand_weekly sd
    ON sd.week_ending_date = e.report_date
LEFT JOIN ngas_futures_daily f
    ON f.trade_date = e.report_date
LEFT JOIN signal_log sl
    ON sl.report_date = e.report_date;

CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_analysis ON weekly_analysis_master(week_ending_date, region, model_name);
