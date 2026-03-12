"""
NatGas Trading Dashboard — Streamlit

Pages:
  1. Overview       — Current week signal card, storage vs 5yr avg, regime banner
  2. Storage        — Weekly storage chart: actual / 5yr avg / year-ago / model estimate
  3. Weather        — Bias-corrected HDD/CDD by model & region (current forecast week)
  4. Model Monitor  — 90-day weather model accuracy scorecard; drift log
  5. Signal History — Signal log table + win-rate stats; price sensitivity regression

Run: streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

# Ensure natgas package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import text

from natgas.db.connection import get_session

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NatGas Trading Dashboard",
    page_icon=":fire:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def query(sql_str: str, params: dict = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    with get_session() as session:
        result = session.execute(text(sql_str), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    return pd.DataFrame(rows, columns=cols)


def signal_badge(signal: int, confidence: float) -> str:
    if signal == 1:
        return f"🟢 BULLISH ({confidence:.0%})"
    elif signal == -1:
        return f"🔴 BEARISH ({confidence:.0%})"
    return f"⚪ NEUTRAL ({confidence:.0%})"


def regime_color(regime: str) -> str:
    return {"Deficit": "#d62728", "Surplus": "#1f77b4", "Balanced": "#2ca02c"}.get(regime, "#7f7f7f")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(":fire: NatGas Trading")
    page = st.radio(
        "Navigate",
        ["Overview", "Storage", "Weather Models", "Model Monitor", "Signal History"],
        index=0,
    )
    st.divider()
    st.caption("Data refreshes every 5 min. EIA release: Thu 10:30 AM ET.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Natural Gas Trading — Weekly Overview")

    # ── Latest Signal ──
    signal_df = query("""
        SELECT report_date, directional_signal, confidence_score,
               model_estimate_bcf, analyst_consensus_bcf, model_vs_consensus_bcf,
               pre_release_price_drift, actual_bcf, signal_correct
        FROM signal_log
        ORDER BY report_date DESC
        LIMIT 1
    """)

    if signal_df.empty:
        st.info("No signal data yet. Run the analysis pipeline to generate signals.")
    else:
        row = signal_df.iloc[0]
        signal = int(row["directional_signal"]) if pd.notna(row["directional_signal"]) else None
        confidence = float(row["confidence_score"]) if pd.notna(row["confidence_score"]) else 0.0
        report_date = row["report_date"]
        actual = row["actual_bcf"]
        consensus = row["analyst_consensus_bcf"]
        model_est = row["model_estimate_bcf"]
        surprise = row["model_vs_consensus_bcf"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            signal_text = signal_badge(signal, confidence) if signal is not None else "—"
            st.metric("Signal (week ending)", signal_text, delta=None)
            st.caption(f"Report date: {report_date}")

        with c2:
            actual_str = f"{actual:.1f} Bcf" if pd.notna(actual) else "Pending"
            consensus_str = f"{consensus:.1f} Bcf" if pd.notna(consensus) else "—"
            st.metric("Actual Storage Change", actual_str, delta=consensus_str + " consensus")

        with c3:
            model_str = f"{model_est:.1f} Bcf" if pd.notna(model_est) else "—"
            surprise_str = f"{surprise:+.1f}" if pd.notna(surprise) else "—"
            st.metric("Model Estimate", model_str, delta=surprise_str + " vs consensus")

        with c4:
            drift = row["pre_release_price_drift"]
            drift_str = f"${drift:+.4f}/MMBtu" if pd.notna(drift) else "—"
            st.metric("Pre-Release Price Drift", drift_str)

    st.divider()

    # ── Storage vs 5-Year Average ──
    st.subheader("Working Gas Storage vs 5-Year Average")

    storage_df = query("""
        SELECT DISTINCT ON (report_date)
            report_date,
            working_gas_bcf,
            five_year_avg_bcf,
            year_ago_bcf,
            net_change_bcf
        FROM eia_storage_weekly
        WHERE region = 'total'
        ORDER BY report_date, published_at DESC
        LIMIT 104
    """)

    if not storage_df.empty:
        storage_df = storage_df.sort_values("report_date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=storage_df["report_date"], y=storage_df["working_gas_bcf"],
            name="Actual", line=dict(color="#ff7f0e", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=storage_df["report_date"], y=storage_df["five_year_avg_bcf"],
            name="5-Year Avg", line=dict(color="#1f77b4", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=storage_df["report_date"], y=storage_df["year_ago_bcf"],
            name="Year Ago", line=dict(color="#2ca02c", dash="dot"),
        ))
        fig.update_layout(
            xaxis_title="Week Ending", yaxis_title="Bcf",
            legend=dict(orientation="h"), height=380,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No storage data available yet.")

    # ── Regime Banner ──
    st.divider()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Storage Regime")
        if not storage_df.empty:
            latest = storage_df.iloc[-1]
            avg = latest["five_year_avg_bcf"]
            actual_val = latest["working_gas_bcf"]
            if pd.notna(avg) and pd.notna(actual_val):
                pct_vs_avg = (actual_val - avg) / avg * 100
                if pct_vs_avg < -5:
                    regime, regime_note = "Deficit", "(>5% below 5yr avg — bullish bias)"
                elif pct_vs_avg > 5:
                    regime, regime_note = "Surplus", "(>5% above 5yr avg — bearish bias)"
                else:
                    regime, regime_note = "Balanced", "(within ±5% of 5yr avg)"
                color = regime_color(regime)
                st.markdown(
                    f"<h3 style='color:{color}'>{regime} {regime_note}</h3>",
                    unsafe_allow_html=True,
                )
                st.metric("vs 5-Year Average", f"{pct_vs_avg:+.1f}%")

    with col_right:
        st.subheader("Signal Win Rate (All Time)")
        winrate_df = query("""
            SELECT
                COUNT(*) FILTER (WHERE signal_correct = TRUE)  AS correct,
                COUNT(*) FILTER (WHERE signal_correct = FALSE) AS incorrect,
                COUNT(*) FILTER (WHERE directional_signal != 0 AND actual_bcf IS NOT NULL) AS total_directional
            FROM signal_log
            WHERE directional_signal != 0
        """)
        if not winrate_df.empty:
            correct = int(winrate_df.iloc[0]["correct"] or 0)
            total = int(winrate_df.iloc[0]["total_directional"] or 0)
            wr = correct / total if total > 0 else 0
            st.metric("Win Rate", f"{wr:.0%}", delta=f"{correct}/{total} calls")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: STORAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Storage":
    st.title("EIA Storage — Weekly Detail")

    region = st.selectbox(
        "Region",
        ["total", "east", "midwest", "mountain", "pacific", "south_central"],
        index=0,
    )
    lookback_weeks = st.slider("Weeks of history", 26, 260, 104, step=26)

    df = query("""
        SELECT DISTINCT ON (report_date)
            report_date, working_gas_bcf, net_change_bcf,
            five_year_avg_bcf, year_ago_bcf, analyst_consensus_bcf,
            storage_surprise_bcf, revision_number
        FROM eia_storage_weekly
        WHERE region = :region
          AND report_date >= NOW() - (:weeks * INTERVAL '7 days')
        ORDER BY report_date DESC, published_at DESC
    """, {"region": region, "weeks": lookback_weeks})

    if df.empty:
        st.info("No data for selected region.")
    else:
        df = df.sort_values("report_date")

        # Merge model estimates from signal_log for "total" region
        if region == "total":
            sig_df = query("""
                SELECT report_date, model_estimate_bcf
                FROM signal_log
                WHERE model_estimate_bcf IS NOT NULL
            """)
            if not sig_df.empty:
                df = df.merge(sig_df, on="report_date", how="left")
            else:
                df["model_estimate_bcf"] = None

        tab1, tab2 = st.tabs(["Storage Level", "Weekly Change"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["report_date"], y=df["working_gas_bcf"],
                name="Working Gas", line=dict(color="#ff7f0e", width=2.5),
                mode="lines+markers", marker=dict(size=4),
            ))
            if "five_year_avg_bcf" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["report_date"], y=df["five_year_avg_bcf"],
                    name="5-Year Avg", line=dict(color="#1f77b4", dash="dash"),
                ))
            if "year_ago_bcf" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["report_date"], y=df["year_ago_bcf"],
                    name="Year Ago", line=dict(color="#2ca02c", dash="dot"),
                ))
            if region == "total" and "model_estimate_bcf" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["report_date"], y=df["model_estimate_bcf"],
                    name="Model Estimate", line=dict(color="#9467bd", dash="dashdot"),
                    mode="lines+markers", marker=dict(size=5, symbol="diamond"),
                ))
            fig.update_layout(
                xaxis_title="Week Ending", yaxis_title="Bcf",
                legend=dict(orientation="h"), height=420,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            change_df = df.dropna(subset=["net_change_bcf"]).tail(52)
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in change_df["net_change_bcf"]]
            fig2 = go.Figure(go.Bar(
                x=change_df["report_date"],
                y=change_df["net_change_bcf"],
                marker_color=colors,
                name="Net Change (Bcf)",
            ))
            fig2.update_layout(
                xaxis_title="Week Ending", yaxis_title="Bcf",
                height=380, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw data table"):
            st.dataframe(df.sort_values("report_date", ascending=False).reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: WEATHER MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Weather Models":
    st.title("Bias-Corrected HDD/CDD Forecast")

    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox(
            "Region", ["national", "midwest", "northeast", "texas", "southeast"], index=0
        )
    with col2:
        selected_var = st.radio("Variable", ["HDD", "CDD"], horizontal=True)

    var_col = "pop_weighted_hdd_corrected" if selected_var == "HDD" else "pop_weighted_cdd_corrected"
    var_raw = "pop_weighted_hdd_raw" if selected_var == "HDD" else "pop_weighted_cdd_raw"

    df_hdd = query("""
        SELECT forecast_date, valid_date, lead_days, model_name,
               pop_weighted_hdd_corrected, pop_weighted_cdd_corrected,
               pop_weighted_hdd_raw, pop_weighted_cdd_raw,
               bias_hdd, bias_cdd
        FROM hdd_cdd_bias_corrected_by_model
        WHERE region = :region
          AND forecast_date = (
              SELECT MAX(forecast_date)
              FROM hdd_cdd_bias_corrected_by_model
              WHERE region = :region
          )
        ORDER BY valid_date, model_name
    """, {"region": selected_region})

    if df_hdd.empty:
        st.info("No bias-corrected forecast data available yet.")
    else:
        st.caption(f"Forecast date: {df_hdd['forecast_date'].iloc[0]}")

        fig = px.line(
            df_hdd, x="valid_date", y=var_col,
            color="model_name", markers=True,
            title=f"Bias-Corrected {selected_var} — {selected_region.title()}",
            labels={"valid_date": "Valid Date", var_col: f"Pop-Weighted {selected_var}"},
        )
        # Overlay raw (dashed)
        for model in df_hdd["model_name"].unique():
            sub = df_hdd[df_hdd["model_name"] == model]
            fig.add_scatter(
                x=sub["valid_date"], y=sub[var_raw],
                mode="lines", name=f"{model} (raw)",
                line=dict(dash="dot"), opacity=0.45,
            )

        fig.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("7-Day Pop-Weighted Forecast Summary")
        summary = (
            df_hdd[df_hdd["lead_days"] <= 7]
            .groupby("model_name")[[var_col, var_raw]]
            .sum()
            .round(2)
            .rename(columns={var_col: f"Corrected {selected_var} (7d sum)",
                              var_raw: f"Raw {selected_var} (7d sum)"})
        )
        st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Monitor":
    st.title("Weather Model Accuracy Scorecard")

    lookback = st.slider("Lookback days", 30, 90, 90, step=10)

    scorecard_df = query("""
        SELECT
            bc.model_name,
            bc.region,
            bc.lead_days,
            AVG(bc.pop_weighted_hdd_corrected - act.pop_weighted_hdd) AS mean_hdd_residual,
            SQRT(AVG(POWER(bc.pop_weighted_hdd_corrected - act.pop_weighted_hdd, 2))) AS rmse_hdd,
            COUNT(*) AS n_obs
        FROM hdd_cdd_bias_corrected_by_model bc
        JOIN hdd_cdd_daily_by_model act
            ON act.valid_date = bc.valid_date
           AND act.region = bc.region
           AND act.model_name = 'ACTUAL'
           AND act.lead_days = 0
        WHERE bc.valid_date >= NOW() - (:lookback * INTERVAL '1 day')
          AND bc.pop_weighted_hdd_corrected IS NOT NULL
          AND act.pop_weighted_hdd IS NOT NULL
        GROUP BY bc.model_name, bc.region, bc.lead_days
        HAVING COUNT(*) >= 3
        ORDER BY bc.model_name, bc.region, bc.lead_days
    """, {"lookback": lookback})

    if scorecard_df.empty:
        st.info("No accuracy data available yet (need actuals to compare against).")
    else:
        st.subheader(f"HDD RMSE by Model and Lead Day (last {lookback} days)")

        pivot = scorecard_df.pivot_table(
            index=["model_name", "region"],
            columns="lead_days",
            values="rmse_hdd",
        ).round(2)

        st.dataframe(
            pivot.style.background_gradient(axis=None, cmap="YlOrRd"),
            use_container_width=True,
        )

        st.subheader("Residual Bias (positive = model runs warm/high)")
        bias_pivot = scorecard_df.pivot_table(
            index=["model_name", "region"],
            columns="lead_days",
            values="mean_hdd_residual",
        ).round(3)
        st.dataframe(
            bias_pivot.style.background_gradient(axis=None, cmap="coolwarm", vmin=-2, vmax=2),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Model Drift Log (last 30 days)")

    drift_df = query("""
        SELECT detected_at, model_name, region, lead_days,
               residual_bias, z_score, consecutive_violations,
               alert_triggered, mos_window_adjusted, notes
        FROM model_stability_log
        WHERE detected_at >= NOW() - INTERVAL '30 days'
        ORDER BY detected_at DESC
        LIMIT 50
    """)

    if drift_df.empty:
        st.success("No model drift events in the last 30 days.")
    else:
        st.dataframe(drift_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: SIGNAL HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Signal History":
    st.title("Signal Log & Performance")

    sig_df = query("""
        SELECT
            s.report_date,
            s.model_estimate_bcf,
            s.analyst_consensus_bcf,
            s.model_vs_consensus_bcf,
            s.pre_release_price_drift,
            s.directional_signal,
            s.confidence_score,
            s.actual_bcf,
            s.actual_signal,
            s.signal_correct,
            i.price_t_minus_5min,
            i.price_t_plus_15min,
            (i.price_t_plus_15min - i.price_t_minus_5min) AS price_move_t15
        FROM signal_log s
        LEFT JOIN ngas_futures_release_day_intraday i
            ON i.report_date = s.report_date
        ORDER BY s.report_date DESC
        LIMIT 104
    """)

    if sig_df.empty:
        st.info("No signal data yet.")
    else:
        completed = sig_df.dropna(subset=["actual_bcf", "signal_correct"])
        directional = completed[completed["directional_signal"] != 0]

        c1, c2, c3, c4 = st.columns(4)
        total_calls = len(directional)
        correct_calls = int(directional["signal_correct"].sum()) if total_calls > 0 else 0
        win_rate = correct_calls / total_calls if total_calls > 0 else 0

        avg_confidence = float(directional["confidence_score"].mean()) if total_calls > 0 else 0
        avg_surprise = float(completed["model_vs_consensus_bcf"].abs().mean()) if len(completed) > 0 else 0

        with c1:
            st.metric("Total Directional Calls", total_calls)
        with c2:
            st.metric("Win Rate", f"{win_rate:.0%}", delta=f"{correct_calls} correct")
        with c3:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
        with c4:
            st.metric("Avg |Surprise| vs Consensus", f"{avg_surprise:.1f} Bcf")

        st.divider()

        # ── Surprise vs Price Move scatter ──
        st.subheader("Storage Surprise vs 15-Min Price Reaction")
        scatter_df = completed.dropna(subset=["model_vs_consensus_bcf", "price_move_t15"])
        if not scatter_df.empty:
            scatter_df["Signal"] = scatter_df["directional_signal"].map(
                {1: "Bullish", -1: "Bearish", 0: "Neutral"}
            )
            fig = px.scatter(
                scatter_df,
                x="model_vs_consensus_bcf",
                y="price_move_t15",
                color="Signal",
                color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728", "Neutral": "#7f7f7f"},
                labels={
                    "model_vs_consensus_bcf": "Model vs Consensus (Bcf)",
                    "price_move_t15": "Price Move T-5 to T+15 ($/MMBtu)",
                },
                hover_data=["report_date", "confidence_score"],
                trendline="ols",
                trendline_scope="overall",
                height=400,
            )
            fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        # ── Signal table ──
        st.subheader("Full Signal Log")
        display_cols = [
            "report_date", "directional_signal", "confidence_score",
            "model_estimate_bcf", "analyst_consensus_bcf", "model_vs_consensus_bcf",
            "actual_bcf", "signal_correct",
            "price_t_minus_5min", "price_t_plus_15min", "price_move_t15",
        ]
        display_df = sig_df[display_cols].copy()
        display_df["signal_correct"] = display_df["signal_correct"].map(
            {True: "✅", False: "❌", None: "—"}
        )

        st.dataframe(
            display_df.rename(columns={
                "report_date": "Week Ending",
                "directional_signal": "Signal",
                "confidence_score": "Confidence",
                "model_estimate_bcf": "Model (Bcf)",
                "analyst_consensus_bcf": "Consensus (Bcf)",
                "model_vs_consensus_bcf": "M-C (Bcf)",
                "actual_bcf": "Actual (Bcf)",
                "signal_correct": "Correct?",
                "price_t_minus_5min": "T-5 Price",
                "price_t_plus_15min": "T+15 Price",
                "price_move_t15": "ΔPrice",
            }),
            use_container_width=True,
            height=500,
        )
