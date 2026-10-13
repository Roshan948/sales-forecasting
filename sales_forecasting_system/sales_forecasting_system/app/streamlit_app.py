"""Streamlit UI. All forecasting logic stays inside src/."""
from __future__ import annotations
import io
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import ForecastConfig
from src.ingestion import load_file
from src.column_detection import detect_columns
from src.engine import run_forecast
from src.contracts import ForecastingError

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting System")
st.caption("Upload historical sales, review detected columns, compare models, and export a forecast.")
with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Forecast horizon (periods)", min_value=1, max_value=90, value=12)
    frequency_label = st.selectbox("Frequency", ["Auto detect", "Daily", "Weekly (Sunday end)", "Monthly", "Quarterly"])
    frequency = {"Auto detect": "auto", "Daily": "D", "Weekly (Sunday end)": "W-SUN", "Monthly": "MS", "Quarterly": "QS"}[frequency_label]
    model_label = st.selectbox("Model", ["Automatic selection", "Naive baseline", "ARIMA", "Exponential smoothing"])
    model = {"Automatic selection": "auto", "Naive baseline": "naive", "ARIMA": "arima", "Exponential smoothing": "ets"}[model_label]
    dayfirst = st.selectbox("Ambiguous dates", ["Ask me / default month-first", "Day-first"], help="Use Day-first if dates such as 04/05/2024 mean 4 May.")
    gap_policy_label = st.selectbox("Missing-period policy", ["Fill with zero and warn", "Interpolate", "Leave missing"])
    gap_policy = {"Fill with zero and warn": "zero", "Interpolate": "interpolate", "Leave missing": "missing"}[gap_policy_label]

uploaded = st.file_uploader("Upload sales data", type=["csv", "xlsx", "xls"])
if uploaded:
    try:
        uploaded.seek(0)
        ingestion = load_file(uploaded)
        st.subheader("Data preview")
        st.caption(f"{ingestion.file_info.name} · {ingestion.file_info.format} · {ingestion.file_info.rows:,} rows · {len(ingestion.file_info.columns)} columns")
        st.dataframe(ingestion.frame.head(20), use_container_width=True)
        detection = detect_columns(ingestion.frame)
        st.subheader("Confirm columns")
        date_options = list(ingestion.frame.columns)
        sales_options = list(ingestion.frame.columns)
        date_default = date_options.index(detection.selected_date) if detection.selected_date in date_options else 0
        sales_default = sales_options.index(detection.selected_sales) if detection.selected_sales in sales_options else min(1, len(sales_options)-1)
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date column", date_options, index=date_default)
            st.write("Date candidates", [(c.column, c.confidence) for c in detection.date_candidates[:5]])
        with col2:
            sales_col = st.selectbox("Sales column", sales_options, index=sales_default)
            st.write("Sales candidates", [(c.column, c.confidence) for c in detection.sales_candidates[:5]])
        if detection.needs_confirmation:
            st.warning("Automatic detection is uncertain. Confirm the columns above before running.")
        st.subheader("Run forecast")
        if st.button("Generate forecast", type="primary", use_container_width=True):
            cfg = ForecastConfig(horizon=horizon, frequency=frequency, model=model, gap_policy=gap_policy)
            with st.spinner("Preparing data, validating models, and generating forecast..."):
                try:
                    uploaded.seek(0)
                    result = run_forecast(uploaded, cfg, {"date_col": date_col, "sales_col": sales_col,
                        "dayfirst": dayfirst == "Day-first"})
                    st.session_state["forecast_result"] = result
                except Exception as exc:
                    st.error(str(exc))
        result = st.session_state.get("forecast_result")
        if result and result.status == "success":
            st.divider()
            st.subheader("Forecast results")
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected model", result.selected_model.upper())
            m2.metric("Historical periods", len(result.historical))
            m3.metric("Forecast periods", len(result.forecast))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.historical["date"], y=result.historical["sales"], name="Historical sales", mode="lines"))
            fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["upper"], name="Upper interval", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["lower"], name="Approx. prediction interval",
                fill="tonexty", line=dict(width=0), opacity=.2))
            fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["forecast"], name="Forecast", mode="lines+markers", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="Date", yaxis_title="Sales", hovermode="x unified", legend_orientation="h")
            st.plotly_chart(fig, use_container_width=True)
            st.info(result.rationale)
            st.subheader("Model comparison")
            rows = []
            if result.evaluation:
                for item in result.evaluation.metrics:
                    rows.append({"Model": item.model, "Eligible": item.eligible, "MAE": item.mae, "RMSE": item.rmse,
                                 "MASE": item.mase, "sMAPE (%)": item.smape, "Reason": item.reason})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.subheader("Data quality")
            st.json(result.quality.__dict__ if result.quality else {})
            if result.warnings:
                st.subheader("Warnings and notes")
                for warning in result.warnings:
                    st.warning(f"[{warning.stage}/{warning.code}] {warning.message}")
            else:
                st.success("No warnings reported.")
            st.download_button("Download forecast CSV", result.forecast.to_csv(index=False).encode("utf-8"),
                               file_name="sales_forecast.csv", mime="text/csv")
            if result.evaluation:
                metrics_df = pd.DataFrame(rows)
                st.download_button("Download model metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"),
                                   file_name="model_metrics.csv", mime="text/csv")
        elif result and result.status == "needs_confirmation":
            st.warning(result.rationale)
    except Exception as exc:
        st.error(f"Could not prepare this file: {exc}")
else:
    st.markdown("### How it works\n1. Upload a CSV or Excel file.\n2. Confirm the date and sales columns.\n3. Choose the forecast horizon and options.\n4. Generate, compare, and download forecasts.")
