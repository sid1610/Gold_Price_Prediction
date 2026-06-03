from __future__ import annotations

from datetime import date
from pathlib import Path
import pickle

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_gold_price_forecasting_model.pkl"
SCALER_PATH = BASE_DIR / "gold_price_scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
APP_VERSION = "1.0.0"


FEATURE_GROUPS = {
    "Market Session": ["Open", "High", "Low", "Vol.", "Change %"],
    "Trend Indicators": [
        "MA_30",
        "MA_90",
        "Rolling_STD_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30",
        "Rolling_Mean_90",
        "Rolling_STD_7",
        "EMA_7",
        "EMA_30",
    ],
    "Lag and Momentum": [
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Daily_Range",
        "Momentum_7",
        "Momentum_30",
        "ROC_7",
        "ROC_30",
        "Volatility_30",
    ],
}

PREVIEW_FIELDS = ["Open", "High", "Low", "Vol.", "MA_30", "Lag_1", "Momentum_7", "Volatility_30"]
SMALL_STEP_FIELDS = {"ROC_7", "ROC_30", "Volatility_30"}

SAMPLE_VALUES = {
    "Open": 1845.60,
    "High": 1878.90,
    "Low": 1841.20,
    "Vol.": 276170.00,
    "Change %": 0.49,
    "MA_30": 1872.79,
    "MA_90": 1876.29,
    "Rolling_STD_30": 31.07,
    "Lag_1": 1841.20,
    "Lag_7": 1868.50,
    "Lag_30": 1859.10,
    "Rolling_Mean_7": 1857.90,
    "Rolling_Mean_30": 1872.79,
    "Rolling_Mean_90": 1876.29,
    "Rolling_STD_7": 21.34,
    "EMA_7": 1856.85,
    "EMA_30": 1869.40,
    "Daily_Range": 37.70,
    "Momentum_7": -18.20,
    "Momentum_30": -8.80,
    "ROC_7": -0.009740,
    "ROC_30": -0.004733,
    "Volatility_30": 0.016591,
    "Year": 2021,
    "Month": 1,
    "Day": 29,
    "DayOfWeek": 4,
    "Quarter": 1,
}


st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --surface: #ffffff;
            --ink: #15202b;
            --muted: #667085;
            --line: #d9e0e7;
            --accent: #0f766e;
            --accent-strong: #134e4a;
        }

        .stApp {
            background: #f4f7f8;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #0e1b1f;
        }

        [data-testid="stSidebar"] * {
            color: #e8f0f2;
        }

        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p {
            color: #b7c5cb;
        }

        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        .app-header {
            border-bottom: 1px solid var(--line);
            padding-bottom: 1.15rem;
            margin-bottom: 1.3rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .app-header h1 {
            color: var(--ink);
            font-size: clamp(2.1rem, 3.8vw, 3.4rem);
            font-weight: 760;
            line-height: 1.02;
            letter-spacing: 0;
            margin: 0;
        }

        .app-header p {
            color: var(--muted);
            max-width: 760px;
            font-size: 1rem;
            margin: 0.75rem 0 0;
        }

        .section-title {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 720;
            margin: 0.4rem 0 0.45rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.4rem 0 1rem;
        }

        .quality-item {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem;
        }

        .quality-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .quality-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 760;
            line-height: 1.05;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--accent-strong);
            font-weight: 780;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.55rem 0.95rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            font-weight: 700;
            min-height: 2.55rem;
        }

        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }

        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with FEATURES_PATH.open("rb") as file:
        feature_columns = pickle.load(file)
    return model, scaler, feature_columns


def create_sample_frame(feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{feature: SAMPLE_VALUES.get(feature, 0.0) for feature in feature_columns}])


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def coerce_prediction_input(input_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    numeric_frame = input_frame.copy()
    for column in feature_columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    return numeric_frame.reindex(columns=feature_columns)


def validate_input(input_frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    issues: list[str] = []
    missing_columns = [feature for feature in feature_columns if feature not in input_frame.columns]
    if missing_columns:
        issues.append("Missing required columns: " + ", ".join(missing_columns))
        return issues

    ordered_frame = coerce_prediction_input(input_frame, feature_columns)
    null_columns = ordered_frame.columns[ordered_frame.isna().any()].tolist()
    if null_columns:
        issues.append("Non-numeric or blank values found in: " + ", ".join(null_columns))

    market_columns = [column for column in ["Open", "High", "Low", "Vol."] if column in ordered_frame]
    negative_market_columns = [
        column for column in market_columns if (ordered_frame[column] < 0).any()
    ]
    if negative_market_columns:
        issues.append("Market price and volume fields cannot be negative: " + ", ".join(negative_market_columns))

    if {"High", "Low"}.issubset(ordered_frame.columns):
        if (ordered_frame["High"] < ordered_frame["Low"]).any():
            issues.append("High must be greater than or equal to Low.")

    return issues


def predict_prices(input_frame: pd.DataFrame) -> pd.Series:
    model, scaler, feature_columns = load_artifacts()
    ordered_frame = coerce_prediction_input(input_frame, feature_columns)
    scaled_values = scaler.transform(ordered_frame)
    scaled_frame = pd.DataFrame(
        scaled_values,
        columns=feature_columns,
        index=ordered_frame.index,
    )
    predictions = model.predict(scaled_frame)
    return pd.Series(predictions, index=ordered_frame.index, name="Predicted Price")


def run_prediction(input_frame: pd.DataFrame) -> pd.Series | None:
    try:
        return predict_prices(input_frame)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None


def sync_date_features(selected_date: date) -> None:
    current_date = pd.Timestamp(selected_date)
    st.session_state["Year"] = current_date.year
    st.session_state["Month"] = current_date.month
    st.session_state["Day"] = current_date.day
    st.session_state["DayOfWeek"] = current_date.dayofweek
    st.session_state["Quarter"] = current_date.quarter


def render_quality_strip(feature_columns: list[str]) -> None:
    artifacts = {
        "Model": MODEL_PATH.exists(),
        "Scaler": SCALER_PATH.exists(),
        "Features": FEATURES_PATH.exists(),
        "Inputs": bool(feature_columns),
    }
    html = '<div class="metric-strip">'
    for label, available in artifacts.items():
        value = "Ready" if available else "Missing"
        html += (
            '<div class="quality-item">'
            f'<div class="quality-label">{label}</div>'
            f'<div class="quality-value">{value}</div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_manual_input(feature_columns: list[str]) -> pd.DataFrame:
    sample = create_sample_frame(feature_columns).iloc[0]

    with st.container():
        st.markdown('<div class="section-title">Forecast Inputs</div>', unsafe_allow_html=True)
        selected_date = st.date_input(
            "Trading date",
            value=date(int(sample["Year"]), int(sample["Month"]), int(sample["Day"])),
            help="Date features are calculated automatically from this value.",
        )
        sync_date_features(selected_date)

    for group, fields in FEATURE_GROUPS.items():
        available_fields = [field for field in fields if field in feature_columns]
        if not available_fields:
            continue
        st.markdown(f'<div class="section-title">{group}</div>', unsafe_allow_html=True)
        columns = st.columns(3)
        for index, field in enumerate(available_fields):
            with columns[index % 3]:
                step = 0.0001 if field in SMALL_STEP_FIELDS else 0.01
                min_value = 0.0 if field in {"Open", "High", "Low", "Vol."} else None
                st.number_input(
                    field,
                    value=float(sample[field]),
                    min_value=min_value,
                    step=step,
                    format="%.6f" if step == 0.0001 else "%.2f",
                    key=field,
                )

    manual_row = {}
    for feature in feature_columns:
        manual_row[feature] = st.session_state.get(feature, sample.get(feature, 0.0))
    return pd.DataFrame([manual_row])


def render_prediction_summary(prediction: float, manual_input: pd.DataFrame) -> None:
    open_price = float(manual_input["Open"].iloc[0]) if "Open" in manual_input else 0.0
    daily_range = float(manual_input["Daily_Range"].iloc[0]) if "Daily_Range" in manual_input else 0.0
    change_pct = float(manual_input["Change %"].iloc[0]) if "Change %" in manual_input else 0.0
    delta = prediction - open_price

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Predicted close", format_price(prediction))
    col_b.metric("Move from open", format_price(delta), f"{delta:+,.2f}")
    col_c.metric("Input daily range", f"{daily_range:,.2f}")
    col_d.metric("Input change", f"{change_pct:+.2f}%")


def render_forecast_result(scored: pd.DataFrame) -> None:
    prediction = float(scored["Predicted Price"].iloc[0])
    render_prediction_summary(prediction, scored)
    st.dataframe(
        scored,
        width="stretch",
        hide_index=True,
        column_config={
            "Predicted Price": st.column_config.NumberColumn("Predicted Price", format="$%.2f"),
        },
    )
    st.download_button(
        "Download Forecast Row",
        scored.to_csv(index=False),
        file_name="gold_manual_forecast.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )


def render_batch_quality(batch_input: pd.DataFrame, feature_columns: list[str]) -> None:
    valid_columns = [feature for feature in feature_columns if feature in batch_input.columns]
    missing_columns = [feature for feature in feature_columns if feature not in batch_input.columns]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", f"{len(batch_input):,}")
    col_b.metric("Required columns", f"{len(valid_columns)}/{len(feature_columns)}")
    col_c.metric("Missing", f"{len(missing_columns)}")
    col_d.metric("Extra columns", f"{len([c for c in batch_input.columns if c not in feature_columns])}")


with st.sidebar:
    st.markdown("### Gold Forecasting")
    st.caption("Linear regression model served with the saved scaler and feature order from the final notebook.")
    st.divider()
    st.markdown("**Release**")
    st.caption(f"App version {APP_VERSION}")
    st.divider()
    st.markdown("**Expected feature families**")
    st.caption("OHLC market data, volume, rolling indicators, lag values, momentum, rate of change, volatility, and date parts.")
    st.divider()
    st.markdown("**Deployment assets**")
    st.caption("Keep the model, scaler, and feature list in the same folder as `app.py`.")
    st.caption("Forecasts are model estimates for academic use, not investment advice.")


st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow">Financial Trend Intelligence</div>
        <h1>Gold Price Forecasting</h1>
        <p>Operational Streamlit interface for manual scoring, CSV batch prediction, schema checks, and downloadable forecast outputs using the final trained capstone model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    _, _, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()


render_quality_strip(feature_columns)

tab_manual, tab_batch, tab_schema = st.tabs(["Manual Forecast", "Batch CSV", "Input Schema"])

with tab_manual:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        manual_input = render_manual_input(feature_columns)

    with right:
        st.markdown('<div class="section-title">Live Input Review</div>', unsafe_allow_html=True)
        preview_fields = [field for field in PREVIEW_FIELDS if field in manual_input.columns]
        st.dataframe(
            manual_input[preview_fields],
            width="stretch",
            hide_index=True,
            column_config={
                field: st.column_config.NumberColumn(field, format="%.6f" if field in SMALL_STEP_FIELDS else "%.2f")
                for field in preview_fields
            },
        )

        issues = validate_input(manual_input, feature_columns)
        if issues:
            for issue in issues:
                st.error(issue)
        else:
            st.success("Inputs are complete and ready for prediction.")

        if st.button("Generate Forecast", type="primary", icon=":material/monitoring:", width="stretch", disabled=bool(issues)):
            predictions = run_prediction(manual_input)
            if predictions is not None:
                scored = manual_input.copy()
                scored["Predicted Price"] = predictions
                st.session_state["manual_forecast_result"] = scored

        if "manual_forecast_result" in st.session_state:
            st.markdown('<div class="section-title">Forecast Output</div>', unsafe_allow_html=True)
            render_forecast_result(st.session_state["manual_forecast_result"])

with tab_batch:
    sample_frame = create_sample_frame(feature_columns)
    upload_col, action_col = st.columns([1.2, 1], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader("Upload prediction CSV", type=["csv"])
        st.download_button(
            "Download Sample CSV",
            sample_frame.to_csv(index=False),
            file_name="gold_prediction_template.csv",
            mime="text/csv",
            icon=":material/table:",
            width="stretch",
        )

    if uploaded_file is not None:
        try:
            batch_input = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        with action_col:
            render_batch_quality(batch_input, feature_columns)

        issues = validate_input(batch_input, feature_columns)
        if issues:
            for issue in issues:
                st.error(issue)
        else:
            predictions = run_prediction(batch_input)
            if predictions is not None:
                output = batch_input.copy()
                output["Predicted Price"] = predictions

                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
                st.dataframe(
                    output,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Predicted Price": st.column_config.NumberColumn("Predicted Price", format="$%.2f"),
                    },
                )
                st.download_button(
                    "Download Predictions",
                    output.to_csv(index=False),
                    file_name="gold_price_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    icon=":material/download:",
                    width="stretch",
                )
    else:
        with action_col:
            st.info("Upload a CSV with the required columns or start from the sample template.")

with tab_schema:
    schema = pd.DataFrame(
        {
            "Column": feature_columns,
            "Group": [
                next((group for group, fields in FEATURE_GROUPS.items() if column in fields), "Date Features")
                for column in feature_columns
            ],
            "Type": ["Numeric" for _ in feature_columns],
        }
    )
    st.markdown('<div class="section-title">Required Prediction Schema</div>', unsafe_allow_html=True)
    st.dataframe(schema, width="stretch", hide_index=True)
    st.download_button(
        "Download Schema",
        schema.to_csv(index=False),
        file_name="gold_prediction_schema.csv",
        mime="text/csv",
        icon=":material/download:",
    )
