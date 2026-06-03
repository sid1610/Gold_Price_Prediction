from pathlib import Path
import pickle

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_gold_price_forecasting_model.pkl"
SCALER_PATH = BASE_DIR / "gold_price_scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with FEATURES_PATH.open("rb") as file:
        feature_columns = pickle.load(file)
    return model, scaler, feature_columns


def predict_prices(input_frame: pd.DataFrame) -> pd.Series:
    model, scaler, feature_columns = load_artifacts()
    ordered_frame = input_frame.reindex(columns=feature_columns)
    scaled_values = scaler.transform(ordered_frame)
    scaled_frame = pd.DataFrame(
        scaled_values,
        columns=feature_columns,
        index=ordered_frame.index,
    )
    predictions = model.predict(scaled_frame)
    return pd.Series(predictions, index=ordered_frame.index, name="Predicted Price")


st.set_page_config(
    page_title="Gold Price Prediction",
    layout="wide",
)

st.title("Gold Price Prediction")

try:
    _, _, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

tab_manual, tab_batch = st.tabs(["Manual", "Batch CSV"])

with tab_manual:
    defaults = pd.DataFrame([{feature: 0.0 for feature in feature_columns}])
    manual_input = st.data_editor(
        defaults,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
    )

    if st.button("Predict", type="primary"):
        prediction = predict_prices(manual_input).iloc[0]
        st.metric("Predicted gold price", f"{prediction:,.2f}")

with tab_batch:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_input = pd.read_csv(uploaded_file)
        missing_columns = [
            feature for feature in feature_columns if feature not in batch_input.columns
        ]

        if missing_columns:
            st.error("Missing required columns: " + ", ".join(missing_columns))
        else:
            predictions = predict_prices(batch_input)
            output = batch_input.copy()
            output["Predicted Price"] = predictions
            st.dataframe(output, use_container_width=True)
            st.download_button(
                "Download predictions",
                output.to_csv(index=False),
                file_name="gold_price_predictions.csv",
                mime="text/csv",
            )
