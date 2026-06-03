# Gold Price Forecasting

This repository contains the production Streamlit app, saved model artifacts, and the corrected final notebook for a gold price forecasting capstone project.

## Files

- `final_gold_price_forecasting_model.pkl` - trained forecasting model
- `gold_price_scaler.pkl` - fitted feature scaler
- `feature_columns.pkl` - feature order expected by the model
- `app.py` - production Streamlit app for manual and CSV batch prediction
- `.streamlit/config.toml` - deployed app theme configuration
- `requirements.txt` - Python dependencies for the app
- `capstone-3.ipynb` - corrected final project notebook
- `__notebook_source__.ipynb` - exported notebook source

## Run The App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Notes

Keep `app.py`, the three `.pkl` artifacts, and `requirements.txt` together in the deployed app root. The model artifacts were saved with `scikit-learn==1.6.1`, so the dependency is pinned to that version for consistent loading and prediction behavior.

The app includes manual forecasting, CSV batch scoring, input validation, sample/template downloads, and the required input schema for deployment review.
