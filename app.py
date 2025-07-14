import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import sweetviz as sv
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Prophet (for time series forecasting)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="No-Code Sales Forecast & BI", layout="wide")
st.title("No-Code Sales Forecasting & BI Dashboard App")

# --- Upload CSV ---
st.sidebar.header("Upload Sales Data (CSV)")
data_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if data_file is not None:
    df = pd.read_csv(data_file)
    st.subheader("Sample Data")
    st.dataframe(df.head())
else:
    st.info("Upload a CSV to get started.")
    st.stop()

# --- Auto EDA with Sweetviz ---
if st.sidebar.button("Run Auto EDA (Sweetviz)"):
    report = sv.analyze(df)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        report.show_html(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
        os.remove(tmp.name)

# --- Data Cleaning ---
st.sidebar.header("Data Cleaning")
drop_na = st.sidebar.checkbox("Drop rows with missing values?", value=True)
if drop_na:
    df = df.dropna()
    st.write(f"After dropping NA, shape: {df.shape}")

# --- Feature & Target Selection ---
st.sidebar.header("ML Setup")
all_cols = list(df.columns)
target_col = st.sidebar.selectbox("Select Target Variable", all_cols, index=len(all_cols)-1)
feature_cols = st.sidebar.multiselect("Select Features", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col])

if len(feature_cols) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# --- Encode Categoricals ---
df_model = pd.get_dummies(df, columns=[c for c in feature_cols if df[c].dtype=='object'])
X = df_model[[col for col in df_model.columns if col != target_col]]
y = df_model[target_col]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Selection ---
model_choice = st.sidebar.radio("Choose Model", ["RandomForest", "LinearRegression", "XGBoost"])
if model_choice == "RandomForest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "LinearRegression":
    model = LinearRegression()
else:
    model = XGBRegressor(n_estimators=100, random_state=42)

if st.sidebar.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.success(f"Model trained! MAE: {mean_absolute_error(y_test, y_pred):.2f} | R2: {r2_score(y_test, y_pred):.2f}")

    # Download predictions
    pred_df = X_test.copy()
    pred_df['Actual'] = y_test.values
    pred_df['Predicted'] = y_pred
    csv = pred_df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

    # Save model
    joblib.dump(model, "trained_model.pkl")
    with open("trained_model.pkl", "rb") as f:
        st.download_button("Download Model", f.read(), "trained_model.pkl", "application/octet-stream")

# --- Prophet Forecast ---
st.sidebar.markdown("---")
if Prophet and (any("date" in c.lower() for c in df.columns) or any("month" in c.lower() for c in df.columns)):
    time_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower()]
    date_col = st.sidebar.selectbox("Time Column (for Prophet)", time_cols)
    if st.sidebar.button("Run Prophet Forecast"):
        df_prophet = df[[date_col, target_col]].rename(columns={date_col:'ds', target_col:'y'})
        m = Prophet(yearly_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        st.write("Prophet Forecast (tail):", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        fig = px.line(forecast, x='ds', y='yhat', title='Prophet Forecast')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.sidebar.info("To use Prophet, make sure you have a date column and Prophet installed.")

# --- Dashboards ---
st.header("Dashboards & Analytics")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.line(df, x=feature_cols[0], y=target_col, title=f"{target_col} over {feature_cols[0]}")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    if len(feature_cols) > 1:
        fig2 = px.bar(df, x=feature_cols[1], y=target_col, color=feature_cols[0], barmode='group', title=f"{target_col} by {feature_cols[1]}/{feature_cols[0]}")
        st.plotly_chart(fig2, use_container_width=True)
