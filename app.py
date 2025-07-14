# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile, os, io

# --- EDA & Profiling ---
import sweetviz as sv

# --- ML & Forecasting ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Advanced Forecast & BI App", layout="wide")
st.title("Advanced Forecasting & BI Dashboard App")

# --- Sidebar: Data Upload & Cleaning ---
st.sidebar.header("1. Upload & Clean Data")
data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not data_file:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(data_file)
st.sidebar.success("Data loaded!")

# Drop missing if desired
if st.sidebar.checkbox("Drop rows with missing values?", value=True):
    df = df.dropna()

# Show sample
st.subheader("Sample Data")
st.dataframe(df.head())

# --- Manual EDA ---
st.subheader("Manual EDA")
st.write("**Shape:**", df.shape)
st.write("**Dtypes:**"); st.write(df.dtypes)
st.write("**Summary stats:**")
st.write(df.describe(include='all').T)

# Missing
miss = df.isna().sum()
miss_pct = (miss/len(df)*100).round(2)
st.write("**Missing values:**")
st.write(pd.DataFrame({"count":miss, "%":miss_pct}).query("count>0"))

# Categorical counts
cat_cols = df.select_dtypes(["object","category"]).columns
for c in cat_cols:
    st.write(f"**{c}** value counts"); st.write(df[c].value_counts().head(10))

# Numeric distributions & outliers
num_cols = df.select_dtypes(np.number).columns
for c in num_cols:
    fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.box(df, y=c, title=f"Boxplot of {c}")
    st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
if len(num_cols)>1:
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)

# --- Auto EDA ---
if st.sidebar.button("Run Auto EDA (Sweetviz)"):
    report = sv.analyze(df)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    report.show_html(tmp.name)
    html = open(tmp.name,"r",encoding="utf-8").read()
    st.components.v1.html(html, height=700, scrolling=True)
    os.remove(tmp.name)

# --- Cross-sectional ML ---
st.sidebar.header("2. Cross-Sectional ML")
cols = list(df.columns)
target = st.sidebar.selectbox("Target column", cols, index=len(cols)-1)
features = st.sidebar.multiselect("Feature columns", [c for c in cols if c!=target], default=[c for c in cols if c!=target])
if features:
    X = pd.get_dummies(df[features])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model_choice = st.sidebar.radio("Model", ["RandomForest","LinearRegression","XGBoost"])
    if st.sidebar.button("Train Cross-Sectional Model"):
        if model_choice=="RandomForest":
            m = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice=="LinearRegression":
            m = LinearRegression()
        else:
            m = XGBRegressor(n_estimators=100, random_state=42)
        m.fit(X_train,y_train)
        preds = m.predict(X_test)
        st.success(f"{model_choice} MAE: {mean_absolute_error(y_test,preds):.2f} | R2: {r2_score(y_test,preds):.2f}")
        # Download
        out = X_test.copy()
        out["Actual"]=y_test.values; out["Pred"]=preds
        st.download_button("Download Predictions", out.to_csv(index=False).encode(), "preds.csv")

        # Save model
        joblib.dump(m,"cs_model.pkl")
        with open("cs_model.pkl","rb") as f:
            st.download_button("Download Model", f.read(), "cs_model.pkl")

# --- Time Series Forecasting ---
st.sidebar.header("3. Time-Series Forecasting")
ts_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower()]
if ts_cols:
    date_col = st.sidebar.selectbox("Date column", ts_cols)
    df_ts = df.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
    df_ts = df_ts.sort_values(date_col).set_index(date_col)
    ts = df_ts[target]

    # ARIMA
    if st.sidebar.checkbox("ARIMA"):
        periods = st.sidebar.number_input("Periods to forecast", min_value=1, max_value=24, value=6)
        order = st.sidebar.text_input("ARIMA order p,d,q", value="1,1,1")
        if st.sidebar.button("Run ARIMA"):
            p,d,q = map(int,order.split(","))
            ar = ARIMA(ts, order=(p,d,q)).fit()
            fc = ar.forecast(periods)
            st.write("ARIMA forecast:")
            st.write(fc)
            fig = px.line(x=ts.index, y=ts.values, labels={"x":date_col,"y":target}, title="ARIMA: Historical + Forecast")
            idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
            fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
            st.plotly_chart(fig, use_container_width=True)

    # LSTM
    if st.sidebar.checkbox("LSTM"):
        lag = st.sidebar.slider("LSTM lookback", 1, 12, 3)
        epochs = st.sidebar.number_input("Epochs", 1, 200, 50)
        if st.sidebar.button("Run LSTM"):
            arr = ts.values.reshape(-1,1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(arr)
            Xs, ys = [], []
            for i in range(lag, len(scaled)):
                Xs.append(scaled[i-lag:i,0])
                ys.append(scaled[i,0])
            Xs, ys = np.array(Xs), np.array(ys)
            Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
            split = int(0.8*len(Xs))
            X_train, X_test = Xs[:split], Xs[split:]
            y_train, y_test = ys[:split], ys[split:]
            model = Sequential([LSTM(50, input_shape=(lag,1)), Dense(1)])
            model.compile("adam","mse")
            model.fit(X_train,y_train,epochs=epochs,verbose=0)
            # Forecast next periods
            last_seq = scaled[-lag:].reshape(1,lag,1)
            preds = []
            for _ in range(periods):
                next_ = model.predict(last_seq)[0,0]
                preds.append(next_)
                last_seq = np.roll(last_seq, -1)
                last_seq[0,-1,0] = next_
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
            st.write("LSTM forecast:")
            st.write(pd.Series(preds,index=idx))
            fig = px.line(x=ts.index, y=ts.values, labels={"x":date_col,"y":target}, title="LSTM: Historical + Forecast")
            fig.add_scatter(x=idx, y=preds, mode="lines", name="Forecast")
            st.plotly_chart(fig, use_container_width=True)

    # XGBoost TS
    if st.sidebar.checkbox("XGBoost (lag features)"):
        n_lags = st.sidebar.slider("Number of lag features", 1, 12, 3)
        if st.sidebar.button("Run XGB-TS"):
            df_lag = pd.DataFrame({target: ts})
            for l in range(1, n_lags+1):
                df_lag[f"lag_{l}"] = df_lag[target].shift(l)
            df_lag = df_lag.dropna()
            X = df_lag.drop(target,axis=1)
            y = df_lag[target]
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
            xgb = XGBRegressor(n_estimators=100)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            st.success(f"XGB-TS MAE: {mean_absolute_error(y_test,y_pred):.2f}")
            # iterative forecast
            last_vals = df_lag[target].iloc[-n_lags:].values.tolist()
            fc = []
            for _ in range(periods):
                inp = np.array(last_vals[-n_lags:]).reshape(1,-1)
                nxt = xgb.predict(inp)[0]
                fc.append(nxt)
                last_vals.append(nxt)
            idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
            st.write("XGB-TS forecast:")
            st.write(pd.Series(fc,index=idx))
            fig = px.line(x=ts.index, y=ts.values, labels={"x":date_col,"y":target}, title="XGB-TS: Historical + Forecast")
            fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.sidebar.info("No date column detected for time-series.")

# --- Dashboards ---
st.header("4. Dashboards & Analytics")
col1, col2 = st.columns(2)
if features:
    fig1 = px.line(df, x=features[0], y=target, title=f"{target} vs {features[0]}")
    st.plotly_chart(fig1, use_container_width=True)
if len(features)>1:
    fig2 = px.bar(df, x=features[1], y=target, color=features[0], barmode="group",
                  title=f"{target} by {features[1]} & {features[0]}")
    st.plotly_chart(fig2, use_container_width=True)
