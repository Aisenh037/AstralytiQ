# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile, os
import streamlit.components.v1 as components

# --- EDA & Profiling ---
import sweetviz as sv
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

# --- ML & Forecasting ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from statsmodels.tsa.arima.model import ARIMA

# LSTM (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    Sequential = LSTM = Dense = None

# ── CACHING UTILITIES ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def run_sweetviz(df):
    report = sv.analyze(df)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    report.show_html(tmp.name)
    return tmp.name

@st.cache_resource(show_spinner=False)
def train_cs_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# ── APP CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Advanced Forecast & BI App", layout="wide")
st.title("Advanced Forecasting & BI Dashboard App")

# ── DATA UPLOAD & CLEANING ─────────────────────────────────────────────────────
st.sidebar.header("1. Upload & Clean Data")
data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not data_file:
    st.info("Upload a CSV on the left to begin.")
    st.stop()

df = load_data(data_file)
st.sidebar.success("Data loaded!")

if st.sidebar.checkbox("Drop rows with missing values?", value=True):
    df = df.dropna()

# ── TABS LAYOUT ────────────────────────────────────────────────────────────────
tab_eda, tab_ml, tab_ts, tab_dash = st.tabs(
    ["Manual & Auto EDA", "Cross-Sectional ML", "Time-Series Forecasting", "Dashboards"]
)

# ── TAB 1: MANUAL & AUTO EDA ──────────────────────────────────────────────────
with tab_eda:
    st.header("1. Manual EDA")
    # Shape & dtypes
    st.write("**Shape:**", df.shape)
    st.write("**Dtypes:**")
    st.write(df.dtypes)

    # Summary stats
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all").T)

    # Missing values
    st.subheader("Missing Values")
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    st.write(pd.DataFrame({"count": miss, "%": miss_pct}).query("count>0"))

    # Categorical counts
    cat_cols = df.select_dtypes(["object", "category"]).columns
    if len(cat_cols):
        st.subheader("Categorical Value Counts")
        for c in cat_cols:
            st.write(f"**{c}**")
            st.write(df[c].value_counts().head(10))

    # Numeric distributions & outliers
    num_cols = df.select_dtypes(np.number).columns
    if len(num_cols):
        st.subheader("Numeric Distributions & Outliers")
        for c in num_cols:
            fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(df, y=c, title=f"Boxplot of {c}")
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Auto‐EDA via Sweetviz / ProfileReport
    st.header("2. Auto EDA")
    if st.button("▶ Run Sweetviz Report"):
        tmp_html = None
        try:
            tmp_html = run_sweetviz(df)
        except Exception:
            if ProfileReport:
                prof = ProfileReport(df, explorative=True)
                tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                prof.to_file(tmp.name)
                tmp_html = tmp.name
            else:
                st.error("Neither Sweetviz nor ydata_profiling is installed.")

        if tmp_html:
            html = open(tmp_html, "r", encoding="utf-8").read()
            components.html(html, height=700, scrolling=True)
            os.remove(tmp_html)

# ── TAB 2: CROSS-SECTIONAL ML ─────────────────────────────────────────────────
with tab_ml:
    st.header("Cross-Sectional ML")
    cols = df.columns.tolist()
    target = st.selectbox("Select target column", cols, index=len(cols)-1)
    features = st.multiselect("Select feature columns", [c for c in cols if c != target])

    if features:
        # split numeric vs categorical
        num_feats = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
        cat_feats = [c for c in features if c not in num_feats]

        X_num = df[num_feats].to_numpy() if num_feats else np.empty((len(df), 0))
        if cat_feats:
            ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
            X_cat = ohe.fit_transform(df[cat_feats].astype(str))
        else:
            X_cat = sparse.csr_matrix((len(df), 0))

        X = sparse.hstack([X_num, X_cat], format="csr")
        y = df[target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_choice = st.radio("Model", ["LinearRegression", "RandomForest", "XGBoost"])

        if st.button("▶ Train Model"):
            if model_choice == "LinearRegression":
                m = LinearRegression()
            elif model_choice == "RandomForest":
                m = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                m = XGBRegressor(n_estimators=100, random_state=42)

            with st.spinner("Training…"):
                m = train_cs_model(m, X_train, y_train)

            preds = m.predict(X_test)
            st.success(f"MAE: {mean_absolute_error(y_test, preds):.2f} | R²: {r2_score(y_test, preds):.2f}")

            # download preds
            out = pd.DataFrame(X_test.toarray(),
                               columns=[*num_feats, *ohe.get_feature_names_out(cat_feats)])
            out["Actual"] = y_test
            out["Predicted"] = preds
            st.download_button("Download Predictions", out.to_csv(index=False).encode(), "preds.csv")

            # download model
            joblib.dump(m, "cs_model.pkl")
            with open("cs_model.pkl", "rb") as f:
                st.download_button("Download Model", f.read(), "cs_model.pkl")

# ── TAB 3: TIME-SERIES FORECASTING ─────────────────────────────────────────────
with tab_ts:
    st.header("Time-Series Forecasting")
    ts_cols = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "month"])]
    if not ts_cols:
        st.warning("No date/month column found.")
    else:
        date_col = st.selectbox("Select date column", ts_cols)
        periods = st.number_input("Periods to forecast", min_value=1, max_value=24, value=6)
        df_ts = df.copy()
        # robust parse
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], dayfirst=True, errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col, target]).set_index(date_col)
        ts = df_ts[target].sort_index()

        # ARIMA
        if st.checkbox("Enable ARIMA"):
            order = st.text_input("ARIMA order (p,d,q)", "1,1,1")
            if st.button("▶ Run ARIMA"):
                p, d, q = map(int, order.split(","))
                with st.spinner("Fitting ARIMA…"):
                    ar = ARIMA(ts, order=(p, d, q)).fit()
                fc = ar.forecast(periods)
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                fig = px.line(ts, title="ARIMA Forecast")
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # LSTM
        if Sequential and st.checkbox("Enable LSTM"):
            lag = st.slider("LSTM lookback", 1, 12, 3)
            epochs = st.number_input("Epochs", 1, 200, 50)
            if st.button("▶ Run LSTM"):
                arr = ts.values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(arr)
                Xs, ys = [], []
                for i in range(lag, len(scaled)):
                    Xs.append(scaled[i - lag:i, 0])
                    ys.append(scaled[i, 0])
                Xs = np.array(Xs).reshape(-1, lag, 1)
                ys = np.array(ys)
                split = int(0.8 * len(Xs))
                X_tr, X_te = Xs[:split], Xs[split:]
                y_tr, y_te = ys[:split], ys[split:]
                model = Sequential([LSTM(50, input_shape=(lag, 1)), Dense(1)])
                model.compile("adam", "mse")
                with st.spinner("Training LSTM…"):
                    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)
                last_seq = scaled[-lag:].reshape(1, lag, 1)
                fc = []
                for _ in range(periods):
                    nxt = model.predict(last_seq)[0, 0]
                    fc.append(nxt)
                    last_seq = np.roll(last_seq, -1)
                    last_seq[0, -1, 0] = nxt
                fc = scaler.inverse_transform(np.array(fc).reshape(-1, 1)).flatten()
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                fig = px.line(ts, title="LSTM Forecast")
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # XGB-TS
        if XGBRegressor and st.checkbox("Enable XGBoost TS"):
            n_lags = st.slider("XGB lag features", 1, 12, 3)
            if st.button("▶ Run XGB-TS"):
                df_lag = pd.DataFrame({target: ts})
                for i in range(1, n_lags + 1):
                    df_lag[f"lag_{i}"] = df_lag[target].shift(i)
                df_lag.dropna(inplace=True)
                X_all, y_all = df_lag.drop(target, axis=1), df_lag[target]
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)
                xgb = XGBRegressor(n_estimators=100, random_state=42)
                with st.spinner("Training XGB-TS…"):
                    xgb.fit(X_tr, y_tr)
                preds = xgb.predict(X_te)
                st.success(f"MAE: {mean_absolute_error(y_te, preds):.2f}")
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                fig = px.line(ts, title="XGB-TS Forecast")
                fig.add_scatter(x=idx, y=preds[:periods], mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: DASHBOARDS ──────────────────────────────────────────────────────────
with tab_dash:
    st.header("Interactive Dashboards")
    cols = df.columns.tolist()
    target = st.selectbox("Select target", cols, index=len(cols)-1)
    feats = st.multiselect("Select features", [c for c in cols if c != target])
    if feats:
        fig1 = px.line(df, x=feats[0], y=target, title=f"{target} vs {feats[0]}")
        st.plotly_chart(fig1, use_container_width=True)
        if len(feats) > 1:
            fig2 = px.bar(df,
                          x=feats[1],
                          y=target,
                          color=feats[0],
                          barmode="group",
                          title=f"{target} by {feats[1]} & {feats[0]}")
            st.plotly_chart(fig2, use_container_width=True)
