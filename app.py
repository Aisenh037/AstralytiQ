# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile, os, io
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
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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
st.title("🚀 Advanced Forecasting & BI Dashboard App")

# ── DATA UPLOAD & CLEANING ─────────────────────────────────────────────────────
st.sidebar.header("1. Upload & Clean Data")
data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not data_file:
    st.info("📥 Upload a CSV on the left to begin.")
    st.stop()

df = load_data(data_file)
st.sidebar.success("✅ Data loaded!")

if st.sidebar.checkbox("Drop rows with missing values?", value=True):
    df = df.dropna()

# ── TABS LAYOUT ────────────────────────────────────────────────────────────────
tab_eda, tab_ml, tab_ts, tab_dash = st.tabs(
    ["📊 Manual & Auto EDA", "🤖 Cross-Sectional ML", "⏳ Time-Series Forecasting", "📈 Dashboards"]
)

# ── TAB 1: MANUAL & AUTO EDA ──────────────────────────────────────────────────
with tab_eda:
    st.header("1. Manual EDA")
    st.write("**Shape:**", df.shape)
    st.write("**Dtypes:**"); st.write(df.dtypes)
    st.subheader("Summary Statistics"); st.write(df.describe(include='all').T)

    # Missing values
    miss = df.isna().sum()
    miss_pct = (miss/len(df)*100).round(2)
    st.subheader("Missing Values")
    st.write(pd.DataFrame({"count": miss, "%": miss_pct}).query("count>0"))

    # Categorical value counts
    cat_cols = df.select_dtypes(["object","category"]).columns
    if len(cat_cols):
        st.subheader("Categorical Distributions")
        for c in cat_cols:
            st.write(f"**{c}**"); st.write(df[c].value_counts().head(10))

    # Numeric distributions + outliers
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

    # ── Auto EDA (Sweetviz → fallback to ProfileReport) ─────────────────────────
    st.header("2. Auto EDA")
    if st.button("▶️ Run Auto EDA (Sweetviz)"):
        try:
            tmp_html = run_sweetviz(df)
        except Exception:
            if ProfileReport:
                prof = ProfileReport(df, explorative=True)
                tmp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
                prof.to_file(tmp_html)
            else:
                st.error("Both Sweetviz and ydata_profiling failed or aren’t installed.")
                tmp_html = None

        if tmp_html:
            html = open(tmp_html, "r", encoding="utf-8").read()
            components.html(html, height=700, scrolling=True)
            os.remove(tmp_html)

# ── TAB 2: CROSS-SECTIONAL ML ─────────────────────────────────────────────────
with tab_ml:
    st.header("Cross-Sectional ML")
    cols = list(df.columns)
    target = st.selectbox("Select target column", cols, index=len(cols)-1)
    features = st.multiselect("Select feature columns", [c for c in cols if c != target], default=[c for c in cols if c != target])

    if features:
        # sparse one-hot encoding
        num_feats = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
        cat_feats = [c for c in features if c not in num_feats]

        # numeric part
        X_num = df[num_feats].astype("float32").to_numpy() if num_feats else np.empty((len(df),0))
        # categorical part
        ohe = OneHotEncoder(sparse=True, handle_unknown="ignore", max_categories=30)
        X_cat = ohe.fit_transform(df[cat_feats].astype(str)) if cat_feats else sparse.csr_matrix((len(df),0))
        X = sparse.hstack([X_num, X_cat], format="csr")
        y = df[target]

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.radio("Choose model", ["RandomForest","LinearRegression","XGBoost"])
        if st.button("▶️ Train Cross-Sectional Model"):
            # instantiate
            if model_choice == "RandomForest":
                m = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_choice == "LinearRegression":
                m = LinearRegression()
            else:
                m = XGBRegressor(n_estimators=100, random_state=42)

            with st.spinner(f"Training {model_choice}…"):
                m = train_cs_model(m, X_train, y_train)

            preds = m.predict(X_test)
            st.success(f"{model_choice} → MAE: {mean_absolute_error(y_test,preds):.2f} | R²: {r2_score(y_test,preds):.2f}")

            # download predictions
            out = pd.DataFrame(X_test.toarray(), columns=[*num_feats, *ohe.get_feature_names_out(cat_feats)])
            out["Actual"] = y_test.values
            out["Predicted"] = preds
            st.download_button("Download Predictions", out.to_csv(index=False).encode(), "preds.csv")

            # download model
            joblib.dump(m, "cs_model.pkl")
            with open("cs_model.pkl", "rb") as f:
                st.download_button("Download Model", f.read(), "cs_model.pkl")

# ── TAB 3: TIME-SERIES FORECASTING ─────────────────────────────────────────────
with tab_ts:
    st.header("Time-Series Forecasting")
    ts_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower()]
    if not ts_cols:
        st.warning("No date/month column found.")
    else:
        date_col = st.selectbox("Select date column", ts_cols)
        periods = st.number_input("Periods to forecast", min_value=1, max_value=24, value=6)
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col).set_index(date_col)
        ts = df_ts[target]

        # ARIMA
        if st.checkbox("Enable ARIMA"):
            order = st.text_input("ARIMA order p,d,q", value="1,1,1")
            if st.button("▶️ Run ARIMA"):
                p, d, q = map(int, order.split(","))
                with st.spinner("Fitting ARIMA…"):
                    ar = ARIMA(ts, order=(p, d, q)).fit()
                fc = ar.forecast(periods)
                st.write("**ARIMA Forecast:**", fc)
                fig = px.line(x=ts.index, y=ts.values, labels={"x": date_col, "y": target}, title="ARIMA: History + Forecast")
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # LSTM
        if st.checkbox("Enable LSTM"):
            lag = st.slider("LSTM lookback (months)", 1, 12, 3)
            epochs = st.number_input("Epochs", 1, 200, 50)
            if st.button("▶️ Run LSTM"):
                arr = ts.values.reshape(-1,1)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(arr)
                Xs, ys = [], []
                for i in range(lag, len(scaled)):
                    Xs.append(scaled[i-lag:i,0])
                    ys.append(scaled[i,0])
                Xs, ys = np.array(Xs), np.array(ys)
                Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
                split = int(0.8 * len(Xs))
                X_train, X_test_ = Xs[:split], Xs[split:]
                y_train, y_test_ = ys[:split], ys[split:]
                model = Sequential([LSTM(50, input_shape=(lag,1)), Dense(1)])
                model.compile("adam","mse")
                with st.spinner("Training LSTM…"):
                    model.fit(X_train, y_train, epochs=epochs, verbose=0)
                # forecast
                last_seq = scaled[-lag:].reshape(1,lag,1)
                preds = []
                for _ in range(periods):
                    nxt = model.predict(last_seq)[0,0]
                    preds.append(nxt)
                    last_seq = np.roll(last_seq, -1)
                    last_seq[0,-1,0] = nxt
                preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                st.write("**LSTM Forecast:**", pd.Series(preds, index=idx))
                fig = px.line(x=ts.index, y=ts.values, labels={"x":date_col,"y":target}, title="LSTM: History + Forecast")
                fig.add_scatter(x=idx, y=preds, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # XGB-TS
        if st.checkbox("Enable XGBoost TS"):
            n_lags = st.slider("XGB lag features", 1, 12, 3)
            if st.button("▶️ Run XGB-TS"):
                df_lag = pd.DataFrame({target: ts})
                for l in range(1, n_lags+1):
                    df_lag[f"lag_{l}"] = df_lag[target].shift(l)
                df_lag.dropna(inplace=True)
                X_all, y_all = df_lag.drop(target,axis=1), df_lag[target]
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)
                xgb = XGBRegressor(n_estimators=100, random_state=42)
                with st.spinner("Training XGB-TS…"):
                    xgb.fit(X_tr, y_tr)
                y_pred = xgb.predict(X_te)
                st.success(f"XGB-TS MAE: {mean_absolute_error(y_te,y_pred):.2f}")
                # iterative
                last_vals = df_lag[target].iloc[-n_lags:].tolist()
                fc = []
                for _ in range(periods):
                    inp = np.array(last_vals[-n_lags:]).reshape(1,-1)
                    nxt = xgb.predict(inp)[0]
                    fc.append(nxt)
                    last_vals.append(nxt)
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq="M")[1:]
                st.write("**XGB-TS Forecast:**", pd.Series(fc, index=idx))
                fig = px.line(x=ts.index, y=ts.values, labels={"x":date_col,"y":target}, title="XGB-TS: History + Forecast")
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: DASHBOARDS ──────────────────────────────────────────────────────────
with tab_dash:
    st.header("Interactive Dashboards")
    if features:
        fig1 = px.line(df, x=features[0], y=target, title=f"{target} vs {features[0]}")
        st.plotly_chart(fig1, use_container_width=True)
    if len(features) > 1:
        fig2 = px.bar(
            df,
            x=features[1],
            y=target,
            color=features[0],
            barmode="group",
            title=f"{target} by {features[1]} & {features[0]}"
        )
        st.plotly_chart(fig2, use_container_width=True)
