# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile, os
import streamlit.components.v1 as components

# ── EDA & Profiling ────────────────────────────────────────────────────────────
import sweetviz as sv
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None

# ── ML & FORECASTING ───────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from scipy import sparse
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
from sklearn.ensemble import IsolationForest

# LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    Sequential = LSTM = Dense = None

# LightGBM
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ── HELPERS ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def run_sweetviz(df):
    report = sv.analyze(df)
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    report.show_html(tmp.name)
    html = open(tmp.name, "r", encoding="utf-8").read()
    os.remove(tmp.name)
    return html

# no caching here to avoid hashing errors
def train_cs_model(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    return _model

# ── APP CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Advanced Forecast & BI App", layout="wide")
st.title("Advanced Forecasting & BI Dashboard App")

# ── DATA UPLOAD & CLEANING ─────────────────────────────────────────────────────
st.sidebar.header("1. Upload & Clean Data")
data_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if not data_file:
    st.info("Please upload a CSV to begin.")
    st.stop()

df = load_data(data_file)
st.sidebar.success("Data loaded!")

if st.sidebar.checkbox("Drop rows with missing values?", value=True):
    df = df.dropna()

# ── DATA PREPROCESSING ────────────────────────────────────────────────────────
st.sidebar.header("2. Preprocessing")
if st.sidebar.checkbox("Apply preprocessing"):
    # Scaling
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and st.sidebar.checkbox("Scale numeric columns (MinMax)"):
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Polynomial features
    if num_cols and st.sidebar.checkbox("Add polynomial features (degree 2)"):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[num_cols])
        poly_cols = [f"poly_{i}" for i in range(poly_features.shape[1])]
        df_poly = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
        df = pd.concat([df, df_poly], axis=1)

    # Feature selection (basic: correlation threshold)
    if len(num_cols) > 1 and st.sidebar.checkbox("Feature selection (corr > 0.8)"):
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        df = df.drop(columns=to_drop)
        st.sidebar.info(f"Dropped highly correlated features: {to_drop}")

# Export processed data
st.sidebar.header("3. Export")
st.sidebar.download_button("Download Processed Data (CSV)", df.to_csv(index=False).encode(), "processed_data.csv", mime="text/csv")
st.sidebar.download_button("Download Processed Data (Excel)", df.to_excel("processed_data.xlsx", index=False), "processed_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_eda, tab_ml, tab_ts, tab_dash = st.tabs([
    "Manual & Auto EDA",
    "Cross-Sectional ML",
    "Time-Series Forecasting",
    "Interactive Dashboards"
])

# ── TAB 1: MANUAL & AUTO EDA ──────────────────────────────────────────────────
with tab_eda:
    st.header("Manual EDA")
    st.write("**Shape:**", df.shape)
    st.write("**Dtypes:**")
    st.write(df.dtypes)
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all").T)

    st.subheader("Missing Values")
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    st.write(pd.DataFrame({"count": miss, "%": miss_pct}).query("count > 0"))

    cat_cols = df.select_dtypes(["object","category"]).columns
    if len(cat_cols):
        st.subheader("Categorical Value Counts")
        for c in cat_cols:
            st.write(f"**{c}**")
            st.write(df[c].value_counts().head(10))

    num_cols = df.select_dtypes(np.number).columns
    if len(num_cols):
        st.subheader("Numeric Distributions & Outliers")
        for c in num_cols:
            fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(df, y=c, title=f"Boxplot of {c}")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter Plot
        if len(num_cols) > 1:
            x_scatter = st.selectbox("Select X for scatter", num_cols, key="scatter_x")
            y_scatter = st.selectbox("Select Y for scatter", [c for c in num_cols if c != x_scatter], key="scatter_y")
            fig_scatter = px.scatter(df, x=x_scatter, y=y_scatter, title=f"Scatter: {x_scatter} vs {y_scatter}")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Violin Plot
        violin_col = st.selectbox("Select column for violin plot", num_cols, key="violin")
        fig_violin = px.violin(df, y=violin_col, title=f"Violin Plot of {violin_col}")
        st.plotly_chart(fig_violin, use_container_width=True)

        # Clustering
        if len(num_cols) >= 2:
            st.subheader("Clustering (K-Means)")
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df[num_cols])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df_clust = df.copy()
            df_clust['Cluster'] = clusters
            fig_clust = px.scatter(df_clust, x=num_cols[0], y=num_cols[1], color='Cluster', title="K-Means Clusters")
            st.plotly_chart(fig_clust, use_container_width=True)

            # Elbow Plot
            inertia = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X_scaled)
                inertia.append(km.inertia_)
            fig_elbow = px.line(x=range(1,11), y=inertia, title="Elbow Plot for K-Means")
            st.plotly_chart(fig_elbow, use_container_width=True)

        # PCA
        if len(num_cols) > 2:
            st.subheader("PCA Visualization")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(scaler.fit_transform(df[num_cols]))
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            fig_pca = px.scatter(df_pca, x='PC1', y='PC2', title="PCA 2D Projection")
            st.plotly_chart(fig_pca, use_container_width=True)
            st.write(f"Explained Variance: {pca.explained_variance_ratio_}")

    st.header("Auto EDA")
    if st.button("▶ Run Sweetviz Report"):
        html = None
        with st.spinner("Generating report…"):
            try:
                html = run_sweetviz(df)
            except Exception:
                if ProfileReport:
                    prof = ProfileReport(df, explorative=True)
                    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                    prof.to_file(tmp.name)
                    html = open(tmp.name,"r",encoding="utf-8").read()
                    os.remove(tmp.name)
                else:
                    st.error("Neither Sweetviz nor ydata_profiling is installed.")
        if html:
            components.html(html, height=700, scrolling=True)

# ── TAB 2: CROSS-SECTIONAL ML ─────────────────────────────────────────────────
with tab_ml:
    st.header("Cross-Sectional ML")
    cols = df.columns.tolist()
    target = st.selectbox("Select target column", cols, index=len(cols)-1)
    features = st.multiselect("Select feature columns", [c for c in cols if c != target])

    if features:
        num_feats = [c for c in features if np.issubdtype(df[c].dtype, np.number)]
        cat_feats = [c for c in features if c not in num_feats]

        X_num = df[num_feats].to_numpy() if num_feats else np.empty((len(df),0))
        if cat_feats:
            ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
            X_cat = ohe.fit_transform(df[cat_feats].astype(str))
        else:
            X_cat = sparse.csr_matrix((len(df),0))

        X = sparse.hstack([X_num, X_cat], format="csr")
        y = df[target].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "SVR": SVR(),
            "MLPRegressor": MLPRegressor(random_state=42, max_iter=500)
        }
        if lgb:
            models["LightGBM"] = lgb.LGBMRegressor(random_state=42)

        model_choice = st.radio("Choose model", list(models.keys()))
        tune_hyper = st.checkbox("Enable Hyperparameter Tuning")
        compare_models = st.checkbox("Compare All Models")

        if compare_models:
            results = {}
            for name, model in models.items():
                try:
                    with st.spinner(f"Training {name}…"):
                        model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    results[name] = {"MAE": mae, "R²": r2, "MSE": mse, "CV_R²": cv_scores.mean()}
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
            if results:
                results_df = pd.DataFrame(results).T
                st.write("Model Comparison:")
                st.dataframe(results_df)
                fig_comp = px.bar(results_df, x=results_df.index, y="R²", title="Model R² Comparison")
                st.plotly_chart(fig_comp, use_container_width=True)

        else:
            if tune_hyper:
                param_grids = {
                    "LinearRegression": {},
                    "RandomForest": {"n_estimators": [50, 100, 200]},
                    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]},
                    "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    "MLPRegressor": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]}
                }
                if lgb:
                    param_grids["LightGBM"] = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1]}
                grid = GridSearchCV(models[model_choice], param_grids[model_choice], cv=3, scoring='r2')
                with st.spinner("Tuning hyperparameters…"):
                    grid.fit(X_train, y_train)
                m = grid.best_estimator_
                st.write(f"Best Params: {grid.best_params_}")
            else:
                m = models[model_choice]
                with st.spinner("Training model…"):
                    m = train_cs_model(m, X_train, y_train)

            preds = m.predict(X_test)
            st.success(f"MAE: {mean_absolute_error(y_test,preds):.2f} | R²: {r2_score(y_test,preds):.2f} | MSE: {mean_squared_error(y_test,preds):.2f}")

            out = pd.DataFrame(
                X_test.toarray(),
                columns=[*num_feats, *ohe.get_feature_names_out(cat_feats)]
            )
            out["Actual"] = y_test
            out["Predicted"] = preds
            st.download_button("Download Predictions", out.to_csv(index=False).encode(), "preds.csv")

            joblib.dump(m, "cs_model.pkl")
            with open("cs_model.pkl","rb") as f:
                st.download_button("Download Model", f.read(), "cs_model.pkl")

        # Load pre-trained model
        uploaded_model = st.file_uploader("Upload pre-trained model (.pkl)", type="pkl")
        if uploaded_model:
            loaded_model = joblib.load(uploaded_model)
            st.success("Model loaded!")
            # Allow prediction on new data or something, but for now just info

# ── TAB 3: TIME-SERIES FORECASTING ─────────────────────────────────────────────
with tab_ts:
    st.header("Time-Series Forecasting")
    ts_cols = [c for c in df.columns if any(x in c.lower() for x in ["date","time","month"])]
    if not ts_cols:
        st.warning("No date/time/month column found.")
    else:
        date_col = st.selectbox("Select date column", ts_cols)
        periods  = st.number_input("Forecast periods", min_value=1, max_value=24, value=6)

        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(
            df_ts[date_col],
            dayfirst=True,
            infer_datetime_format=True,
            errors="coerce"
        )
        df_ts = (
            df_ts
            .dropna(subset=[date_col,target])
            .sort_values(date_col)
            .set_index(date_col)
        )
        ts = df_ts[target]
        freq = pd.infer_freq(ts.index) or "M"

        # ARIMA
        if st.checkbox("Enable ARIMA"):
            order = st.text_input("ARIMA order (p,d,q)", "1,1,1")
            if st.button("▶ Run ARIMA"):
                p,d,q = map(int,order.split(","))
                with st.spinner("Fitting ARIMA…"):
                    ar = ARIMA(ts, order=(p,d,q)).fit()
                fc  = ar.forecast(periods)
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq=freq)[1:]
                fig = px.line(ts, title="ARIMA Forecast")
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # Prophet
        if Prophet and st.checkbox("Enable Prophet"):
            if st.button("▶ Run Prophet"):
                df_p = ts.reset_index().rename(columns={date_col:"ds", target:"y"})
                with st.spinner("Fitting Prophet…"):
                    m = Prophet(yearly_seasonality=True)
                    m.fit(df_p)
                future = m.make_future_dataframe(periods=periods, freq=freq)
                fc     = m.predict(future)
                fig    = px.line(fc, x="ds", y="yhat", title="Prophet Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # SARIMA
        if st.checkbox("Enable SARIMA"):
            order = st.text_input("SARIMA order (p,d,q)(P,D,Q,s)", "1,1,1,1,1,1,12", key="sarima_order")
            if st.button("▶ Run SARIMA"):
                p,d,q,P,D,Q,s = map(int,order.split(","))
                with st.spinner("Fitting SARIMA…"):
                    sar = SARIMAX(ts, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
                fc  = sar.forecast(periods)
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq=freq)[1:]
                fig = px.line(ts, title="SARIMA Forecast")
                fig.add_scatter(x=idx, y=fc, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # LSTM
        if Sequential and st.checkbox("Enable LSTM"):
            seq_length = st.slider("Sequence Length", 1, 20, 10, key="seq_len")
            epochs = st.slider("Epochs", 10, 100, 50, key="epochs")
            if st.button("▶ Run LSTM"):
                scaler = MinMaxScaler()
                ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
                X, y = [], []
                for i in range(len(ts_scaled) - seq_length):
                    X.append(ts_scaled[i:i+seq_length])
                    y.append(ts_scaled[i+seq_length])
                X, y = np.array(X), np.array(y)
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                model = Sequential()
                model.add(LSTM(50, input_shape=(seq_length, 1)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                with st.spinner("Training LSTM…"):
                    model.fit(X_train, y_train, epochs=epochs, verbose=0)

                preds_scaled = model.predict(X_test)
                preds = scaler.inverse_transform(preds_scaled).flatten()
                actual = scaler.inverse_transform(y_test).flatten()

                # Forecast future
                last_seq = ts_scaled[-seq_length:]
                future_preds = []
                for _ in range(periods):
                    pred = model.predict(last_seq.reshape(1, seq_length, 1))
                    future_preds.append(pred[0][0])
                    last_seq = np.append(last_seq[1:], pred)
                future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
                idx = pd.date_range(ts.index[-1], periods=periods+1, freq=freq)[1:]

                fig = px.line(ts, title="LSTM Forecast")
                fig.add_scatter(x=ts.index[train_size+seq_length:], y=actual, mode="lines", name="Test Actual")
                fig.add_scatter(x=ts.index[train_size+seq_length:], y=preds, mode="lines", name="Test Predicted")
                fig.add_scatter(x=idx, y=future_preds, mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

        # Anomaly Detection
        if st.checkbox("Enable Anomaly Detection"):
            contamination = st.slider("Contamination (anomaly fraction)", 0.01, 0.1, 0.05, key="contam")
            if st.button("▶ Detect Anomalies"):
                iso = IsolationForest(contamination=contamination, random_state=42)
                anomalies = iso.fit_predict(ts.values.reshape(-1,1))
                df_anom = ts.copy()
                df_anom['Anomaly'] = anomalies == -1
                fig = px.line(df_anom, y=target, title="Anomaly Detection")
                fig.add_scatter(x=df_anom[df_anom['Anomaly']].index, y=df_anom[df_anom['Anomaly']][target], mode="markers", name="Anomalies", marker=dict(color="red", size=10))
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: INTERACTIVE DASHBOARDS ───────────────────────────────────────────────
with tab_dash:
    st.header("Interactive Dashboards")

    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            num_filter = st.selectbox("Filter by numeric column", ["None"] + df.select_dtypes(include=[np.number]).columns.tolist())
            if num_filter != "None":
                min_val, max_val = st.slider(f"Range for {num_filter}", float(df[num_filter].min()), float(df[num_filter].max()), (float(df[num_filter].min()), float(df[num_filter].max())))
                df_filtered = df[(df[num_filter] >= min_val) & (df[num_filter] <= max_val)]
            else:
                df_filtered = df
        else:
            df_filtered = df
    with col2:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_filter = st.selectbox("Filter by categorical column", ["None"] + cat_cols)
            if cat_filter != "None":
                selected_cats = st.multiselect(f"Select {cat_filter} values", df[cat_filter].unique().tolist(), default=df[cat_filter].unique().tolist())
                df_filtered = df_filtered[df_filtered[cat_filter].isin(selected_cats)]
        else:
            df_filtered = df_filtered

    # Date filter if date column exists
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ["date","time","month"])]
    if date_cols:
        date_col = st.selectbox("Filter by date column", ["None"] + date_cols, key="dash_date")
        if date_col != "None":
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
            start_date, end_date = st.date_input("Date range", [df_filtered[date_col].min(), df_filtered[date_col].max()], key="date_range")
            df_filtered = df_filtered[(df_filtered[date_col] >= pd.to_datetime(start_date)) & (df_filtered[date_col] <= pd.to_datetime(end_date))]

    st.write(f"Filtered data shape: {df_filtered.shape}")

    cols = df_filtered.columns.tolist()
    target = st.selectbox("Select target", cols, index=len(cols)-1)
    feats = st.multiselect("Select features", [c for c in cols if c != target])

    if feats:
        chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter", "Pie", "Heatmap", "Treemap"])

        if chart_type == "Line":
            fig = px.line(df_filtered, x=feats[0], y=target, title=f"{target} vs {feats[0]}")
        elif chart_type == "Bar":
            fig = px.bar(df_filtered, x=feats[0], y=target, color=feats[1] if len(feats)>1 else None, title=f"{target} by {feats[0]}")
        elif chart_type == "Scatter":
            fig = px.scatter(df_filtered, x=feats[0], y=target, color=feats[1] if len(feats)>1 else None, title=f"{target} vs {feats[0]}")
        elif chart_type == "Pie":
            if len(feats) > 0:
                fig = px.pie(df_filtered, names=feats[0], values=target, title=f"{target} by {feats[0]}")
            else:
                st.warning("Need at least one feature for Pie chart.")
                fig = None
        elif chart_type == "Heatmap":
            if len(feats) > 1:
                pivot = df_filtered.pivot_table(values=target, index=feats[0], columns=feats[1], aggfunc='mean')
                fig = px.imshow(pivot, text_auto=True, title=f"Heatmap of {target}")
            else:
                st.warning("Need at least two features for Heatmap.")
                fig = None
        elif chart_type == "Treemap":
            if len(feats) > 0:
                fig = px.treemap(df_filtered, path=feats, values=target, title=f"Treemap of {target}")
            else:
                st.warning("Need at least one feature for Treemap.")
                fig = None

        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Drill-down: Simple click to filter (basic implementation)
        if st.button("Drill-down on selected point (not implemented yet)"):
            st.info("Drill-down feature: Click on a point in the chart to filter data. (Placeholder)")
