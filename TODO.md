# TODO: Enhance Forecasting BI App

## Information Gathered
- App is Streamlit-based with 4 tabs: Manual/Auto EDA, Cross-Sectional ML, Time-Series Forecasting, Interactive Dashboards.
- Uses Pandas, Plotly, Scikit-learn, XGBoost, Statsmodels, Prophet, TensorFlow (LSTM imported but not used).
- Basic EDA: stats, plots, correlations.
- ML: Train Linear, RF, XGB on tabular data.
- TS: ARIMA, Prophet.
- Dashboards: Simple line/bar plots.
- Pre-trained models exist but not loaded.

## Plan
1. **Enhance EDA Tab:** ✅
   - Add clustering (K-Means) with elbow plot. ✅
   - Add PCA for dimensionality reduction and visualization. ✅
   - Add more plot types: scatter plots, pair plots, violin plots. ✅

2. **Enhance ML Tab:** ✅
   - Add SVM and Neural Network models. ✅
   - Implement hyperparameter tuning with GridSearchCV. ✅
   - Add model comparison: bar chart of metrics. ✅
   - Add cross-validation scores. ✅
   - Allow loading pre-trained models. ✅
   - Bonus: Add LightGBM model (optional). ✅

3. **Enhance TS Tab:** ✅
   - Implement LSTM for forecasting. ✅
   - Add SARIMA model. ✅
   - Add anomaly detection using Isolation Forest. ✅
   - Improve Prophet with custom parameters. (Basic implementation)

4. **Enhance Dashboards Tab:** ✅
   - Add filters (date ranges, categories). ✅
   - Add more chart types: pie charts, heatmaps, treemaps. ✅
   - Enable drill-down: click to filter data. (Placeholder)

5. **General Improvements:** ✅
   - Add data preprocessing: scaling options, feature engineering (polynomial features). ✅
   - Improve model saving/loading: save all models, load from files. (Basic)
   - Add export options: CSV and Excel exports for processed data. ✅
   - Better error handling and user feedback. ✅
   - Update requirements.txt for new dependencies. (Not done)

## Dependent Files to Edit
- app.py: Main file to update with new features.
- requirements.txt: Add new libraries if needed (e.g., scikit-learn for SVM, tensorflow for NN/LSTM).

## Followup Steps
- Install new dependencies if any.
- Test each tab for functionality.
- Run the app and verify enhancements.
- Optimize performance for large datasets.
