# 🏁 Onboarding: Data Scientist

Welcome to the AstralytiQ Data Science team! This guide will help you set up your local modeling environment and understand our MLOps workflow.

## 🛠️ Environment Setup

1.  **Clone the Repo**:
    `git clone https://github.com/Aisenh037/AstralytiQ.git`
2.  **Install DS Stack**:
    `pip install -r requirements.txt` (Includes Prophet, Scikit-learn, Pandas)
3.  **Local Data**:
    Place evaluation datasets in `data/raw/` (Git Ignored).

## 📊 MLOps Workflow

### 1. Data Exploration
Use the **ML Studio** tab in the dashboard to upload datasets and view automated health scores.

### 2. Model Training
-   All core model logic resides in `services/forecasting/`.
-   Use `Prophet` for time-series with strong seasonality.
-   Use `Ensemble` for multi-variate sales forecasting.

### 3. Validation
Models must pass the following baselines before deployment:
-   **MAPE**: < 15%
-   **RMSE**: Within 1-sigma of historical variance.

### 4. Deployment
Once a model is promoted in the UI, the FastAPI backend updates the production weights in the DB.

## 🧪 Testing Models
```bash
pytest tests/test_forecast_engine.py
```
