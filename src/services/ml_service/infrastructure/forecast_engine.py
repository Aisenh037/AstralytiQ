"""
Time Series Forecasting Engine
Implements Prophet and ARIMA models for sales forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path


class TimeSeriesDataProcessor:
    """Process and validate time series data for forecasting."""
    
    def __init__(self):
        self.data = None
        self.original_data = None
        
    def validate_data(self, df: pd.DataFrame, date_column: str, value_column: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the dataframe has required columns and proper format.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        if date_column not in df.columns:
            return False, f"Date column '{date_column}' not found in data"
        
        if value_column not in df.columns:
            return False, f"Value column '{value_column}' not found in data"
        
        # Check if date column can be converted to datetime
        try:
            pd.to_datetime(df[date_column])
        except Exception as e:
            return False, f"Date column cannot be converted to datetime: {str(e)}"
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            try:
                pd.to_numeric(df[value_column])
            except Exception:
                return False, f"Value column '{value_column}' is not numeric"
        
        # Check minimum data points
        if len(df) < 10:
            return False, "Need at least 10 data points for forecasting"
        
        return True, None
    
    def prepare_data(self, df: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns).
        
        Args:
            df: Input dataframe
            date_column: Name of date column
            value_column: Name of value column
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        # Store original data
        self.original_data = df.copy()
        
        # Create new dataframe with standard column names
        prepared_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_column]),
            'y': pd.to_numeric(df[value_column], errors='coerce')
        })
        
        # Sort by date
        prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)
        
        # Handle missing values in y
        prepared_df = self.handle_missing_data(prepared_df)
        
        self.data = prepared_df
        return prepared_df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using linear interpolation."""
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
        
        # If still have NaNs (shouldn't happen with limit_direction='both'), fill with mean
        if df['y'].isna().any():
            df['y'].fillna(df['y'].mean(), inplace=True)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using z-score method (optional).
        
        Args:
            df: DataFrame with 'y' column
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers marked
        """
        mean = df['y'].mean()
        std = df['y'].std()
        
        df['z_score'] = np.abs((df['y'] - mean) / std)
        df['is_outlier'] = df['z_score'] > threshold
        
        return df
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed data."""
        if self.data is None:
            return {}
        
        return {
            'num_records': len(self.data),
            'date_range_start': self.data['ds'].min().isoformat(),
            'date_range_end': self.data['ds'].max().isoformat(),
            'mean_value': float(self.data['y'].mean()),
            'std_value': float(self.data['y'].std()),
            'min_value': float(self.data['y'].min()),
            'max_value': float(self.data['y'].max()),
            'missing_values': int(self.data['y'].isna().sum())
        }


class ProphetForecaster:
    """Facebook Prophet forecasting model."""
    
    def __init__(self):
        self.model = None
        self.forecast = None
        
    def train(
        self, 
        data: pd.DataFrame, 
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        include_holidays: bool = False,
        country: str = 'US'
    ) -> 'ProphetForecaster':
        """
        Train Prophet model on historical data.
        
        Args:
            data: DataFrame with 'ds' and 'y' columns
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            include_holidays: Include country holidays
            country: Country code for holidays
            
        Returns:
            self for method chaining
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add country holidays if requested
        if include_holidays:
            self.model.add_country_holidays(country_name=country)
        
        # Fit model
        self.model.fit(data)
        
        return self
    
    def predict(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def get_forecast_values(self, future_only: bool = True) -> Dict[str, List]:
        """
        Get forecast values in a clean format.
        
        Args:
            future_only: Return only future predictions (exclude historical fit)
            
        Returns:
            Dictionary with dates, predictions, and confidence intervals
        """
        if self.forecast is None:
            raise ValueError("No predictions available. Call predict() first.")
        
        forecast_df = self.forecast.copy()
        
        if future_only:
            # Only return forecasted values (not fitted historical values)
            forecast_df = forecast_df.tail(len(forecast_df))
        
        return {
            'dates': forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'values': forecast_df['yhat'].tolist(),
            'lower_bound': forecast_df['yhat_lower'].tolist(),
            'upper_bound': forecast_df['yhat_upper'].tolist(),
            'trend': forecast_df['trend'].tolist() if 'trend' in forecast_df.columns else [],
            'weekly': forecast_df['weekly'].tolist() if 'weekly' in forecast_df.columns else [],
            'yearly': forecast_df['yearly'].tolist() if 'yearly' in forecast_df.columns else []
        }
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """Extract forecast components (trend, seasonality)."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return {
            'forecast': self.forecast,
            'components': self.model.plot_components(self.forecast) if self.forecast is not None else None
        }
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame with 'ds' and 'y' columns
            
        Returns:
            Dictionary with error metrics
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Predict on test period
        predictions = self.model.predict(test_data[['ds']])
        
        # Calculate metrics
        y_true = test_data['y'].values
        y_pred = predictions['yhat'].values
        
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else None
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape) if mape is not None else None,
            'r2_score': float(r2)
        }


class ARIMAForecaster:
    """ARIMA/SARIMA forecasting model."""
    
    def __init__(self):
        self.model = None
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
        
    def auto_arima(
        self, 
        data: pd.Series, 
        seasonal: bool = True,
        m: int = 12,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2
    ) -> Tuple[int, int, int]:
        """
        Automatically find best ARIMA parameters.
        
        Args:
            data: Time series data
            seasonal: Use SARIMA (seasonal ARIMA)
            m: Seasonal period (12 for monthly, 7 for weekly, etc.)
            max_p, max_q, max_d: Maximum AR, MA, and differencing orders
            
        Returns:
            Best (p, d, q) order
        """
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("pmdarima not installed. Run: pip install pmdarima")
        
        # Run auto ARIMA
        model = pm.auto_arima(
            data,
            seasonal=seasonal,
            m=m,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            suppress_warnings=True,
            stepwise=True,
            trace=False
        )
        
        self.order = model.order
        self.seasonal_order = model.seasonal_order if seasonal else None
        self.model_fit = model
        
        return self.order
    
    def train(self, data: pd.Series, order: Tuple[int, int, int] = None, seasonal_order: Tuple = None):
        """
        Train ARIMA model.
        
        Args:
            data: Time series data (pd.Series)
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, m)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        if order is None and self.order is None:
            # Use auto_arima if no order specified
            self.auto_arima(data)
            return self
        
        self.order = order or self.order
        self.seasonal_order = seasonal_order or self.seasonal_order
        
        # Fit SARIMA model
        self.model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model_fit = self.model.fit(disp=False)
        return self
    
    def predict(self, periods: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecast.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if self.model_fit is None:
            raise ValueError("Model not trained")
        
        # Get forecast
        forecast = self.model_fit.get_forecast(steps=periods)
        
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)  # 95% confidence interval
        
        return predictions.values, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get model diagnostics (AIC, BIC)."""
        if self.model_fit is None:
            raise ValueError("Model not trained")
        
        return {
            'aic': float(self.model_fit.aic),
            'bic': float(self.model_fit.bic),
            'log_likelihood': float(self.model_fit.llf)
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model_fit is None:
            raise ValueError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_fit': self.model_fit,
                'order': self.order,
                'seasonal_order': self.seasonal_order
            }, f)
    
    def load_model(self, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_fit = data['model_fit']
            self.order = data['order']
            self.seasonal_order = data['seasonal_order']
        return self
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else None
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape) if mape is not None else None
        }
