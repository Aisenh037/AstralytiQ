"""
Unit tests for forecasting engine
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.ml_service.infrastructure.forecast_engine import (
    TimeSeriesDataProcessor,
    ProphetForecaster,
    ARIMAForecaster
)


class TestTimeSeriesDataProcessor:
    """Test TimeSeriesDataProcessor class."""
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'sales': np.random.randn(50) + 100
        })
        
        processor = TimeSeriesDataProcessor()
        is_valid, error = processor.validate_data(df, 'date', 'sales')
        
        assert is_valid is True
        assert error is None
    
    def test_validate_data_missing_column(self):
        """Test validation fails with missing column."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'revenue': np.random.randn(50) + 100
        })
        
        processor = TimeSeriesDataProcessor()
        is_valid, error = processor.validate_data(df, 'date', 'sales')
        
        assert is_valid is False
        assert 'not found' in error.lower()
    
    def test_validate_data_too_few_rows(self):
        """Test validation fails with insufficient data."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'sales': [100, 110, 105, 120, 115]
        })
        
        processor = TimeSeriesDataProcessor()
        is_valid, error = processor.validate_data(df, 'date', 'sales')
        
        assert is_valid is False
        assert 'at least 10' in error.lower()
    
    def test_prepare_data(self):
        """Test data preparation for Prophet format."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'sales': np.random.randn(50) + 100
        })
        
        processor = TimeSeriesDataProcessor()
        prepared = processor.prepare_data(df, 'date', 'sales')
        
        assert 'ds' in prepared.columns
        assert 'y' in prepared.columns
        assert len(prepared) == 50
        assert prepared['ds'].is_monotonic_increasing
    
    def test_handle_missing_data(self):
        """Test missing data handling."""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=50),
            'y': np.random.randn(50) + 100
        })
        
        # Introduce missing values
        df.loc[10:15, 'y'] = np.nan
        
        processor = TimeSeriesDataProcessor()
        processed = processor.handle_missing_data(df)
        
        assert processed['y'].isna().sum() == 0
    
    def test_get_data_stats(self):
        """Test data statistics generation."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.randn(100) + 100
        })
        
        processor = TimeSeriesDataProcessor()
        processor.prepare_data(df, 'date', 'sales')
        stats = processor.get_data_stats()
        
        assert 'num_records' in stats
        assert 'date_range_start' in stats
        assert 'mean_value' in stats
        assert stats['num_records'] == 100


class TestProphetForecaster:
    """Test ProphetForecaster class."""
    
    def test_train_basic(self):
        """Test basic Prophet model training."""
        # Create simple time series
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=100),
            'y': np.sin(np.linspace(0, 4*np.pi, 100)) + 100
        })
        
        forecaster = ProphetForecaster()
        forecaster.train(df)
        
        assert forecaster.model is not None
    
    def test_predict(self):
        """Test generating predictions."""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=100),
            'y': np.sin(np.linspace(0, 4*np.pi, 100)) + 100
        })
        
        forecaster = ProphetForecaster()
        forecaster.train(df)
        forecast = forecaster.predict(periods=30)
        
        assert forecast is not None
        assert len(forecast) > 100  # Should include historical + forecast
    
    def test_get_forecast_values(self):
        """Test extracting forecast values."""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=100),
            'y': np.linspace(100, 200, 100)  # Linear trend
        })
        
        forecaster = ProphetForecaster()
        forecaster.train(df)
        forecaster.predict(periods=30)
        values = forecaster.get_forecast_values(future_only=True)
        
        assert 'dates' in values
        assert 'values' in values
        assert 'lower_bound' in values
        assert 'upper_bound' in values
        assert len(values['dates']) > 0
    
    def test_save_load_model(self, tmp_path):
        """Test model serialization."""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=100),
            'y': np.random.randn(100) + 100
        })
        
        # Train and save
        forecaster1 = ProphetForecaster()
        forecaster1.train(df)
        model_path = tmp_path / "test_model.pkl"
        forecaster1.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load and verify
        forecaster2 = ProphetForecaster()
        forecaster2.load_model(str(model_path))
        
        assert forecaster2.model is not None
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Create predictable data
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=100),
            'y': np.linspace(100, 200, 100)
        })
        
        # Split for training and testing
        train_df = df[:80]
        test_df = df[80:]
        
        forecaster = ProphetForecaster()
        forecaster.train(train_df)
        metrics = forecaster.evaluate(test_df)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2_score' in metrics
        assert metrics['rmse'] >= 0


class TestARIMAForecaster:
    """Test ARIMAForecaster class."""
    
    def test_auto_arima(self):
        """Test automatic ARIMA parameter selection."""
        # Create simple time series
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        
        forecaster = ARIMAForecaster()
        order = forecaster.auto_arima(data, seasonal=False, m=7)
        
        assert order is not None
        assert len(order) == 3  # (p, d, q)
        assert all(isinstance(x, int) for x in order)
    
    def test_train(self):
        """Test ARIMA model training."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        
        forecaster = ARIMAForecaster()
        forecaster.train(data, order=(1, 1, 1))
        
        assert forecaster.model_fit is not None
    
    def test_predict(self):
        """Test generating ARIMA predictions."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        
        forecaster = ARIMAForecaster()
        forecaster.train(data, order=(1, 1, 1))
        predictions, lower, upper = forecaster.predict(periods=10)
        
        asserert len(predictions) == 10
        assert len(lower) == 10
        assert len(upper) == 10
    
    def test_get_diagnostics(self):
        """Test model diagnostics."""
        np.random.seed(42)
        data = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        
        forecaster = ARIMAForecaster()
        forecaster.train(data, order=(1, 1, 1))
        diagnostics = forecaster.get_diagnostics()
        
        assert 'aic' in diagnostics
        assert 'bic' in diagnostics
        assert 'log_likelihood' in diagnostics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
