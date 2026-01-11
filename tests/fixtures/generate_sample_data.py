"""
Generate sample sales data for testing forecasting models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_sales_data(
    start_date: str = "2023-01-01",
    periods: int = 365,
    base_sales: float = 1000,
    trend: float = 2,
    weekly_seasonality: float = 0.3,
    yearly_seasonality: float = 0.2,
    noise_level: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic sales data with trend and seasonality.
    
    Args:
        start_date: Start date for the time series
        periods: Number of days to generate
        base_sales: Base line sales value
        trend: Daily growth rate
        weekly_seasonality: Amplitude of weekly pattern (0-1)
        yearly_seasonality: Amplitude of yearly pattern (0-1)
        noise_level: Random noise level (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with 'date' and 'sales' columns
    """
    np.random.seed(seed)
    
    # Generate date range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate trend
    trend_component = np.arange(periods) * trend
    
    # Generate weekly seasonality (higher on weekends)
    weekly_seasonality_component = base_sales * weekly_seasonality * np.sin(
        2 * np.pi * np.arange(periods) / 7
    )
    
    # Generate yearly seasonality (higher in Q4)
    yearly_seasonality_component = base_sales * yearly_seasonality * np.sin(
        2 * np.pi * np.arange(periods) / 365 - np.pi / 2
    )
    
    # Generate random noise
    noise = np.random.normal(0, base_sales * noise_level, periods)
    
    # Combine all components
    sales = (
        base_sales +
        trend_component +
        weekly_seasonality_component +
        yearly_seasonality_component +
        noise
    )
    
    # Ensure sales are positive
    sales = np.maximum(sales, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales.round(2)
    })
    
    return df


def generate_sample_revenue_data(periods: int = 180) -> pd.DataFrame:
    """Generate sample revenue data with monthly frequency."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2022-01-01", periods=periods, freq='D')
    
    # Exponential growth
    base = 50000
    growth_rate = 0.002
    revenue = base * np.exp(growth_rate * np.arange(periods))
    
    # Add monthly seasonality
    monthly_pattern = 5000 * np.sin(2 * np.pi * np.arange(periods) / 30)
    
    # Add noise
    noise = np.random.normal(0, 2000, periods)
    
    revenue = revenue + monthly_pattern + noise
    revenue = np.maximum(revenue, 0)
    
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue.round(2)
    })
    
    return df


def generate_sample_customers_data(periods: int = 270) -> pd.DataFrame:
    """Generate sample customer count data."""
    np.random.seed(42)
    
    dates = pd.date_range(start="2022-06-01", periods=periods, freq='D')
    
    # Step growth pattern
    base_customers = 100
    step_growth = np.floor(np.arange(periods) / 30) * 10  # Growth every month
    
    # Weekly pattern (fewer customers on weekdays)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_effect = np.where(day_of_week >= 5, 20, -10)  # +20 on weekends, -10 on weekdays
    
    # Random variations
    noise = np.random.normal(0, 5, periods)
    
    customers = base_customers + step_growth + weekly_effect + noise
    customers = np.maximum(customers, 0).round(0)
    
    df = pd.DataFrame({
        'date': dates,
        'customers': customers.astype(int)
    })
    
    return df


if __name__ == "__main__":
    # Generate and save sample datasets
    
    # 1. Daily sales data (1 year)
    sales_df = generate_sample_sales_data(periods=365)
    sales_df.to_csv("tests/fixtures/sample_sales.csv", index=False)
    print(f"✓ Generated sample_sales.csv ({len(sales_df)} rows)")
    print(f"  Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"  Sales range: ${sales_df['sales'].min():.2f} to ${sales_df['sales'].max():.2f}")
    print()
    
    # 2. Daily revenue data (6 months)
    revenue_df = generate_sample_revenue_data(periods=180)
    revenue_df.to_csv("tests/fixtures/sample_revenue.csv", index=False)
    print(f"✓ Generated sample_revenue.csv ({len(revenue_df)} rows)")
    print(f"  Date range: {revenue_df['date'].min()} to {revenue_df['date'].max()}")
    print(f"  Revenue range: ${revenue_df['revenue'].min():.2f} to ${revenue_df['revenue'].max():.2f}")
    print()
    
    # 3. Daily customer count (9 months)
    customers_df = generate_sample_customers_data(periods=270)
    customers_df.to_csv("tests/fixtures/sample_customers.csv", index=False)
    print(f"✓ Generated sample_customers.csv ({len(customers_df)} rows)")
    print(f"  Date range: {customers_df['date'].min()} to {customers_df['date'].max()}")
    print(f"  Customers range: {customers_df['customers'].min()} to {customers_df['customers'].max()}")
    print()
    
    print("All sample datasets generated successfully!")
