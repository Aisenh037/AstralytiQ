# Examples Directory

This directory contains sample datasets and demonstration files for AstralytiQ.

## 📊 Available Examples

### `sample_sales_data.csv`
A sample sales forecasting dataset with the following features:
- **date**: Transaction date
- **sales**: Daily sales revenue
- **inventory**: Stock levels
- **customer_traffic**: Daily foot traffic
- **marketing_spend**: Marketing expenditure

**Use Case**: Upload this file in the AstralytiQ dashboard to test the forecasting engine.

## 🚀 Quick Start

1. Launch AstralytiQ:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the **Data Upload** section

3. Upload `sample_sales_data.csv`

4. Select forecasting parameters and generate predictions

## 📝 Creating Your Own Dataset

Your CSV should include:
- A date/timestamp column
- Target variable (e.g., sales, revenue)
- Optional: Additional features (inventory, marketing, seasonality indicators)

**Format**: CSV with headers, no missing values in the target column.
