"""
🤖 ML Studio Components - Enhanced with Real Forecasting
Machine learning model training, deployment, and monitoring with Prophet/ARIMA
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend_integration import get_backend_client


def show_ml_studio(demo_data, user_level):
    """Show ML Studio interface with real forecasting capabilities."""
    st.header("ML Studio")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Forecast Training", 
        "📁 Model Registry", 
        "🚀 Deployment", 
        "📈 Monitoring"
    ])
    
    with tab1:
        show_forecast_training_tab()
    
    with tab2:
        show_model_registry_tab(demo_data)
    
    with tab3:
        show_deployment_tab(demo_data)
    
    with tab4:
        show_monitoring_tab()


def show_forecast_training_tab():
    """Enhanced training tab with real forecast capabilities."""
    st.subheader("Train Sales Forecasting Model")
    
    st.markdown("""
    📝 Upload your historical sales data (CSV) and train a Prophet forecasting model.
    Your CSV should have:
    - **Date column**: Any datetime format (e.g., 2023-01-01)
    - **Value column**: Numeric values to forecast (e.g., sales, revenue)
    """)
    
    # File uploader
    st.markdown("### 📤 Step 1: Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload historical sales/revenue data"
    )
    
    # Sample data download
    col_sample, col_space = st.columns([1, 2])
    with col_sample:
        if st.button("📥 Download Sample Data"):
            st.info("Sample data available in tests/fixtures/sample_sales.csv")
    
    if uploaded_file is not None:
        # Load and preview data
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.markdown(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
            
            # Data preview
            with st.expander("📊 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column selection
            st.markdown("### ⚙️ Step 2: Configure Training")
            col1, col2 = st.columns(2)
            
            with col1:
                date_column = st.selectbox(
                    "📅 Date Column",
                    options=df.columns.tolist(),
                    help="Select the column containing dates",
                    key="date_col"
                )
                
                value_column = st.selectbox(
                    "📈 Value Column",
                    options=[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                    help="Select the numeric column to forecast",
                    key="value_col"
                )
            
            with col2:
                forecast_periods = st.slider(
                    "🔮 Forecast Periods (days)",
                    min_value=7,
                    max_value=365,
                    value=30,
                    help="Number of days to forecast into the future"
                )
                
                seasonality_mode = st.radio(
                    "📊 Seasonality Mode",
                    options=["additive", "multiplicative"],
                    help="Additive for trends that don't change much; Multiplicative for exponential growth"
                )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                include_holidays = st.checkbox(
                    "Include US Holidays",
                    value=False,
                    help="Add holiday effects to the model"
                )
            
            # Data validation preview
            try:
                dates = pd.to_datetime(df[date_column])
                values = pd.to_numeric(df[value_column])
                
                st.markdown("### 📋 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Date Range", f"{(dates.max() - dates.min()).days} days")
                with col2:
                    st.metric("Mean Value", f"{values.mean():.2f}")
                with col3:
                    st.metric("Min Value", f"{values.min():.2f}")
                with col4:
                    st.metric("Max Value", f"{values.max():.2f}")
                
                # Chart preview
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#667eea', width=2)
                ))
                fig.update_layout(
                    title="Historical Data Preview",
                    xaxis_title="Date",
                    yaxis_title=value_column,
                    height=300,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Error validating data: {str(e)}")
                st.stop()
            
            # Training button
            st.markdown("### 🚀 Step 3: Start Training")
            
            if st.button("🎯 Train Forecasting Model", type="primary", use_container_width=True):
                train_forecast_model(
                    uploaded_file=uploaded_file,
                    date_column=date_column,
                    value_column=value_column,
                    forecast_periods=forecast_periods,
                    seasonality_mode=seasonality_mode,
                    include_holidays=include_holidays
                )
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        # Show instructions
        st.info("👆 Upload a CSV file to get started")


def train_forecast_model(uploaded_file, date_column, value_column, forecast_periods, seasonality_mode, include_holidays):
    """Train forecasting model via API."""
    
    with st.spinner("🚀 Uploading data..."):
        try:
            client = get_backend_client()
            
            # Get auth token
            token = st.session_state.get('access_token')
            if not token:
                st.warning("⚠️ Please login first (demo mode: using mock training)")
                simulate_training()
                return
            
            # Upload file using centralized client
            files = {'file': uploaded_file}
            response_data = client.upload_data(files)
            
            if "error" in response_data:
                st.error(f"Upload failed: {response_data['error']}")
                return
            
            dataset_id = response_data['dataset_id']
            st.success(f"✅ Data uploaded! Dataset ID: {dataset_id[:8]}...")
            
        except Exception as e:
            st.error(f"❌ Upload error: {str(e)}")
            return
    
    # Start training
    with st.spinner("🤖 Starting model training..."):
        try:
            config = {
                "dataset_id": dataset_id,
                "date_column": date_column,
                "value_column": value_column,
                "model_type": "prophet",
                "forecast_periods": forecast_periods,
                "seasonality_mode": seasonality_mode,
                "include_holidays": include_holidays
            }
            
            response_data = client.train_forecast(config)
            
            if "error" in response_data:
                st.error(f"Training failed: {response_data['error']}")
                return
            
            job_id = response_data['job_id']
            
            # Poll for training status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_wait = 120
            poll_interval = 2
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_data = client.get_job_status(job_id)
                
                if "error" in status_data:
                    st.error(f"Error checking status: {status_data['error']}")
                    break
                
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                progress_bar.progress(progress / 100)
                status_text.text(f"Training: {progress}% - {status}")
                
                if status == "completed":
                    # For completed jobs, job_status has the model_id and metrics
                    model_id = status_data['model_id']
                    metrics = status_data.get('metrics', {})
                    
                    st.success(f"✅ Training completed! Model ID: {model_id[:8]}...")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                    with col2:
                        st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                    with col3:
                        st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    with col4:
                        mape = metrics.get('mape')
                        st.metric("MAPE", f"{mape:.2f}%" if mape else "N/A")
                    
                    # Store model ID for viewing forecast
                    st.session_state['last_model_id'] = model_id
                    
                    st.balloons()
                    
                    # Show forecast button
                    if st.button("📊 View Forecast", type="primary"):
                        show_forecast_visualization(model_id, forecast_periods)
                    
                    break
                
                elif status == "failed":
                    error = status_data.get('error', 'Unknown error')
                    st.error(f"❌ Training failed: {error}")
                    break
                
                time.sleep(poll_interval)
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")


def show_forecast_visualization(model_id, periods):
    """Display forecast visualization."""
    try:
        client = get_backend_client()
        forecast_data = client.get_forecast(model_id)
        
        if "error" not in forecast_data:
            
            dates = forecast_data['forecast_dates']
            values = forecast_data['forecast_values']
            lower = forecast_data['lower_bound']
            upper = forecast_data['upper_bound']
            
            # Create forecast chart
            fig = go.Figure()
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Forecast',
                line=dict(color='#667eea', width=3)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"📈 Sales Forecast - Next {periods} Days",
                xaxis_title="Date",
                yaxis_title="Forecasted Value",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("📊 Forecast Data"):
                forecast_df = pd.DataFrame({
                    'Date': dates,
                    'Forecast': values,
                    'Lower Bound': lower,
                    'Upper Bound': upper
                })
                st.dataframe(forecast_df, use_container_width=True)
        
        else:
            st.error(f"Failed to get forecast: {response.text}")
    
    except Exception as e:
        st.error(f"Error displaying forecast: {str(e)}")


def simulate_training():
    """Simulate training for demo mode."""
    st.info("🎭 Running in simulation mode (backend not connected)")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    training_steps = [
        ("Uploading data...", 20),
        ("Validating data...", 40),
        ("Training Prophet model...", 70),
        ("Evaluating performance...", 90),
        ("Finalizing model...", 100)
    ]
    
    for step, progress in training_steps:
        status_text.text(step)
        progress_bar.progress(progress / 100)
        time.sleep(0.8)
    
    st.success("✅ Model training completed! (Simulated)")
    
    # Show mock metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", "0.8542")
    with col2:
        st.metric("MAE", "12.34")
    with col3:
        st.metric("RMSE", "18.67")
    with col4:
        st.metric("MAPE", "8.3%")


def show_model_registry_tab(demo_data):
    """Show model registry with enhanced filtering."""
    st.subheader("Model Registry")
    
    models = demo_data['models']
    
    # Model filters
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Filter by type", ["All"] + list(set([m['type'] for m in models])))
    with col2:
        status_filter = st.selectbox("Filter by status", ["All", "Training", "Deployed", "Failed", "Completed"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Name", "Accuracy", "Created Date"])
    
    # Display models
    for model in models:
        if (type_filter == "All" or model['type'] == type_filter) and \
           (status_filter == "All" or model['status'] == status_filter):
            
            with st.expander(f"{model['name']} ({model['type']})"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{model['accuracy']:.2%}")
                with col2:
                    st.metric("Status", model['status'])
                with col3:
                    st.metric("Dataset", model['dataset'])
                with col4:
                    st.metric("Created", model['created'].strftime("%Y-%m-%d"))
                
                # Action buttons
                button_col1, button_col2, button_col3 = st.columns(3)
                with button_col1:
                    if st.button("View Details", key=f"details_{model['id']}"):
                        st.info(f"Opening detailed view for {model['name']}")
                with button_col2:
                    if model['status'] == 'Completed' and st.button("Deploy", key=f"deploy_{model['id']}"):
                        st.success(f"Deploying {model['name']}...")
                with button_col3:
                    if st.button("Compare", key=f"compare_{model['id']}"):
                        st.info("Opening model comparison tool...")


def show_deployment_tab(demo_data):
    """Show deployment tab."""
    st.subheader("Model Deployment")
    
    deployed_models = [m for m in demo_data['models'] if m['status'] == 'Deployed']
    
    if deployed_models:
        st.markdown("### Active Deployments")
        
        for model in deployed_models:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{model['name']}**")
                    st.caption(f"{model['type']} • Accuracy: {model['accuracy']:.2%}")
                
                with col2:
                    st.metric("Requests/day", f"{np.random.randint(100, 1000)}")
                
                with col3:
                    st.metric("Avg Response", f"{np.random.randint(50, 200)}ms")
                
                with col4:
                    if st.button("Manage", key=f"manage_{model['id']}"):
                        st.info(f"Opening management panel for {model['name']}")
                
                st.divider()
    
    else:
        st.info("No models currently deployed. Train and deploy a model from the Model Registry.")


def show_monitoring_tab():
    """Show monitoring tab."""
    st.subheader("Model Monitoring")
    st.info("📊 Model performance monitoring and drift detection features coming soon!")
    
    # Placeholder for future monitoring features
    st.markdown("""
    **Planned Features:**
    - 📈 Real-time performance metrics
    - 🎯 Prediction accuracy tracking
    - ⚠️ Model drift detection
    - 📊 A/B testing results
    - 🔔 Alerting and notifications
    """)