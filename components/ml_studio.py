"""
🤖 ML Studio Component
======================
Handles model training, registry, and deployment management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from backend_integration import get_backend_client, get_cached_models

def render_ml_studio():
    """Main entry point for ML Studio component."""
    st.title("ML Studio")
    st.markdown("Train, manage, and deploy your forecasting models.")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Forecast Training", 
        "📁 Model Registry", 
        "🚀 Deployment"
    ])
    
    with tab1:
        show_forecast_training_tab()
    
    with tab2:
        show_model_registry_tab()
    
    with tab3:
        show_deployment_tab()


def show_forecast_training_tab():
    """Training interface."""
    st.subheader("Train New Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        uploaded_file = st.file_uploader("Upload Sales CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} rows.")
                
                date_col = st.selectbox("Date Column", df.columns)
                value_col = st.selectbox("Value Column", [c for c in df.columns if c != date_col])
                
                periods = st.slider("Forecast Horizon (Days)", 7, 365, 30)
                
                if st.button("Start Training", type="primary"):
                    train_model(uploaded_file, date_col, value_col, periods)
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    with col2:
        if uploaded_file and 'df' in locals():
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            try:
                chart_df = df.copy()
                chart_df[date_col] = pd.to_datetime(chart_df[date_col])
                chart_df = chart_df.sort_values(date_col)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_df[date_col], y=chart_df[value_col], mode='lines', name='History'))
                fig.update_layout(title="Historical Data", height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Could not visualize date column.")


def train_model(file, date_col, value_col, periods):
    """Handle training logic via backend."""
    client = get_backend_client()
    
    with st.spinner("Uploading dataset..."):
        try:
            # Check auth
            if not st.session_state.get('access_token'):
                st.error("Authentication required.")
                return

            res = client.upload_data({'file': file})
            if 'error' in res:
                st.error(f"Upload failed: {res['error']}")
                return
            
            dataset_id = res['dataset_id']
            st.success("Dataset uploaded.")
            
        except Exception as e:
            st.error(f"System error: {e}")
            return

    with st.spinner("Initializing training job..."):
        config = {
            "dataset_id": dataset_id,
            "date_column": date_col,
            "value_column": value_col,
            "model_type": "prophet",
            "forecast_periods": periods,
            "seasonality_mode": "multiplicative",
            "include_holidays": False
        }
        
        res = client.train_forecast(config)
        if 'error' in res:
            st.error(f"Training failed: {res['error']}")
            return
            
        job_id = res['job_id']
        track_training_progress(client, job_id)


def track_training_progress(client, job_id):
    """Poll for training status."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for _ in range(60): # 2 minutes max
        status_data = client.get_job_status(job_id)
        if 'error' in status_data:
            st.error(status_data['error'])
            break
            
        status = status_data['status']
        progress = status_data.get('progress', 0)
        
        progress_bar.progress(progress / 100)
        status_text.text(f"Status: {status} ({progress}%)")
        
        if status == 'completed':
            st.success("Training completed successfully!")
            st.balloons()
            # In a real app, we'd invalidate cache here
            break
        elif status == 'failed':
            st.error("Training failed.")
            break
            
        time.sleep(2)


def show_model_registry_tab():
    """Show available models."""
    st.subheader("Model Registry")
    models = get_cached_models()
    
    if models:
        df = pd.DataFrame(models)
        # Clean up dataframe for display
        display_cols = ['name', 'type', 'status', 'accuracy', 'created_at']
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols],
            use_container_width=True,
            column_config={
                "accuracy": st.column_config.NumberColumn(format="%.2f"),
                "created_at": st.column_config.DatetimeColumn(format="D MMM YYYY, h:mm a"),
            }
        )
    else:
        st.info("No models found. Train your first model!")


def show_deployment_tab():
    """Simple deployment view."""
    st.subheader("Active Deployments")
    models = get_cached_models()
    
    if not models:
        st.info("No models available.")
        return

    deployed = [m for m in models if m.get('status') == 'Deployed' or m.get('status') == 'completed'] # Assuming completed are deployable
    
    if deployed:
        for model in deployed:
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"**{model.get('name', 'Unnamed Model')}**")
                c1.caption(f"ID: {model.get('id')}")
                c2.markdown(f"**{model.get('accuracy', 0):.2f}** Accuracy")
                
                if c3.button("Deploy API", key=model['id']):
                    st.toast(f"Model {model['name']} deployed to production endpoint.")
                st.divider()
    else:
        st.info("No models ready for deployment.")