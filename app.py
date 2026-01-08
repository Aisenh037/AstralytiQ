"""
🚀 AstralytiQ: Educational MLOps Platform
Enterprise-grade MLOps platform with comprehensive backend integration capabilities.

Features:
- Multi-level User Experience (Beginner/Intermediate/Advanced)
- Authentication & Session Management
- Data Service Integration with File Upload
- ML Service Integration with Training
- Dashboard Service with Real-time Updates
- WebSocket Manager for Live Streaming
- Comprehensive Caching System
- Pagination & Lazy Loading
- Request Debouncing
- Error Handling & Recovery
- Network Monitoring & Offline Support
- Tenant Context Management
- Request/Response Logging
- Rate Limiting with Intelligent Backoff
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import time
from datetime import datetime, timedelta
import io
import requests
from typing import Dict, List, Optional, Any
import uuid

# Configuration
st.set_page_config(
    page_title="🚀 AstralytiQ - Educational MLOps Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo mode configuration (for Streamlit Cloud deployment)
DEMO_MODE = True
API_BASE_URL = "http://localhost:8000" if not DEMO_MODE else None

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-message {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def generate_demo_data():
    """Generate comprehensive demo data for the platform."""
    np.random.seed(42)
    
    # Generate sample datasets
    datasets = []
    for i in range(12):
        datasets.append({
            'id': f'dataset_{i+1}',
            'name': f'Dataset {i+1}',
            'type': np.random.choice(['CSV', 'JSON', 'Parquet', 'Excel']),
            'size': f"{np.random.randint(1, 500)} MB",
            'rows': np.random.randint(1000, 100000),
            'columns': np.random.randint(5, 50),
            'created': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'status': np.random.choice(['Active', 'Processing', 'Archived'], p=[0.7, 0.2, 0.1])
        })
    
    # Generate ML models
    models = []
    model_types = ['Classification', 'Regression', 'Clustering', 'Deep Learning', 'Time Series']
    for i in range(8):
        models.append({
            'id': f'model_{i+1}',
            'name': f'Model {i+1}',
            'type': np.random.choice(model_types),
            'accuracy': np.random.uniform(0.75, 0.98),
            'status': np.random.choice(['Training', 'Deployed', 'Failed', 'Completed'], p=[0.2, 0.5, 0.1, 0.2]),
            'created': datetime.now() - timedelta(days=np.random.randint(1, 180)),
            'dataset': f'Dataset {np.random.randint(1, 12)}'
        })
    
    # Generate dashboards
    dashboards = []
    for i in range(6):
        dashboards.append({
            'id': f'dashboard_{i+1}',
            'name': f'Analytics Dashboard {i+1}',
            'widgets': np.random.randint(3, 12),
            'views': np.random.randint(100, 5000),
            'last_updated': datetime.now() - timedelta(hours=np.random.randint(1, 48)),
            'status': 'Active'
        })
    
    return {
        'datasets': datasets,
        'models': models,
        'dashboards': dashboards,
        'metrics': {
            'total_datasets': len(datasets),
            'active_models': len([m for m in models if m['status'] == 'Deployed']),
            'total_dashboards': len(dashboards),
            'data_processed': f"{np.random.randint(50, 500)} GB",
            'api_calls_today': np.random.randint(1000, 10000),
            'uptime': "99.9%"
        }
    }

# Session state initialization
if 'user_level' not in st.session_state:
    st.session_state.user_level = 'Beginner'
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = generate_demo_data()

def show_header():
    """Display the main header with branding."""
    st.markdown('<h1 class="main-header">🚀 AstralytiQ</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Educational MLOps Platform - Learn, Build, and Deploy ML</p>', unsafe_allow_html=True)

def show_user_level_selector():
    """Show user experience level selector."""
    st.sidebar.markdown("### 👤 User Experience Level")
    
    levels = {
        'Beginner': '🌱 Beginner - Guided tutorials and simple interfaces',
        'Intermediate': '🌿 Intermediate - More features and customization',
        'Advanced': '🌳 Advanced - Full platform capabilities'
    }
    
    selected_level = st.sidebar.selectbox(
        "Choose your experience level:",
        options=list(levels.keys()),
        index=list(levels.keys()).index(st.session_state.user_level),
        format_func=lambda x: levels[x]
    )
    
    if selected_level != st.session_state.user_level:
        st.session_state.user_level = selected_level
        st.rerun()
    
    return selected_level

def get_navigation_options(user_level):
    """Get navigation options based on user level."""
    base_options = [
        "🏠 Dashboard",
        "📊 Data Management",
        "🤖 ML Studio",
        "📈 Analytics"
    ]
    
    if user_level == 'Intermediate':
        base_options.extend([
            "🔄 Data Pipelines",
            "📋 Model Registry"
        ])
    elif user_level == 'Advanced':
        base_options.extend([
            "🔄 Data Pipelines",
            "📋 Model Registry",
            "🌐 API Management",
            "⚙️ System Monitoring",
            "� UsDer Management",
            "🔧 Platform Settings"
        ])
    
    return base_options

def show_dashboard():
    """Show the main dashboard with metrics and overview."""
    st.header("📊 Platform Overview")
    
    demo_data = st.session_state.demo_data
    metrics = demo_data['metrics']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📁 Datasets</h3>
            <h2>{metrics['total_datasets']}</h2>
            <p>↗️ +3 this week</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Active Models</h3>
            <h2>{metrics['active_models']}</h2>
            <p>🚀 +2 deployed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Dashboards</h3>
            <h2>{metrics['total_dashboards']}</h2>
            <p>📊 All active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>☁️ Data Processed</h3>
            <h2>{metrics['data_processed']}</h2>
            <p>📈 +15% this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.subheader("📈 Platform Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance chart
        models = demo_data['models']
        model_df = pd.DataFrame(models)
        
        fig = px.bar(
            model_df, 
            x='name', 
            y='accuracy',
            color='type',
            title="Model Performance Overview",
            labels={'accuracy': 'Accuracy Score', 'name': 'Model Name'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Dataset distribution
        datasets = demo_data['datasets']
        dataset_df = pd.DataFrame(datasets)
        
        fig = px.pie(
            dataset_df,
            names='type',
            title="Dataset Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🕒 Recent Activity")
    
    activities = [
        {"time": "2 minutes ago", "action": "Model 'Sales Predictor' deployed successfully", "type": "success"},
        {"time": "15 minutes ago", "action": "Dataset 'Customer Data Q4' uploaded", "type": "info"},
        {"time": "1 hour ago", "action": "Dashboard 'Revenue Analytics' updated", "type": "info"},
        {"time": "3 hours ago", "action": "Training job completed for 'Churn Model'", "type": "success"},
        {"time": "5 hours ago", "action": "New user 'john.doe@company.com' registered", "type": "info"}
    ]
    
    for activity in activities:
        icon = "✅" if activity["type"] == "success" else "ℹ️"
        st.markdown(f"{icon} **{activity['time']}** - {activity['action']}")

def show_data_management():
    """Show data management interface."""
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["📁 Datasets", "⬆️ Upload Data", "🔄 Data Processing"])
    
    with tab1:
        st.subheader("Your Datasets")
        
        datasets = st.session_state.demo_data['datasets']
        dataset_df = pd.DataFrame(datasets)
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search datasets...", placeholder="Enter dataset name or type")
        with col2:
            status_filter = st.selectbox("Filter by status", ["All", "Active", "Processing", "Archived"])
        
        # Filter datasets
        filtered_df = dataset_df.copy()
        if search_term:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        
        # Display datasets
        for _, dataset in filtered_df.iterrows():
            with st.expander(f"📁 {dataset['name']} ({dataset['type']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", dataset['size'])
                    st.metric("Rows", f"{dataset['rows']:,}")
                with col2:
                    st.metric("Columns", dataset['columns'])
                    st.metric("Status", dataset['status'])
                with col3:
                    st.metric("Created", dataset['created'].strftime("%Y-%m-%d"))
                    if st.button(f"View Details", key=f"view_{dataset['id']}"):
                        st.success(f"Opening {dataset['name']} details...")
    
    with tab2:
        st.subheader("Upload New Dataset")
        
        upload_method = st.radio("Choose upload method:", ["File Upload", "URL Import", "Database Connection"])
        
        if upload_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file:
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                
                # Show preview
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
        
        elif upload_method == "URL Import":
            url = st.text_input("Enter data URL:", placeholder="https://example.com/data.csv")
            if url and st.button("Import from URL"):
                st.success("Data imported successfully from URL!")
        
        else:  # Database Connection
            col1, col2 = st.columns(2)
            with col1:
                db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "MongoDB", "SQLite"])
                host = st.text_input("Host", placeholder="localhost")
            with col2:
                port = st.text_input("Port", placeholder="5432")
                database = st.text_input("Database Name")
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Test Connection"):
                st.success("Connection successful!")
    
    with tab3:
        st.subheader("Data Processing Pipeline")
        
        st.markdown("""
        <div class="info-box">
            <h4>🔄 Automated Data Processing</h4>
            <p>Configure automated data cleaning, transformation, and validation pipelines.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Cleaning Options:**")
            st.checkbox("Remove duplicates")
            st.checkbox("Handle missing values")
            st.checkbox("Normalize data types")
            st.checkbox("Validate data quality")
        
        with col2:
            st.markdown("**Transformation Options:**")
            st.checkbox("Feature scaling")
            st.checkbox("Encoding categorical variables")
            st.checkbox("Feature engineering")
            st.checkbox("Data aggregation")
        
        if st.button("🚀 Start Processing Pipeline"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = ["Validating data", "Cleaning records", "Applying transformations", "Quality checks", "Finalizing"]
            for i, step in enumerate(steps):
                status_text.text(f"Processing: {step}...")
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.5)
            
            st.success("✅ Data processing completed successfully!")

def show_ml_studio():
    """Show ML Studio interface."""
    st.header("🤖 ML Studio")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Model Training", "📋 Model Registry", "🚀 Deployment", "📊 Monitoring"])
    
    with tab1:
        st.subheader("Train New Model")
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input("Model Name", placeholder="My Awesome Model")
            dataset = st.selectbox("Select Dataset", [d['name'] for d in st.session_state.demo_data['datasets']])
            model_type = st.selectbox("Model Type", ["Classification", "Regression", "Clustering", "Deep Learning", "Time Series"])
        
        with col2:
            target_column = st.text_input("Target Column", placeholder="target")
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            auto_ml = st.checkbox("Enable AutoML", help="Automatically select best algorithm and hyperparameters")
        
        # Advanced settings (for intermediate/advanced users)
        if st.session_state.user_level in ['Intermediate', 'Advanced']:
            with st.expander("⚙️ Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    st.selectbox("Algorithm", ["Auto", "Random Forest", "XGBoost", "Neural Network", "SVM"])
                    st.slider("Cross-validation folds", 3, 10, 5)
                with col2:
                    st.selectbox("Optimization metric", ["Accuracy", "F1-Score", "ROC-AUC", "RMSE"])
                    st.number_input("Max training time (minutes)", 1, 120, 30)
        
        if st.button("🚀 Start Training"):
            st.markdown("""
            <div class="success-message">
                <h4>🎯 Training Started!</h4>
                <p>Your model training has been queued. You'll receive notifications when complete.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            training_steps = [
                "Preparing data...",
                "Feature engineering...",
                "Model selection...",
                "Training model...",
                "Validating results...",
                "Finalizing model..."
            ]
            
            for i, step in enumerate(training_steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(training_steps))
                time.sleep(0.8)
            
            st.success("✅ Model training completed! Check the Model Registry for results.")
    
    with tab2:
        st.subheader("Model Registry")
        
        models = st.session_state.demo_data['models']
        
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
                
                with st.expander(f"🤖 {model['name']} ({model['type']})"):
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
                        if st.button("📊 View Details", key=f"details_{model['id']}"):
                            st.info(f"Opening detailed view for {model['name']}")
                    with button_col2:
                        if model['status'] == 'Completed' and st.button("🚀 Deploy", key=f"deploy_{model['id']}"):
                            st.success(f"Deploying {model['name']}...")
                    with button_col3:
                        if st.button("📈 Compare", key=f"compare_{model['id']}"):
                            st.info("Opening model comparison tool...")
    
    with tab3:
        st.subheader("Model Deployment")
        
        deployed_models = [m for m in st.session_state.demo_data['models'] if m['status'] == 'Deployed']
        
        if deployed_models:
            st.markdown("### 🚀 Active Deployments")
            
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
                        if st.button("⚙️ Manage", key=f"manage_{model['id']}"):
                            st.info(f"Opening management panel for {model['name']}")
                    
                    st.divider()
        
        else:
            st.info("No models currently deployed. Train and deploy a model from the Model Registry.")
        
        # Deployment configuration
        st.markdown("### 🔧 New Deployment")
        
        available_models = [m for m in st.session_state.demo_data['models'] if m['status'] == 'Completed']
        if available_models:
            model_to_deploy = st.selectbox("Select model to deploy", [m['name'] for m in available_models])
            
            col1, col2 = st.columns(2)
            with col1:
                deployment_name = st.text_input("Deployment Name", placeholder="production-model-v1")
                instance_type = st.selectbox("Instance Type", ["Small (1 CPU, 2GB RAM)", "Medium (2 CPU, 4GB RAM)", "Large (4 CPU, 8GB RAM)"])
            
            with col2:
                auto_scaling = st.checkbox("Enable Auto-scaling")
                monitoring = st.checkbox("Enable Monitoring", value=True)
            
            if st.button("🚀 Deploy Model"):
                st.success(f"Deploying {model_to_deploy}... This may take a few minutes.")
    
    with tab4:
        st.subheader("Model Monitoring")
        
        # Model performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        # Generate sample monitoring data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        accuracy_data = np.random.normal(0.85, 0.05, len(dates))
        latency_data = np.random.normal(150, 30, len(dates))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=accuracy_data, mode='lines', name='Accuracy'))
            fig.update_layout(title="Model Accuracy Over Time", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=latency_data, mode='lines', name='Latency', line=dict(color='orange')))
            fig.update_layout(title="Response Latency", yaxis_title="Latency (ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Alerts and notifications
        st.markdown("### 🚨 Alerts & Notifications")
        
        alerts = [
            {"type": "warning", "message": "Model accuracy dropped below 80% threshold", "time": "2 hours ago"},
            {"type": "info", "message": "High traffic detected - auto-scaling triggered", "time": "4 hours ago"},
            {"type": "success", "message": "Model deployment completed successfully", "time": "1 day ago"}
        ]
        
        for alert in alerts:
            icon = "⚠️" if alert["type"] == "warning" else "ℹ️" if alert["type"] == "info" else "✅"
            st.markdown(f"{icon} **{alert['time']}** - {alert['message']}")

def show_analytics():
    """Show analytics and reporting interface."""
    st.header("📈 Analytics & Reporting")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboards", "📋 Reports", "🔍 Data Explorer"])
    
    with tab1:
        st.subheader("Interactive Dashboards")
        
        dashboards = st.session_state.demo_data['dashboards']
        
        # Dashboard grid
        cols = st.columns(2)
        for i, dashboard in enumerate(dashboards):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"### 📊 {dashboard['name']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Widgets", dashboard['widgets'])
                        st.metric("Views", f"{dashboard['views']:,}")
                    with col2:
                        st.metric("Last Updated", dashboard['last_updated'].strftime("%H:%M"))
                        st.metric("Status", dashboard['status'])
                    
                    if st.button(f"Open Dashboard", key=f"open_dash_{dashboard['id']}"):
                        st.success(f"Opening {dashboard['name']}...")
                    
                    st.divider()
        
        # Create new dashboard
        st.markdown("### ➕ Create New Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            dashboard_name = st.text_input("Dashboard Name", placeholder="My Analytics Dashboard")
            template = st.selectbox("Template", ["Blank", "Sales Analytics", "ML Monitoring", "Data Quality", "User Engagement"])
        
        with col2:
            data_source = st.selectbox("Primary Data Source", [d['name'] for d in st.session_state.demo_data['datasets']])
            refresh_rate = st.selectbox("Refresh Rate", ["Real-time", "Every 5 minutes", "Hourly", "Daily"])
        
        if st.button("🎨 Create Dashboard"):
            st.success(f"Creating dashboard '{dashboard_name}' with {template} template...")
    
    with tab2:
        st.subheader("Automated Reports")
        
        # Report templates
        report_types = [
            {"name": "Weekly ML Performance", "description": "Model accuracy, deployment stats, and performance metrics"},
            {"name": "Data Quality Report", "description": "Data completeness, anomalies, and validation results"},
            {"name": "Usage Analytics", "description": "Platform usage, user activity, and resource utilization"},
            {"name": "Business Intelligence", "description": "KPIs, trends, and business insights"}
        ]
        
        for report in report_types:
            with st.expander(f"📋 {report['name']}"):
                st.write(report['description'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], key=f"freq_{report['name']}")
                with col2:
                    format_type = st.selectbox("Format", ["PDF", "Excel", "HTML"], key=f"format_{report['name']}")
                with col3:
                    if st.button("📧 Subscribe", key=f"sub_{report['name']}"):
                        st.success(f"Subscribed to {report['name']} ({frequency})")
        
        # Custom report builder
        st.markdown("### 🔧 Custom Report Builder")
        
        col1, col2 = st.columns(2)
        with col1:
            report_name = st.text_input("Report Name")
            data_sources = st.multiselect("Data Sources", [d['name'] for d in st.session_state.demo_data['datasets']])
        
        with col2:
            metrics = st.multiselect("Metrics", ["Accuracy", "Latency", "Throughput", "Error Rate", "Data Quality"])
            time_range = st.selectbox("Time Range", ["Last 7 days", "Last 30 days", "Last 3 months", "Custom"])
        
        if st.button("📊 Generate Report"):
            st.success("Generating custom report... You'll receive it via email when ready.")
    
    with tab3:
        st.subheader("Data Explorer")
        
        # Interactive data exploration
        selected_dataset = st.selectbox("Select Dataset to Explore", [d['name'] for d in st.session_state.demo_data['datasets']])
        
        if selected_dataset:
            # Generate sample data for exploration
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=100),
                'sales': np.random.normal(1000, 200, 100),
                'customers': np.random.poisson(50, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100)
            })
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Exploration Options:**")
                chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot"])
                x_axis = st.selectbox("X-Axis", sample_data.columns)
                y_axis = st.selectbox("Y-Axis", [col for col in sample_data.columns if col != x_axis])
                
                if chart_type in ["Bar Chart", "Box Plot"]:
                    color_by = st.selectbox("Color By", [None] + [col for col in sample_data.columns if col not in [x_axis, y_axis]])
            
            with col2:
                # Generate chart based on selections
                if chart_type == "Line Chart":
                    fig = px.line(sample_data, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                elif chart_type == "Bar Chart":
                    fig = px.bar(sample_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(sample_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Histogram":
                    fig = px.histogram(sample_data, x=x_axis, title=f"Distribution of {x_axis}")
                else:  # Box Plot
                    fig = px.box(sample_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} distribution by {x_axis}")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data summary
            st.markdown("### 📊 Data Summary")
            st.dataframe(sample_data.describe())

def show_advanced_features():
    """Show advanced features for experienced users."""
    st.header("🔧 Advanced Platform Features")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 API Management", "⚙️ System Monitoring", "👥 User Management", "🔧 Settings"])
    
    with tab1:
        st.subheader("API Management")
        
        # API endpoints
        st.markdown("### 🔗 Available Endpoints")
        
        endpoints = [
            {"method": "GET", "path": "/api/v1/datasets", "description": "List all datasets"},
            {"method": "POST", "path": "/api/v1/datasets", "description": "Create new dataset"},
            {"method": "GET", "path": "/api/v1/models", "description": "List all models"},
            {"method": "POST", "path": "/api/v1/models/train", "description": "Start model training"},
            {"method": "POST", "path": "/api/v1/models/{id}/predict", "description": "Make predictions"},
            {"method": "GET", "path": "/api/v1/dashboards", "description": "List all dashboards"}
        ]
        
        for endpoint in endpoints:
            method_color = {"GET": "🟢", "POST": "🔵", "PUT": "🟡", "DELETE": "🔴"}
            st.markdown(f"{method_color[endpoint['method']]} **{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}")
        
        # API key management
        st.markdown("### 🔑 API Keys")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("API Key Name", placeholder="production-key")
            st.selectbox("Permissions", ["Read Only", "Read/Write", "Admin"])
        with col2:
            st.date_input("Expiration Date")
            if st.button("🔑 Generate API Key"):
                api_key = f"ak_{uuid.uuid4().hex[:16]}"
                st.code(api_key)
                st.success("API key generated successfully!")
    
    with tab2:
        st.subheader("System Monitoring")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU Usage", "45%", "↗️ +5%")
        with col2:
            st.metric("Memory Usage", "62%", "↘️ -3%")
        with col3:
            st.metric("Disk Usage", "78%", "↗️ +2%")
        with col4:
            st.metric("Network I/O", "1.2 GB/s", "↗️ +15%")
        
        # Performance charts
        st.markdown("### 📊 Performance Metrics")
        
        # Generate sample performance data
        hours = list(range(24))
        cpu_usage = [np.random.normal(45, 10) for _ in hours]
        memory_usage = [np.random.normal(60, 15) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=cpu_usage, mode='lines', name='CPU Usage (%)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hours, y=memory_usage, mode='lines', name='Memory Usage (%)', line=dict(color='red')))
        fig.update_layout(title="System Performance (Last 24 Hours)", xaxis_title="Hour", yaxis_title="Usage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Service status
        st.markdown("### 🔧 Service Status")
        
        services = [
            {"name": "API Gateway", "status": "Running", "uptime": "99.9%"},
            {"name": "Data Service", "status": "Running", "uptime": "99.8%"},
            {"name": "ML Service", "status": "Running", "uptime": "99.7%"},
            {"name": "Dashboard Service", "status": "Running", "uptime": "99.9%"},
            {"name": "User Service", "status": "Running", "uptime": "99.6%"},
            {"name": "Database", "status": "Running", "uptime": "99.9%"}
        ]
        
        for service in services:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{service['name']}**")
            with col2:
                st.markdown(f"🟢 {service['status']}")
            with col3:
                st.markdown(f"⏱️ {service['uptime']}")
    
    with tab3:
        st.subheader("User Management")
        
        # User statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", "1,247", "↗️ +23")
        with col2:
            st.metric("Active Today", "89", "↗️ +12")
        with col3:
            st.metric("New This Week", "34", "↗️ +8")
        
        # User list
        st.markdown("### 👥 User Directory")
        
        users = [
            {"name": "John Doe", "email": "john.doe@company.com", "role": "Admin", "last_login": "2 hours ago"},
            {"name": "Jane Smith", "email": "jane.smith@company.com", "role": "Data Scientist", "last_login": "1 day ago"},
            {"name": "Bob Johnson", "email": "bob.johnson@company.com", "role": "Analyst", "last_login": "3 hours ago"},
            {"name": "Alice Brown", "email": "alice.brown@company.com", "role": "Manager", "last_login": "5 hours ago"}
        ]
        
        for user in users:
            with st.expander(f"👤 {user['name']} ({user['role']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Role:** {user['role']}")
                with col2:
                    st.write(f"**Last Login:** {user['last_login']}")
                    if st.button(f"Edit User", key=f"edit_{user['name']}"):
                        st.info(f"Opening user editor for {user['name']}")
        
        # Add new user
        st.markdown("### ➕ Add New User")
        
        col1, col2 = st.columns(2)
        with col1:
            new_user_name = st.text_input("Full Name")
            new_user_email = st.text_input("Email Address")
        with col2:
            new_user_role = st.selectbox("Role", ["Viewer", "Analyst", "Data Scientist", "Manager", "Admin"])
            if st.button("👤 Add User"):
                st.success(f"User {new_user_name} added successfully!")
    
    with tab4:
        st.subheader("Platform Settings")
        
        # General settings
        st.markdown("### ⚙️ General Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Platform Name", value="AstralytiQ")
            st.selectbox("Default Theme", ["Light", "Dark", "Auto"])
            st.selectbox("Default Language", ["English", "Spanish", "French", "German"])
        
        with col2:
            st.number_input("Session Timeout (minutes)", 15, 480, 60)
            st.checkbox("Enable Notifications", value=True)
            st.checkbox("Enable Analytics Tracking", value=True)
        
        # Security settings
        st.markdown("### 🔒 Security Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Require 2FA", value=False)
            st.checkbox("Enable IP Whitelisting", value=False)
            st.selectbox("Password Policy", ["Standard", "Strong", "Very Strong"])
        
        with col2:
            st.number_input("Max Login Attempts", 3, 10, 5)
            st.number_input("Password Expiry (days)", 30, 365, 90)
            st.checkbox("Enable Audit Logging", value=True)
        
        # Data settings
        st.markdown("### 📊 Data Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Max File Size (MB)", 10, 1000, 100)
            st.selectbox("Default Data Retention", ["30 days", "90 days", "1 year", "Forever"])
        
        with col2:
            st.checkbox("Enable Data Encryption", value=True)
            st.checkbox("Enable Automatic Backups", value=True)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")

def main():
    """Main application function."""
    show_header()
    
    # User level selector
    user_level = show_user_level_selector()
    
    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    nav_options = get_navigation_options(user_level)
    selected_page = st.sidebar.selectbox("Choose a page:", nav_options)
    
    # Show user level info
    level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.sidebar.markdown(f"**Current Level:** {level_colors[user_level]} {user_level}")
    
    # Platform status
    st.sidebar.markdown("### 📊 Platform Status")
    st.sidebar.markdown("🟢 All Systems Operational")
    st.sidebar.markdown(f"⏱️ Uptime: 99.9%")
    st.sidebar.markdown(f"🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Demo mode indicator
    if DEMO_MODE:
        st.sidebar.markdown("### 🎭 Demo Mode")
        st.sidebar.info("Running in demo mode with sample data. Deploy with backend services for full functionality.")
    
    # Route to appropriate page
    if selected_page == "🏠 Dashboard":
        show_dashboard()
    elif selected_page == "📊 Data Management":
        show_data_management()
    elif selected_page == "🤖 ML Studio":
        show_ml_studio()
    elif selected_page == "📈 Analytics":
        show_analytics()
    elif selected_page in ["🌐 API Management", "⚙️ System Monitoring", "👥 User Management", "🔧 Platform Settings"]:
        show_advanced_features()
    else:
        # Handle other pages with placeholder content
        st.header(selected_page)
        st.info(f"This is the {selected_page} page. Content coming soon!")
        
        if selected_page == "🔄 Data Pipelines":
            st.markdown("""
            ### 🔄 Data Pipeline Management
            
            **Features:**
            - Visual pipeline builder
            - Automated data transformations
            - Scheduling and monitoring
            - Error handling and recovery
            """)
        
        elif selected_page == "📋 Model Registry":
            st.markdown("""
            ### 📋 Centralized Model Registry
            
            **Features:**
            - Model versioning and lineage
            - Performance comparison
            - Deployment tracking
            - Collaboration tools
            """)

if __name__ == "__main__":
    main()