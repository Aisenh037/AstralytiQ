"""
Streamlit Frontend for Enterprise SaaS Platform
Quick prototype for testing and demonstration
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import io

# Configuration
API_BASE_URL = "http://localhost:8000"  # API Gateway URL (for local development)
DATA_SERVICE_URL = "http://localhost:8003"  # Direct to data service for development

# Demo mode for Streamlit Cloud (no backend required)
DEMO_MODE = True

# Page configuration
st.set_page_config(
    page_title="AstralytiQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Enterprise SaaS Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📤 Data Upload", "🔄 Data Transformations", "🔗 Data Lineage", "🤖 ML Training", "📊 Analytics", "⚙️ System Status"]
    )
    
    # Route to different pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Data Upload":
        show_data_upload()
    elif page == "🔄 Data Transformations":
        show_transformations()
    elif page == "🔗 Data Lineage":
        show_lineage()
    elif page == "🤖 ML Training":
        show_ml_training()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ System Status":
        show_system_status()

def show_dashboard():
    """Show main dashboard."""
    st.header("📊 AstralytiQ - No Code Analytics Platform")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Datasets", "12", "↗️ +3")
    
    with col2:
        st.metric("Active Transformations", "8", "↗️ +2")
    
    with col3:
        st.metric("Data Quality Score", "94%", "↗️ +2%")
    
    with col4:
        st.metric("API Requests Today", "1,247", "↗️ +15%")
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Activity")
        activity_data = {
            "Time": ["10:30 AM", "10:15 AM", "09:45 AM", "09:30 AM", "09:15 AM"],
            "Activity": [
                "Dataset 'Sales Q4' uploaded",
                "Transformation pipeline completed",
                "New user registered",
                "Data quality check passed",
                "Lineage graph updated"
            ],
            "Status": ["✅ Success", "✅ Success", "✅ Success", "✅ Success", "✅ Success"]
        }
        st.dataframe(pd.DataFrame(activity_data), width="stretch")
    
    with col2:
        st.subheader("🎯 System Health")
        
        # Create a simple health chart
        services = ["API Gateway", "User Service", "Tenant Service", "Data Service"]
        health_scores = [98, 95, 97, 99]
        
        fig = px.bar(
            x=services,
            y=health_scores,
            title="Service Health Scores",
            color=health_scores,
            color_continuous_scale="Greens"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width="stretch")

def show_data_upload():
    """Show data upload interface."""
    st.header("📤 Data Upload & Processing")
    
    # File upload section
    st.subheader("Upload Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json', 'xml', 'tsv', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, XML, TSV, Parquet"
        )
        
        dataset_name = st.text_input("Dataset Name", placeholder="Enter dataset name")
        dataset_description = st.text_area("Description", placeholder="Enter dataset description")
    
    with col2:
        st.info("📋 **Supported Formats**\n\n• CSV (.csv)\n• Excel (.xlsx, .xls)\n• JSON (.json)\n• XML (.xml)\n• TSV (.tsv)\n• Parquet (.parquet)")
    
    if uploaded_file and dataset_name:
        if st.button("🚀 Upload Dataset", type="primary"):
            with st.spinner("Uploading and processing dataset..."):
                try:
                    # Simulate upload (in real implementation, call API)
                    success = upload_dataset(uploaded_file, dataset_name, dataset_description)
                    
                    if success:
                        st.success("✅ Dataset uploaded successfully!")
                        
                        # Show file preview
                        st.subheader("📋 File Preview")
                        if uploaded_file.type == "text/csv":
                            df = pd.read_csv(uploaded_file)
                            st.dataframe(df.head(10), width="stretch")
                            
                            # Basic statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", len(df))
                            with col2:
                                st.metric("Columns", len(df.columns))
                            with col3:
                                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
                    else:
                        st.error("❌ Upload failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Recent uploads
    st.markdown("---")
    st.subheader("📚 Recent Uploads")
    
    # Mock data for recent uploads
    recent_uploads = {
        "Dataset": ["Sales Data Q4", "Customer Demographics", "Product Inventory", "Marketing Campaigns"],
        "Format": ["CSV", "Excel", "JSON", "CSV"],
        "Size": ["2.3 MB", "1.8 MB", "0.9 MB", "3.1 MB"],
        "Status": ["✅ Processed", "✅ Processed", "🔄 Processing", "✅ Processed"],
        "Uploaded": ["2 hours ago", "1 day ago", "3 days ago", "1 week ago"]
    }
    
    st.dataframe(pd.DataFrame(recent_uploads), width="stretch")

def show_transformations():
    """Show data transformation interface."""
    st.header("🔄 Data Transformations")
    
    # Dataset selection
    st.subheader("Select Dataset")
    datasets = ["Sales Data Q4", "Customer Demographics", "Product Inventory"]
    selected_dataset = st.selectbox("Choose dataset to transform", datasets)
    
    if selected_dataset:
        # Transformation pipeline builder
        st.subheader("🛠️ Build Transformation Pipeline")
        
        transformations = []
        
        # Add transformation steps
        with st.expander("➕ Add Transformation Steps", expanded=True):
            transformation_type = st.selectbox(
                "Transformation Type",
                [
                    "Remove Duplicates",
                    "Fill Missing Values", 
                    "Remove Outliers",
                    "Standardize Text",
                    "Min-Max Scaling",
                    "Z-Score Normalization",
                    "Group By Aggregation",
                    "Row Filter",
                    "Column Filter",
                    "Create Derived Column"
                ]
            )
            
            # Parameters based on transformation type
            if transformation_type == "Remove Duplicates":
                keep_option = st.radio("Keep which duplicate?", ["first", "last"])
                if st.button("Add Step"):
                    transformations.append({
                        "step": "remove_duplicates",
                        "parameters": {"keep": keep_option}
                    })
            
            elif transformation_type == "Fill Missing Values":
                strategy = st.selectbox("Fill Strategy", ["mean", "median", "mode", "constant"])
                if strategy == "constant":
                    fill_value = st.text_input("Fill Value")
                else:
                    fill_value = None
                
                if st.button("Add Step"):
                    params = {"strategy": strategy}
                    if fill_value:
                        params["fill_value"] = fill_value
                    transformations.append({
                        "step": "fill_missing_values",
                        "parameters": params
                    })
            
            elif transformation_type == "Min-Max Scaling":
                min_val = st.number_input("Min Value", value=0.0)
                max_val = st.number_input("Max Value", value=1.0)
                if st.button("Add Step"):
                    transformations.append({
                        "step": "min_max_scaling",
                        "parameters": {"feature_range": [min_val, max_val]}
                    })
        
        # Show current pipeline
        if transformations or st.session_state.get('transformations', []):
            if 'transformations' not in st.session_state:
                st.session_state.transformations = []
            
            st.subheader("🔗 Current Pipeline")
            for i, transform in enumerate(st.session_state.transformations):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{i+1}. {transform['step']}")
                with col2:
                    st.write(f"Parameters: {len(transform['parameters'])}")
                with col3:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.transformations.pop(i)
                        st.experimental_rerun()
        
        # Execute pipeline
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Preview Transformation", type="secondary"):
                st.info("Preview functionality would show sample results here")
        
        with col2:
            if st.button("▶️ Execute Pipeline", type="primary"):
                with st.spinner("Executing transformation pipeline..."):
                    # Simulate transformation execution
                    st.success("✅ Transformation pipeline executed successfully!")
                    
                    # Show results summary
                    st.subheader("📊 Transformation Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Before", "10,000")
                    with col2:
                        st.metric("Rows After", "9,847", "-153")
                    with col3:
                        st.metric("Processing Time", "2.3s")

def show_lineage():
    """Show data lineage visualization."""
    st.header("🔗 Data Lineage & Provenance")
    
    # Dataset selection for lineage
    st.subheader("Select Dataset for Lineage Analysis")
    datasets = ["Sales Data Q4", "Customer Demographics", "Product Inventory", "Processed Sales Data"]
    selected_dataset = st.selectbox("Choose dataset", datasets, key="lineage_dataset")
    
    if selected_dataset:
        # Lineage direction
        col1, col2 = st.columns(2)
        with col1:
            direction = st.radio("Lineage Direction", ["Upstream", "Downstream", "Both"])
        with col2:
            max_depth = st.slider("Maximum Depth", 1, 10, 5)
        
        if st.button("🔍 Analyze Lineage"):
            # Mock lineage data
            st.subheader("📊 Lineage Analysis Results")
            
            # Lineage statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Source Datasets", "3")
            with col2:
                st.metric("Derived Datasets", "5")
            with col3:
                st.metric("Transformations", "12")
            with col4:
                st.metric("Max Depth", "4")
            
            # Lineage graph visualization
            st.subheader("🌐 Lineage Graph")
            
            # Create a simple network-like visualization
            import networkx as nx
            
            # Mock graph data
            G = nx.DiGraph()
            G.add_edges_from([
                ("Raw Sales", "Cleaned Sales"),
                ("Customer Data", "Cleaned Sales"),
                ("Cleaned Sales", "Aggregated Sales"),
                ("Aggregated Sales", "Sales Report"),
                ("Cleaned Sales", "Sales Analysis")
            ])
            
            # Convert to plotly
            pos = nx.spring_layout(G)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=50,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Data Lineage Graph',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Interactive lineage visualization",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="#888", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            st.plotly_chart(fig, width="stretch")
            
            # Impact analysis
            st.subheader("💥 Impact Analysis")
            if selected_dataset == "Raw Sales":
                st.warning("⚠️ **High Impact Dataset**\n\nChanges to this dataset will affect 5 downstream datasets and 12 transformation processes.")
                
                impact_data = {
                    "Affected Dataset": ["Cleaned Sales", "Sales Analysis", "Sales Report", "Monthly Summary", "Customer Insights"],
                    "Impact Level": ["High", "Medium", "High", "Low", "Medium"],
                    "Transformations": [3, 2, 4, 1, 2]
                }
                st.dataframe(pd.DataFrame(impact_data), width="stretch")

def show_ml_training():
    """Show ML model training interface."""
    st.header("🤖 Machine Learning Model Training")
    
    # Model training section
    st.subheader("🚀 Train New Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset selection
        st.write("**Dataset Configuration**")
        datasets = ["Sales Data Q4", "Customer Demographics", "Product Inventory", "Marketing Campaigns"]
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        target_column = st.text_input("Target Column", placeholder="e.g., sales_amount, churn_probability")
        feature_columns = st.multiselect(
            "Feature Columns",
            ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            default=["feature_1", "feature_2", "feature_3"]
        )
        
        # Model configuration
        st.write("**Model Configuration**")
        model_type = st.selectbox(
            "Model Type",
            ["Linear Regression", "Random Forest", "XGBoost", "Logistic Regression"]
        )
        
        framework = st.selectbox("Framework", ["scikit_learn", "xgboost"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.05)
        with col_b:
            cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        
        # Hyperparameter tuning
        st.write("**Hyperparameter Tuning**")
        optimization_method = st.selectbox(
            "Optimization Method",
            ["grid_search", "random_search", "bayesian"]
        )
        max_trials = st.slider("Max Trials", 5, 50, 10)
        
        model_name = st.text_input("Model Name (Optional)", placeholder="My Awesome Model")
    
    with col2:
        st.info("📋 **Training Tips**\n\n• Choose appropriate model type for your problem\n• More trials = better optimization but longer training\n• Cross-validation helps prevent overfitting\n• Feature selection impacts model performance")
        
        # Training progress (if training is running)
        if st.session_state.get('training_in_progress', False):
            st.warning("🔄 **Training in Progress**")
            progress_bar = st.progress(0.6)
            st.write("Current step: Hyperparameter optimization")
            st.write("Estimated time remaining: 2 minutes")
    
    # Start training button
    if target_column and feature_columns:
        if st.button("🚀 Start Training", type="primary", disabled=st.session_state.get('training_in_progress', False)):
            with st.spinner("Starting model training..."):
                # Simulate training start
                st.session_state.training_in_progress = True
                st.success("✅ Training job started successfully!")
                
                # Show training configuration summary
                st.subheader("📋 Training Configuration")
                config_data = {
                    "Parameter": ["Dataset", "Target Column", "Features", "Model Type", "Framework", "Validation Split", "CV Folds", "Optimization", "Max Trials"],
                    "Value": [selected_dataset, target_column, len(feature_columns), model_type, framework, f"{validation_split:.1%}", cv_folds, optimization_method, max_trials]
                }
                st.dataframe(pd.DataFrame(config_data), width="stretch")
                
                # Mock training job ID
                st.info(f"🆔 **Training Job ID**: `train_job_12345`")
    
    # Training history and models
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Recent Training Jobs")
        
        # Mock training jobs data
        training_jobs = {
            "Job ID": ["train_job_12345", "train_job_12344", "train_job_12343", "train_job_12342"],
            "Model Type": ["Random Forest", "XGBoost", "Linear Regression", "Random Forest"],
            "Status": ["🔄 Running", "✅ Completed", "✅ Completed", "❌ Failed"],
            "Accuracy": ["-", "0.892", "0.756", "-"],
            "Started": ["2 min ago", "1 hour ago", "3 hours ago", "1 day ago"]
        }
        
        jobs_df = pd.DataFrame(training_jobs)
        st.dataframe(jobs_df, width="stretch")
        
        # Job actions
        selected_job = st.selectbox("Select job for actions", training_jobs["Job ID"])
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📊 View Details"):
                st.info("Job details would be shown here")
        with col_b:
            if st.button("📋 View Logs"):
                st.text_area("Training Logs", "2024-01-15 10:30:15 - Starting training...\n2024-01-15 10:30:20 - Loading data...\n2024-01-15 10:30:25 - Preprocessing features...", height=100)
        with col_c:
            if st.button("🛑 Cancel", disabled=True):
                st.warning("Job cannot be cancelled")
    
    with col2:
        st.subheader("🎯 Trained Models")
        
        # Mock models data
        models = {
            "Model Name": ["Sales Predictor v2", "Churn Classifier", "Revenue Forecaster", "Customer Segmenter"],
            "Type": ["Random Forest", "XGBoost", "Linear Regression", "K-Means"],
            "Status": ["🚀 Deployed", "📦 Trained", "🚀 Deployed", "📦 Trained"],
            "Performance": ["89.2%", "87.5%", "75.6%", "N/A"],
            "Created": ["1 hour ago", "2 hours ago", "1 day ago", "3 days ago"]
        }
        
        models_df = pd.DataFrame(models)
        st.dataframe(models_df, width="stretch")
        
        # Model actions
        selected_model = st.selectbox("Select model for actions", models["Model Name"])
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🚀 Deploy"):
                st.success("Model deployed successfully!")
        with col_b:
            if st.button("📈 Evaluate"):
                st.info("Model evaluation results would be shown here")
        with col_c:
            if st.button("🔮 Predict"):
                st.info("Prediction interface would be shown here")
    
    # Model comparison
    st.markdown("---")
    st.subheader("⚖️ Model Comparison")
    
    # Model selection for comparison
    models_to_compare = st.multiselect(
        "Select models to compare",
        models["Model Name"],
        default=models["Model Name"][:2]
    )
    
    if len(models_to_compare) >= 2:
        # Create comparison chart
        comparison_data = {
            "Model": models_to_compare,
            "Accuracy": [0.892, 0.875, 0.756, 0.823][:len(models_to_compare)],
            "Precision": [0.885, 0.881, 0.742, 0.819][:len(models_to_compare)],
            "Recall": [0.898, 0.869, 0.771, 0.827][:len(models_to_compare)],
            "F1-Score": [0.891, 0.875, 0.756, 0.823][:len(models_to_compare)]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Metrics comparison chart
        fig = px.bar(
            comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title='Model Performance Comparison',
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
        
        # Best model recommendation
        best_model_idx = comparison_df['Accuracy'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]['Model']
        best_accuracy = comparison_df.iloc[best_model_idx]['Accuracy']
        
        st.success(f"🏆 **Recommended Model**: {best_model} (Accuracy: {best_accuracy:.1%})")
    
    # AutoML section
    st.markdown("---")
    st.subheader("🤖 AutoML - Automated Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Let AutoML find the best model for your dataset automatically!")
        
        automl_dataset = st.selectbox("Select Dataset for AutoML", datasets, key="automl_dataset")
        automl_target = st.text_input("Target Column", key="automl_target")
        
        col_a, col_b = st.columns(2)
        with col_a:
            time_budget = st.slider("Time Budget (minutes)", 5, 120, 30)
        with col_b:
            quality_metric = st.selectbox("Quality Metric", ["accuracy", "f1", "roc_auc", "r2"])
        
        feature_selection = st.checkbox("Enable Feature Selection", value=True)
        
        if st.button("🚀 Start AutoML", type="primary"):
            with st.spinner("AutoML is analyzing your dataset and training models..."):
                # Simulate AutoML process
                import time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                steps = [
                    "Analyzing dataset characteristics...",
                    "Selecting candidate algorithms...",
                    "Training baseline models...",
                    "Optimizing hyperparameters...",
                    "Evaluating model performance...",
                    "Generating final recommendations..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)
                
                st.success("✅ AutoML completed successfully!")
                
                # Show AutoML results
                st.subheader("🏆 AutoML Results")
                
                automl_results = {
                    "Rank": [1, 2, 3, 4, 5],
                    "Algorithm": ["XGBoost", "Random Forest", "LightGBM", "Extra Trees", "Linear Model"],
                    "Score": [0.924, 0.918, 0.915, 0.912, 0.847],
                    "Training Time": ["45s", "32s", "38s", "41s", "12s"]
                }
                
                results_df = pd.DataFrame(automl_results)
                st.dataframe(results_df, width="stretch")
                
                st.info("🎯 **Best Model**: XGBoost with 92.4% accuracy")
    
    with col2:
        st.info("🤖 **AutoML Benefits**\n\n• Automatic algorithm selection\n• Hyperparameter optimization\n• Feature engineering\n• Model ensembling\n• No ML expertise required")


def show_analytics():
    """Show analytics and insights."""
    st.header("📊 Analytics & Insights")
    
    # Sample analytics dashboard
    st.subheader("📈 Data Processing Metrics")
    
    # Create sample time series data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = {
        'Date': dates,
        'Datasets Processed': [10 + i + (i % 7) * 3 for i in range(30)],
        'Transformations': [25 + i * 2 + (i % 5) * 5 for i in range(30)],
        'Data Quality Score': [85 + (i % 10) + (i % 3) * 2 for i in range(30)]
    }
    df = pd.DataFrame(data)
    
    # Time series charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(df, x='Date', y='Datasets Processed', title='Daily Dataset Processing')
        st.plotly_chart(fig1, width="stretch")
    
    with col2:
        fig2 = px.line(df, x='Date', y='Data Quality Score', title='Data Quality Trend')
        st.plotly_chart(fig2, width="stretch")
    
    # Transformation types distribution
    st.subheader("🔄 Transformation Types Usage")
    
    transform_data = {
        'Transformation': ['Remove Duplicates', 'Fill Missing', 'Normalize', 'Filter', 'Aggregate'],
        'Usage Count': [45, 38, 32, 28, 22],
        'Success Rate': [98, 95, 97, 99, 94]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.bar(transform_data, x='Transformation', y='Usage Count', title='Most Used Transformations')
        st.plotly_chart(fig3, width="stretch")
    
    with col2:
        fig4 = px.bar(transform_data, x='Transformation', y='Success Rate', title='Transformation Success Rates')
        st.plotly_chart(fig4, width="stretch")

def show_system_status():
    """Show system status and health."""
    st.header("⚙️ System Status")
    
    # Service health checks
    st.subheader("🏥 Service Health")
    
    services = [
        {"name": "API Gateway", "url": "http://localhost:8000", "status": "healthy"},
        {"name": "User Service", "url": "http://localhost:8001", "status": "healthy"},
        {"name": "Tenant Service", "url": "http://localhost:8002", "status": "healthy"},
        {"name": "Data Service", "url": "http://localhost:8003", "status": "healthy"},
        {"name": "ML Service", "url": "http://localhost:8004", "status": "healthy"}
    ]
    
    for service in services:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.write(f"**{service['name']}**")
        with col2:
            st.write(service['url'])
        with col3:
            if service['status'] == 'healthy':
                st.success("✅ Healthy")
            else:
                st.error("❌ Down")
        with col4:
            if st.button("Test", key=f"test_{service['name']}"):
                with st.spinner("Testing..."):
                    # In real implementation, make actual health check
                    st.success("✅ OK")
    
    # System metrics
    st.subheader("📊 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "45%", "↗️ +5%")
    with col2:
        st.metric("Memory Usage", "62%", "↗️ +8%")
    with col3:
        st.metric("Disk Usage", "34%", "↗️ +2%")
    with col4:
        st.metric("Network I/O", "1.2 GB/s", "↗️ +0.3 GB/s")
    
    # Database status
    st.subheader("🗄️ Database Status")
    
    databases = [
        {"name": "PostgreSQL", "status": "Connected", "connections": 12, "size": "2.3 GB"},
        {"name": "MongoDB", "status": "Connected", "connections": 8, "size": "1.8 GB"},
        {"name": "Redis", "status": "Connected", "connections": 15, "size": "256 MB"}
    ]
    
    db_df = pd.DataFrame(databases)
    st.dataframe(db_df, width="stretch")
    
    # Recent logs
    st.subheader("📝 Recent System Logs")
    
    logs = [
        {"timestamp": "2024-01-15 10:30:15", "level": "INFO", "service": "Data Service", "message": "Dataset processed successfully"},
        {"timestamp": "2024-01-15 10:29:45", "level": "INFO", "service": "API Gateway", "message": "Request routed to data service"},
        {"timestamp": "2024-01-15 10:29:30", "level": "WARN", "service": "User Service", "message": "Rate limit approaching for user"},
        {"timestamp": "2024-01-15 10:28:12", "level": "INFO", "service": "Tenant Service", "message": "Quota updated for tenant"},
        {"timestamp": "2024-01-15 10:27:55", "level": "ERROR", "service": "Data Service", "message": "Transformation failed: invalid parameters"}
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df, width="stretch")

def upload_dataset(file, name, description):
    """Upload dataset to the API (mock implementation)."""
    # In real implementation, this would make an API call
    # For now, just simulate success
    import time
    time.sleep(2)  # Simulate processing time
    return True

if __name__ == "__main__":

    main()

