"""
AstralytiQ - Enterprise MLOps Platform
======================================
Professional, scalable, and cloud-native MLOps solution.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime

# Import Backend Integration
try:
    from backend_integration import (
        check_backend_connection, authenticate_user, is_authenticated,
        get_current_user, logout_user, get_cached_metrics,
        get_cached_datasets, get_cached_models
    )
    from components.ml_studio import render_ml_studio
except ImportError:
    st.error("Critical System Error: Backend integration module missing.")
    st.stop()

# --- Configuration ---
st.set_page_config(
    page_title="AstralytiQ MLOps",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Styling ---
st.markdown("""
<style>
    /* Global Clean Typography */
    .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0f172a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Components ---

def show_login_page():
    """Clean, professional login interface."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("## Sign In")
        st.markdown("Access your enterprise MLOps dashboard.")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="admin@astralytiq.com")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("Login to Dashboard", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.warning("Please enter credentials.")
                else:
                    with st.spinner("Authenticating securely..."):
                        if authenticate_user(email, password):
                            st.success("Authentication successful.")
                            st.rerun()
                        else:
                            st.error("Invalid credentials or backend unavailable.")

        st.markdown("---")
        st.caption("🔒 Secured with JWT Authentication")
        
        # Connection Status
        if check_backend_connection():
            st.success("🟢 System Online: Connected to Cloud Engine")
        else:
            st.error("🔴 System Offline: Backend Unreachable")

def show_dashboard(user):
    """Main executive dashboard."""
    st.title(f"Welcome back, {user.get('name', 'Admin')}")
    st.markdown("Here is your real-time system overview.")
    
    # metrics
    metrics = get_cached_metrics() or {}
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Datasets", metrics.get("total_datasets", 0), delta="Stored Securely")
    c2.metric("Active Models", metrics.get("active_models", 0), delta="Production")
    c3.metric("API Requests", f"{metrics.get('api_calls_today', 0):,}", delta="Today")
    c4.metric("System Uptime", f"{metrics.get('uptime_percentage', 99.9)}%", delta="Healthy")
    
    st.markdown("### 📉 System Analytics")
    
    # Mock visual for resume - shows ability to use Plotly
    # In a real app, this would come from the backend time-series endpoint
    dates = pd.date_range(end=datetime.now(), periods=30)
    data = pd.DataFrame({
        "Date": dates,
        "Inference Volume": [x * 10 + 50 for x in range(30)],
        "Training Jobs": [x * 2 + 10 for x in range(30)]
    })
    
    fig = px.line(data, x="Date", y=["Inference Volume", "Training Jobs"], 
                 title="30-Day Platform Activity", 
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Recent Models Table
    st.markdown("### 🧩 Recent Models")
    models = get_cached_models()
    if models:
        df = pd.DataFrame(models)
        st.dataframe(
            df[["name", "type", "accuracy", "status", "created_at"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No models found. Go to ML Studio to train one.")

# --- Main Application Logic ---

def main():
    # Authentication Check
    if not is_authenticated():
        show_login_page()
        return

    user = get_current_user()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### AstralytiQ")
        st.caption("Enterprise Edition v2.1")
        
        menu = st.radio("Navigation", ["Dashboard", "ML Studio", "Data Management", "Settings"])
        
        st.markdown("---")
        if st.button("Log Out"):
            logout_user()
            st.rerun()
            
        st.markdown("---")
        # Connection Status in Sidebar
        if check_backend_connection():
            st.success("Backend: Online")
        else:
            st.error("Backend: Offline")

    # Routing
    if menu == "Dashboard":
        show_dashboard(user)
    elif menu == "ML Studio":
        # Import only when needed
        try:
            render_ml_studio()
        except Exception as e:
            st.error(f"Error loading ML Studio: {e}")
    elif menu == "Data Management":
        st.title("Data Management")
        datasets = get_cached_datasets()
        if datasets:
            st.dataframe(pd.DataFrame(datasets), use_container_width=True)
        else:
            st.info("No datasets uploaded yet.")
    elif menu == "Settings":
        st.title("System Settings")
        st.text_input("API Endpoint", value=st.secrets.get("API_BASE_URL", "http://localhost:8081"), disabled=True)
        st.button("Clear Cache", on_click=lambda: st.cache_data.clear())

if __name__ == "__main__":
    main()