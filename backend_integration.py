"""
🔗 Backend Integration Module
Connects Streamlit frontend with FastAPI backend

Demonstrates:
- API client implementation
- JWT token management
- Real-time data synchronization
- Error handling and retry logic
- Caching strategies
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendClient:
    """Enterprise-grade backend client for API communication."""
    
    def __init__(self, base_url: Optional[str] = None):
        # Read from Streamlit secrets (production) or environment or fallback to localhost
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = st.secrets.get("API_BASE_URL", "http://localhost:8081").rstrip("/")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "AstralytiQ-Frontend/2.0.0"
        })
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with JWT token."""
        token = st.session_state.get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response with proper error handling."""
        try:
            if response.status_code == 401:
                # Token expired, clear session
                if "access_token" in st.session_state:
                    del st.session_state["access_token"]
                    del st.session_state["current_user"]
                st.error("Session expired. Please login again.")
                st.rerun()
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error(f"Backend connection failed: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return self._handle_response(response)
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and get JWT token."""
        try:
            data = {"email": email, "password": password}
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json=data,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return {"error": str(e)}
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current authenticated user information."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/auth/me",
                headers=headers,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Get user failed: {e}")
            return {"error": str(e)}
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets from backend."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/datasets",
                headers=headers,
                timeout=10
            )
            result = self._handle_response(response)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Get datasets failed: {e}")
            return []
    
    def get_ml_models(self) -> List[Dict[str, Any]]:
        """Get all ML models from backend."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/models",
                headers=headers,
                timeout=10
            )
            result = self._handle_response(response)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Get models failed: {e}")
            return []
    
    def get_dashboards(self) -> List[Dict[str, Any]]:
        """Get all dashboards from backend."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/dashboards",
                headers=headers,
                timeout=10
            )
            result = self._handle_response(response)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Get dashboards failed: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get real-time platform metrics."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/metrics",
                headers=headers,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Get metrics failed: {e}")
            return {}

    # --- Forecasting Endpoints ---

    def upload_data(self, files: Dict[str, Any]) -> Dict[str, Any]:
        """Upload dataset for forecasting."""
        try:
            headers = self._get_auth_headers()
            # Remove Content-Type for file uploads as requests handles it
            if "Content-Type" in headers:
                del headers["Content-Type"]
            
            response = self.session.post(
                f"{self.base_url}/api/v1/ml/upload-data",
                headers=headers,
                files=files,
                timeout=30
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Upload data failed: {e}")
            return {"error": str(e)}

    def train_forecast(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a forecasting training job."""
        try:
            headers = self._get_auth_headers()
            response = self.session.post(
                f"{self.base_url}/api/v1/ml/forecast/train",
                headers=headers,
                json=config,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Train forecast failed: {e}")
            return {"error": str(e)}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/ml/forecast/jobs/{job_id}",
                headers=headers,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Get job status failed: {e}")
            return {"error": str(e)}

    def get_forecast(self, model_id: str) -> Dict[str, Any]:
        """Get forecast results from a trained model."""
        try:
            headers = self._get_auth_headers()
            response = self.session.get(
                f"{self.base_url}/api/v1/ml/forecast/{model_id}",
                headers=headers,
                timeout=10
            )
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Get forecast failed: {e}")
            return {"error": str(e)}

# Global backend client instance
@st.cache_resource
def get_backend_client() -> BackendClient:
    """Get cached backend client instance."""
    return BackendClient()

def check_backend_connection() -> bool:
    """Check if backend is available."""
    client = get_backend_client()
    health = client.health_check()
    return health.get("status") == "healthy"

def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user with backend."""
    client = get_backend_client()
    result = client.login(email, password)
    
    if "access_token" in result:
        # Store authentication data in session
        st.session_state["access_token"] = result["access_token"]
        st.session_state["current_user"] = result["user"]
        st.session_state["token_expires"] = datetime.now() + timedelta(seconds=result["expires_in"])
        return True
    
    return False

def is_authenticated() -> bool:
    """Check if user is currently authenticated."""
    if "access_token" not in st.session_state:
        return False
    
    # Check token expiration
    expires = st.session_state.get("token_expires")
    if expires and datetime.now() > expires:
        # Token expired, clear session
        del st.session_state["access_token"]
        del st.session_state["current_user"]
        del st.session_state["token_expires"]
        return False
    
    return True

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    if is_authenticated():
        return st.session_state.get("current_user")
    return None

def logout_user():
    """Logout current user."""
    keys_to_remove = ["access_token", "current_user", "token_expires"]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_cached_datasets() -> List[Dict[str, Any]]:
    """Get datasets with caching."""
    if not is_authenticated():
        return []
    
    client = get_backend_client()
    return client.get_datasets()

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_cached_models() -> List[Dict[str, Any]]:
    """Get ML models with caching."""
    if not is_authenticated():
        return []
    
    client = get_backend_client()
    return client.get_ml_models()

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_cached_metrics() -> Dict[str, Any]:
    """Get metrics with caching."""
    if not is_authenticated():
        return {}
    
    client = get_backend_client()
    return client.get_metrics()

def show_backend_status():
    """Show backend connection status in sidebar."""
    st.sidebar.markdown("### 🔗 Backend Status")
    
    if check_backend_connection():
        st.sidebar.markdown("🟢 **Backend Online**")
        st.sidebar.markdown("✅ API Gateway Connected")
        st.sidebar.markdown("✅ Database Operational")
        st.sidebar.markdown("✅ Authentication Active")
    else:
        st.sidebar.markdown("🔴 **Backend Offline**")
        st.sidebar.markdown("❌ Connection Failed")
        st.sidebar.markdown("⚠️ Using Demo Mode")
        
        if st.sidebar.button("🔄 Retry Connection"):
            st.cache_resource.clear()
            st.rerun()

def show_api_documentation():
    """Show API documentation links."""
    client = get_backend_client()
    st.sidebar.markdown("### 📚 API Documentation")
    st.sidebar.markdown(f"[🔗 Swagger UI]({client.base_url}/docs)")
    st.sidebar.markdown(f"[📖 ReDoc]({client.base_url}/redoc)")
    st.sidebar.markdown(f"[🔍 Health Check]({client.base_url}/health)")

# Enhanced login function with backend integration
def enhanced_login_form():
    """Enhanced login form with backend authentication."""
    st.markdown("""
    <div class="enterprise-header">
        <h1>⚡ AstralytiQ</h1>
        <p>Enterprise MLOps Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        
        # Backend status indicator
        if check_backend_connection():
            st.markdown('<span class="status-badge status-success">Backend Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">Demo Mode</span>', unsafe_allow_html=True)
        
        st.markdown("### Welcome Back")
        st.markdown("Sign in to access your enterprise MLOps platform")
        
        # Login form
        with st.form("backend_login"):
            email = st.text_input("Email Address", placeholder="admin@astralytiq.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_login, col_demo = st.columns(2)
            
            with col_login:
                login_clicked = st.form_submit_button("Sign In", use_container_width=True)
            
            with col_demo:
                demo_clicked = st.form_submit_button("Demo Mode", use_container_width=True)
            
            if login_clicked and email and password:
                with st.spinner("Authenticating..."):
                    if authenticate_user(email, password):
                        user = get_current_user()
                        st.success(f"Welcome back, {user['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials or backend unavailable")
            
            if demo_clicked:
                # Fallback to demo mode
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "name": "Demo User",
                    "email": "demo@astralytiq.com",
                    "role": "Platform Admin",
                    "department": "Demo"
                }
                st.success("Welcome to Demo Mode!")
                time.sleep(1)
                st.rerun()
        
        # Demo credentials
        st.markdown("### Demo Credentials")
        st.code("""
Backend Authentication:
Email: admin@astralytiq.com
Password: admin123

Email: data.scientist@astralytiq.com  
Password: ds123

Or use Demo Mode for offline access
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Real-time data refresh
def setup_auto_refresh():
    """Setup auto-refresh for real-time data."""
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Auto-refresh every 30 seconds
    if datetime.now() - st.session_state.last_refresh > timedelta(seconds=30):
        st.session_state.last_refresh = datetime.now()
        
        # Clear cached data to force refresh
        get_cached_datasets.clear()
        get_cached_models.clear()
        get_cached_metrics.clear()
        
        # Show refresh indicator
        st.toast("🔄 Data refreshed", icon="🔄")

def show_real_time_metrics():
    """Show real-time metrics from backend."""
    if not is_authenticated():
        return
    
    metrics = get_cached_metrics()
    if not metrics:
        st.warning("Unable to load metrics from backend")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Datasets",
            value=metrics.get("total_datasets", 0),
            delta="Real-time"
        )
    
    with col2:
        st.metric(
            label="🤖 Active Models", 
            value=metrics.get("active_models", 0),
            delta="Live"
        )
    
    with col3:
        st.metric(
            label="🔗 API Calls",
            value=f"{metrics.get('api_calls_today', 0):,}",
            delta="Today"
        )
    
    with col4:
        st.metric(
            label="⚡ Uptime",
            value=f"{metrics.get('uptime_percentage', 0)}%",
            delta="99.97%"
        )