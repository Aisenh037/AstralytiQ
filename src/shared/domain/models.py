"""
Core domain models for the enterprise SaaS platform.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from .base import Entity, ValueObject, AggregateRoot


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class SubscriptionPlan(str, Enum):
    """Subscription plan types."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class UserProfile(ValueObject):
    """User profile information."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    timezone: str = "UTC"
    preferences: Dict = Field(default_factory=dict)


class User(AggregateRoot):
    """User entity."""
    email: str
    password_hash: str
    role: UserRole
    tenant_id: UUID
    profile: UserProfile = Field(default_factory=UserProfile)
    last_login: Optional[datetime] = None
    email_verified: bool = False
    
    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class TenantSettings(ValueObject):
    """Tenant-specific settings."""
    branding: Dict = Field(default_factory=dict)
    features: Dict = Field(default_factory=dict)
    limits: Dict = Field(default_factory=dict)
    integrations: Dict = Field(default_factory=dict)


class Tenant(AggregateRoot):
    """Tenant entity for multi-tenancy."""
    name: str
    domain: Optional[str] = None
    subscription_plan: SubscriptionPlan
    settings: TenantSettings = Field(default_factory=TenantSettings)
    owner_id: Optional[UUID] = None
    
    def update_subscription(self, plan: SubscriptionPlan) -> None:
        """Update subscription plan."""
        self.subscription_plan = plan
        self.updated_at = datetime.utcnow()


class DataSourceType(str, Enum):
    """Types of data sources."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    DATABASE = "database"
    API = "api"


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DataSchema(ValueObject):
    """Data schema definition."""
    columns: List[Dict] = Field(default_factory=list)
    primary_key: Optional[str] = None
    indexes: List[str] = Field(default_factory=list)
    constraints: Dict = Field(default_factory=dict)


class Dataset(AggregateRoot):
    """Dataset entity."""
    name: str
    description: Optional[str] = None
    tenant_id: UUID
    created_by: UUID
    file_path: str
    file_size: int
    status: DatasetStatus = DatasetStatus.UPLOADED
    schema: Optional[Dict] = None
    metadata: Optional[Dict] = None
    
    # Legacy fields for compatibility
    data_schema: Optional[DataSchema] = None
    source_type: Optional[DataSourceType] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None


class ModelType(str, Enum):
    """Types of ML models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ARIMA = "arima"
    PROPHET = "prophet"


class ModelStatus(str, Enum):
    """Model status."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ModelMetrics(ValueObject):
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict = Field(default_factory=dict)


class MLModel(AggregateRoot):
    """ML Model entity."""
    name: str
    tenant_id: UUID
    ml_model_type: ModelType
    version: str
    parameters: Dict = Field(default_factory=dict)
    metrics: Optional[ModelMetrics] = None
    artifact_path: Optional[str] = None
    status: ModelStatus = ModelStatus.TRAINING
    
    class Config:
        protected_namespaces = ()
    
    def update_status(self, status: ModelStatus) -> None:
        """Update model status."""
        self.status = status
        self.updated_at = datetime.utcnow()


class Prediction(Entity):
    """Prediction entity."""
    ml_model_id: UUID
    input_data: Dict
    output_data: Dict
    confidence: Optional[float] = None
    execution_time_ms: Optional[int] = None
    
    class Config:
        protected_namespaces = ()