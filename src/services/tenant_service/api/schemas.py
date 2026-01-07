"""
API schemas for tenant service.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field

from src.shared.domain.models import SubscriptionPlan


# Request schemas
class CreateTenantRequest(BaseModel):
    """Create tenant request."""
    name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    domain: Optional[str] = Field(None, description="Custom domain")
    subscription_plan: SubscriptionPlan = Field(default=SubscriptionPlan.FREE, description="Subscription plan")
    owner_id: Optional[UUID] = Field(None, description="Owner user ID")


class UpdateTenantRequest(BaseModel):
    """Update tenant request."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Tenant name")
    domain: Optional[str] = Field(None, description="Custom domain")


class UpdateSubscriptionRequest(BaseModel):
    """Update subscription request."""
    subscription_plan: SubscriptionPlan = Field(..., description="New subscription plan")
    force_downgrade: bool = Field(default=False, description="Force downgrade even if it violates quotas")


class TenantBrandingRequest(BaseModel):
    """Tenant branding update request."""
    logo_url: Optional[str] = Field(None, description="Logo URL")
    primary_color: str = Field("#007bff", description="Primary color (hex)")
    secondary_color: str = Field("#6c757d", description="Secondary color (hex)")
    company_name: Optional[str] = Field(None, description="Company name")
    custom_domain: Optional[str] = Field(None, description="Custom domain")
    favicon_url: Optional[str] = Field(None, description="Favicon URL")


class TenantSettingsRequest(BaseModel):
    """Tenant settings update request."""
    settings: Dict[str, Any] = Field(..., description="Settings to update")


# Response schemas
class TenantBrandingResponse(BaseModel):
    """Tenant branding response."""
    logo_url: Optional[str]
    primary_color: str
    secondary_color: str
    company_name: Optional[str]
    custom_domain: Optional[str]
    favicon_url: Optional[str]
    
    class Config:
        from_attributes = True


class TenantFeaturesResponse(BaseModel):
    """Tenant features response."""
    advanced_analytics: bool
    custom_models: bool
    api_access: bool
    data_export: bool
    white_labeling: bool
    sso_integration: bool
    audit_logs: bool
    priority_support: bool
    
    class Config:
        from_attributes = True


class TenantQuotaResponse(BaseModel):
    """Tenant quota response."""
    max_users: int
    max_datasets: int
    max_models: int
    max_storage_gb: int
    max_api_calls_per_month: int
    max_concurrent_jobs: int
    
    class Config:
        from_attributes = True


class TenantSettingsResponse(BaseModel):
    """Tenant settings response."""
    branding: Dict[str, Any]
    features: Dict[str, Any]
    limits: Dict[str, Any]
    integrations: Dict[str, Any]
    
    class Config:
        from_attributes = True


class TenantResponse(BaseModel):
    """Tenant response schema."""
    id: UUID
    name: str
    domain: Optional[str]
    subscription_plan: SubscriptionPlan
    settings: TenantSettingsResponse
    owner_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class TenantSummaryResponse(BaseModel):
    """Tenant summary response (for lists)."""
    id: UUID
    name: str
    domain: Optional[str]
    subscription_plan: SubscriptionPlan
    owner_id: Optional[UUID]
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class TenantStatsResponse(BaseModel):
    """Tenant statistics response."""
    user_count: int
    dataset_count: int
    model_count: int
    storage_gb: float
    storage_bytes: int


class TenantUsageResponse(BaseModel):
    """Tenant usage response."""
    tenant_id: UUID
    current_usage: Dict[str, int]
    quota_limits: TenantQuotaResponse
    usage_percentages: Dict[str, float]
    warnings: List[Dict[str, Any]]
    status: str


class QuotaViolationResponse(BaseModel):
    """Quota violation response."""
    type: str
    current_usage: int
    limit: int
    message: str
    timestamp: datetime


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    success: bool = False


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str
    details: List[str]
    success: bool = False