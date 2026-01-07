"""
API schemas for user service.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.shared.domain.models import UserRole


# Request schemas
class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class RegisterRequest(BaseModel):
    """User registration request."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")
    tenant_id: UUID = Field(..., description="Tenant ID")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: str = Field(..., description="User email address")


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str = Field(..., description="Refresh token")


class UserProfileUpdateRequest(BaseModel):
    """User profile update request."""
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    timezone: Optional[str] = Field(None, description="User timezone")


# Response schemas
class UserProfileResponse(BaseModel):
    """User profile response."""
    first_name: Optional[str]
    last_name: Optional[str]
    phone: Optional[str]
    timezone: str
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """User response schema."""
    id: UUID
    email: str
    role: UserRole
    tenant_id: UUID
    profile: UserProfileResponse
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]
    email_verified: bool
    is_active: bool
    
    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """Login response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


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
    details: list[str]
    success: bool = False