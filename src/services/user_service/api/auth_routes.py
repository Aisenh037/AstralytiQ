"""
Authentication API routes.
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import (
    LoginRequest, LoginResponse, RegisterRequest, UserResponse,
    PasswordChangeRequest, PasswordResetRequest, PasswordResetConfirmRequest,
    RefreshTokenRequest, TokenResponse, MessageResponse, ValidationErrorResponse
)
from ..infrastructure.auth import auth_service
from ..infrastructure.dependencies import (
    get_user_repository, get_current_user, UserRepository
)
from ..infrastructure.security import password_validator, email_validator
from ..infrastructure.password_reset import get_password_reset_service, get_email_service
from ..domain.entities import User, UserDomainService
from src.shared.domain.models import UserProfile, UserRole


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Register a new user."""
    
    # Validate email format
    if not UserDomainService.is_email_valid_format(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password strength
    is_strong, password_errors = password_validator.validate_password_strength(request.password)
    if not is_strong:
        return ValidationErrorResponse(
            error="Password does not meet strength requirements",
            details=password_errors
        )
    
    # Check if email already exists
    if await user_repository.email_exists(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered"
        )
    
    # Create user profile
    profile = UserProfile(
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone
    )
    
    # Create new user
    user = User.create_new_user(
        email=request.email,
        password=request.password,
        role=request.role,
        tenant_id=request.tenant_id,
        profile=profile
    )
    
    # Save user
    saved_user = await user_repository.save(user)
    
    return UserResponse.model_validate(saved_user)


@router.post("/login", response_model=LoginResponse)
async def login_user(
    request: LoginRequest,
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Login user and return tokens."""
    
    # Authenticate user
    tokens = await auth_service.login_user(
        email=request.email,
        password=request.password,
        user_repository=user_repository
    )
    
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user for response
    user = await user_repository.get_by_email(request.email)
    
    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        user=UserResponse.model_validate(user)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Refresh access token."""
    
    tokens = await auth_service.refresh_access_token(
        refresh_token=request.refresh_token,
        user_repository=user_repository
    )
    
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in
    )


@router.post("/logout", response_model=MessageResponse)
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """Logout current user."""
    
    # In a production system, you would blacklist the token
    # For now, we'll just return success
    
    return MessageResponse(
        message="Successfully logged out",
        success=True
    )


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Change user password."""
    
    # Verify current password
    if not current_user.verify_password(request.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password strength
    is_strong, password_errors = password_validator.validate_password_strength(request.new_password)
    if not is_strong:
        return ValidationErrorResponse(
            error="New password does not meet strength requirements",
            details=password_errors
        )
    
    # Change password
    current_user.change_password(request.new_password)
    await user_repository.save(current_user)
    
    return MessageResponse(
        message="Password changed successfully",
        success=True
    )


@router.post("/request-password-reset", response_model=MessageResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    user_repository: UserRepository = Depends(get_user_repository),
    password_reset_service = Depends(get_password_reset_service),
    email_service = Depends(get_email_service)
):
    """Request password reset."""
    
    # Check if user exists
    user = await user_repository.get_by_email(request.email)
    if not user:
        # Don't reveal if email exists or not
        return MessageResponse(
            message="If the email exists, a password reset link has been sent",
            success=True
        )
    
    # Revoke any existing tokens for this user
    await password_reset_service.revoke_user_tokens(user.id)
    
    # Create new password reset token
    reset_token = await password_reset_service.create_reset_token(user.id)
    
    # Send email with reset link
    await email_service.send_password_reset_email(
        email=user.email,
        reset_token=reset_token.token,
        user_name=user.profile.first_name
    )
    
    return MessageResponse(
        message="If the email exists, a password reset link has been sent",
        success=True
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    request: PasswordResetConfirmRequest,
    user_repository: UserRepository = Depends(get_user_repository),
    password_reset_service = Depends(get_password_reset_service),
    email_service = Depends(get_email_service)
):
    """Reset password using token."""
    
    # Validate new password strength
    is_strong, password_errors = password_validator.validate_password_strength(request.new_password)
    if not is_strong:
        return ValidationErrorResponse(
            error="New password does not meet strength requirements",
            details=password_errors
        )
    
    # Consume the reset token
    user_id = await password_reset_service.consume_reset_token(request.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Get user and update password
    user = await user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update password
    user.change_password(request.new_password)
    await user_repository.save(user)
    
    # Send confirmation email
    await email_service.send_password_changed_notification(
        email=user.email,
        user_name=user.profile.first_name
    )
    
    return MessageResponse(
        message="Password reset successfully",
        success=True
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse.model_validate(current_user)