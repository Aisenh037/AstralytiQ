"""
User management API routes with RBAC.
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import (
    UserResponse, UserProfileUpdateRequest, MessageResponse,
    ValidationErrorResponse
)
from ..infrastructure.dependencies import (
    get_user_repository, get_current_user, require_role, require_permission,
    UserRepository
)
from ..domain.entities import User, UserDomainService
from ..domain.rbac import Permission, RBACService
from src.shared.domain.models import UserRole, UserProfile


router = APIRouter(prefix="/users", tags=["user-management"])


@router.get("/", response_model=List[UserResponse])
async def list_users(
    tenant_id: Optional[UUID] = Query(None, description="Filter by tenant ID"),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return"),
    offset: int = Query(0, ge=0, description="Number of users to skip"),
    current_user: User = Depends(require_permission(Permission.USER_LIST)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """List users with optional filtering."""
    
    # If tenant_id is provided, check if user can access that tenant
    if tenant_id and not RBACService.user_can_access_resource(
        current_user.role, current_user.tenant_id, tenant_id, Permission.USER_LIST
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access users from different tenant"
        )
    
    # If no tenant_id provided, use current user's tenant (unless admin)
    if not tenant_id:
        if current_user.role == UserRole.ADMIN:
            # Admins can see all users
            users = await user_repository.get_active_users(limit=limit, offset=offset)
        else:
            # Non-admins can only see users from their tenant
            users = await user_repository.get_by_tenant_id(current_user.tenant_id)
            users = users[offset:offset + limit]  # Apply pagination
    else:
        users = await user_repository.get_by_tenant_id(tenant_id)
        users = users[offset:offset + limit]  # Apply pagination
    
    # Filter by role if specified
    if role:
        users = [user for user in users if user.role == role]
    
    return [UserResponse.model_validate(user) for user in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user: User = Depends(require_permission(Permission.USER_READ)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Get user by ID."""
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if current user can access this user
    if not RBACService.user_can_access_resource(
        current_user.role, current_user.tenant_id, target_user.tenant_id, Permission.USER_READ
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access user from different tenant"
        )
    
    return UserResponse.model_validate(target_user)


@router.put("/{user_id}/profile", response_model=UserResponse)
async def update_user_profile(
    user_id: UUID,
    request: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Update user profile."""
    
    # Users can update their own profile, or admins can update any profile
    if user_id != current_user.id:
        # Check if current user has permission to update other users
        if not RBACService.user_has_permission(current_user.role, Permission.USER_UPDATE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot update other user's profile"
            )
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check tenant access for cross-user updates
    if user_id != current_user.id and not RBACService.user_can_access_resource(
        current_user.role, current_user.tenant_id, target_user.tenant_id, Permission.USER_UPDATE
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot update user from different tenant"
        )
    
    # Update profile
    updated_profile = UserProfile(
        first_name=request.first_name or target_user.profile.first_name,
        last_name=request.last_name or target_user.profile.last_name,
        phone=request.phone or target_user.profile.phone,
        timezone=request.timezone or target_user.profile.timezone,
        preferences=target_user.profile.preferences  # Keep existing preferences
    )
    
    target_user.update_profile(updated_profile)
    updated_user = await user_repository.save(target_user)
    
    return UserResponse.model_validate(updated_user)


@router.put("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: UUID,
    new_role: UserRole,
    current_user: User = Depends(require_permission(Permission.USER_UPDATE)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Update user role (admin only)."""
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if current user can manage the target user
    if not RBACService.can_manage_user(
        current_user.role, target_user.role, current_user.tenant_id, target_user.tenant_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot manage this user's role"
        )
    
    # Update role
    target_user.role = new_role
    updated_user = await user_repository.save(target_user)
    
    return UserResponse.model_validate(updated_user)


@router.put("/{user_id}/activate", response_model=MessageResponse)
async def activate_user(
    user_id: UUID,
    current_user: User = Depends(require_permission(Permission.USER_UPDATE)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Activate user account."""
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if current user can manage the target user
    if not RBACService.can_manage_user(
        current_user.role, target_user.role, current_user.tenant_id, target_user.tenant_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot manage this user"
        )
    
    # Activate user
    target_user.activate()
    await user_repository.save(target_user)
    
    return MessageResponse(
        message=f"User {target_user.email} activated successfully",
        success=True
    )


@router.put("/{user_id}/deactivate", response_model=MessageResponse)
async def deactivate_user(
    user_id: UUID,
    current_user: User = Depends(require_permission(Permission.USER_DELETE)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Deactivate user account."""
    
    # Prevent self-deactivation
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if current user can manage the target user
    if not RBACService.can_manage_user(
        current_user.role, target_user.role, current_user.tenant_id, target_user.tenant_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot manage this user"
        )
    
    # Deactivate user
    target_user.deactivate()
    await user_repository.save(target_user)
    
    return MessageResponse(
        message=f"User {target_user.email} deactivated successfully",
        success=True
    )


@router.get("/{user_id}/permissions", response_model=List[str])
async def get_user_permissions(
    user_id: UUID,
    current_user: User = Depends(require_permission(Permission.USER_READ)),
    user_repository: UserRepository = Depends(get_user_repository)
):
    """Get user's permissions."""
    
    # Get the target user
    target_user = await user_repository.get_by_id(user_id)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if current user can access this user
    if not RBACService.user_can_access_resource(
        current_user.role, current_user.tenant_id, target_user.tenant_id, Permission.USER_READ
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access user from different tenant"
        )
    
    # Get permissions for the user's role
    permissions = RBACService.get_accessible_permissions(target_user.role)
    return [perm.value for perm in permissions]


@router.get("/roles/permissions", response_model=dict)
async def get_role_permissions(
    current_user: User = Depends(require_permission(Permission.USER_READ))
):
    """Get permissions for all roles."""
    
    role_permissions = {}
    for role in UserRole:
        permissions = RBACService.get_accessible_permissions(role)
        role_permissions[role.value] = [perm.value for perm in permissions]
    
    return role_permissions