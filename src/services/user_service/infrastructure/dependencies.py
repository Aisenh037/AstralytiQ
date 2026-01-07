"""
FastAPI dependencies for user service.
"""
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import jwt_manager, TokenData
from .repositories import UserRepository
from ..domain.entities import User
from src.shared.domain.models import UserRole
from src.shared.infrastructure.database import get_postgres_session


# Security scheme
security = HTTPBearer()


async def get_user_repository(
    session: AsyncSession = Depends(get_postgres_session)
) -> UserRepository:
    """Get user repository dependency."""
    return UserRepository(session)


async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Get current user from JWT token."""
    token = credentials.credentials
    
    # Verify token
    token_data = jwt_manager.verify_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data


async def get_current_user(
    token_data: TokenData = Depends(get_current_user_token),
    user_repository: UserRepository = Depends(get_user_repository)
) -> User:
    """Get current authenticated user."""
    user = await user_repository.get_by_id(UUID(token_data.user_id))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (alias for clarity)."""
    return current_user


def require_role(required_role: UserRole):
    """Dependency factory for role-based access control."""
    
    async def check_role(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Check if user has required role."""
        from ..domain.rbac import RBACService
        
        # Check if user role meets minimum requirement
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )
        
        return current_user
    
    return check_role


def require_permission(required_permission):
    """Dependency factory for permission-based access control."""
    
    async def check_permission(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Check if user has required permission."""
        from ..domain.rbac import RBACService
        
        if not RBACService.user_has_permission(current_user.role, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required permission: {required_permission.value}"
            )
        
        return current_user
    
    return check_permission


def require_tenant_access(tenant_id: UUID):
    """Dependency factory for tenant-based access control."""
    
    async def check_tenant_access(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Check if user can access the specified tenant."""
        from ..domain.entities import UserDomainService
        
        if not UserDomainService.can_user_access_tenant(current_user, tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this tenant"
            )
        
        return current_user
    
    return check_tenant_access


# Convenience dependencies for common roles
require_admin = require_role(UserRole.ADMIN)
require_analyst = require_role(UserRole.ANALYST)
require_viewer = require_role(UserRole.VIEWER)


class OptionalAuth:
    """Optional authentication dependency."""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        user_repository: UserRepository = Depends(get_user_repository)
    ) -> Optional[User]:
        """Get current user if authenticated, None otherwise."""
        if not credentials:
            return None
        
        try:
            token_data = jwt_manager.verify_token(credentials.credentials)
            if not token_data:
                return None
            
            user = await user_repository.get_by_id(UUID(token_data.user_id))
            if not user or not user.is_active:
                return None
            
            return user
        except Exception:
            return None


# Instance for optional authentication
optional_auth = OptionalAuth()