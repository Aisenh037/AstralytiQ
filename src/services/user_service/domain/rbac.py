"""
Role-Based Access Control (RBAC) domain models and services.
"""
from enum import Enum
from typing import List, Set, Dict, Optional
from uuid import UUID

from pydantic import BaseModel
from src.shared.domain.base import ValueObject, DomainService
from src.shared.domain.models import UserRole


class Permission(str, Enum):
    """System permissions."""
    
    # User management permissions
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_LIST = "user:list"
    
    # Tenant management permissions
    TENANT_CREATE = "tenant:create"
    TENANT_READ = "tenant:read"
    TENANT_UPDATE = "tenant:update"
    TENANT_DELETE = "tenant:delete"
    TENANT_LIST = "tenant:list"
    
    # Data management permissions
    DATA_UPLOAD = "data:upload"
    DATA_READ = "data:read"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"
    DATA_PROCESS = "data:process"
    DATA_EXPORT = "data:export"
    
    # ML model permissions
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_PREDICT = "model:predict"
    
    # Dashboard permissions
    DASHBOARD_CREATE = "dashboard:create"
    DASHBOARD_READ = "dashboard:read"
    DASHBOARD_UPDATE = "dashboard:update"
    DASHBOARD_DELETE = "dashboard:delete"
    DASHBOARD_SHARE = "dashboard:share"
    
    # Billing permissions
    BILLING_READ = "billing:read"
    BILLING_UPDATE = "billing:update"
    
    # System administration permissions
    SYSTEM_ADMIN = "system:admin"
    AUDIT_READ = "audit:read"


class ResourceType(str, Enum):
    """Resource types for permission checking."""
    USER = "user"
    TENANT = "tenant"
    DATASET = "dataset"
    MODEL = "model"
    DASHBOARD = "dashboard"
    BILLING = "billing"
    SYSTEM = "system"


class RolePermissions(ValueObject):
    """Role permissions mapping."""
    role: UserRole
    permissions: Set[Permission]
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if role has any of the specified permissions."""
        return any(perm in self.permissions for perm in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if role has all of the specified permissions."""
        return all(perm in self.permissions for perm in permissions)


class PermissionContext(BaseModel):
    """Context for permission checking."""
    user_id: UUID
    tenant_id: UUID
    resource_type: ResourceType
    resource_id: Optional[UUID] = None
    action: Permission
    
    class Config:
        use_enum_values = True


class RBACService(DomainService):
    """Role-Based Access Control service."""
    
    # Define role permissions
    ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
        UserRole.VIEWER: {
            Permission.USER_READ,
            Permission.TENANT_READ,
            Permission.DATA_READ,
            Permission.MODEL_READ,
            Permission.MODEL_PREDICT,
            Permission.DASHBOARD_READ,
            Permission.BILLING_READ,
        },
        
        UserRole.ANALYST: {
            # Inherit viewer permissions
            *{
                Permission.USER_READ,
                Permission.TENANT_READ,
                Permission.DATA_READ,
                Permission.MODEL_READ,
                Permission.MODEL_PREDICT,
                Permission.DASHBOARD_READ,
                Permission.BILLING_READ,
            },
            # Additional analyst permissions
            Permission.DATA_UPLOAD,
            Permission.DATA_UPDATE,
            Permission.DATA_PROCESS,
            Permission.DATA_EXPORT,
            Permission.MODEL_CREATE,
            Permission.MODEL_UPDATE,
            Permission.MODEL_TRAIN,
            Permission.DASHBOARD_CREATE,
            Permission.DASHBOARD_UPDATE,
            Permission.DASHBOARD_SHARE,
        },
        
        UserRole.ADMIN: {
            # All permissions
            *[perm for perm in Permission],
        }
    }
    
    @classmethod
    def get_role_permissions(cls, role: UserRole) -> RolePermissions:
        """Get permissions for a role."""
        permissions = cls.ROLE_PERMISSIONS.get(role, set())
        return RolePermissions(role=role, permissions=permissions)
    
    @classmethod
    def user_has_permission(
        cls,
        user_role: UserRole,
        permission: Permission,
        context: Optional[PermissionContext] = None
    ) -> bool:
        """Check if user has specific permission."""
        role_permissions = cls.get_role_permissions(user_role)
        
        # Check basic permission
        if not role_permissions.has_permission(permission):
            return False
        
        # Additional context-based checks can be added here
        if context:
            return cls._check_context_permissions(user_role, permission, context)
        
        return True
    
    @classmethod
    def user_can_access_resource(
        cls,
        user_role: UserRole,
        user_tenant_id: UUID,
        resource_tenant_id: UUID,
        permission: Permission
    ) -> bool:
        """Check if user can access resource based on tenant isolation."""
        # Check basic permission first
        if not cls.user_has_permission(user_role, permission):
            return False
        
        # Tenant isolation check
        if user_tenant_id != resource_tenant_id:
            # Only system admins can access cross-tenant resources
            return user_role == UserRole.ADMIN and Permission.SYSTEM_ADMIN in cls.ROLE_PERMISSIONS[user_role]
        
        return True
    
    @classmethod
    def get_accessible_permissions(cls, user_role: UserRole) -> List[Permission]:
        """Get list of permissions accessible to a role."""
        role_permissions = cls.get_role_permissions(user_role)
        return list(role_permissions.permissions)
    
    @classmethod
    def can_manage_user(
        cls,
        manager_role: UserRole,
        target_role: UserRole,
        manager_tenant_id: UUID,
        target_tenant_id: UUID
    ) -> bool:
        """Check if a user can manage another user."""
        # Must have user management permission
        if not cls.user_has_permission(manager_role, Permission.USER_UPDATE):
            return False
        
        # Tenant isolation check
        if manager_tenant_id != target_tenant_id and manager_role != UserRole.ADMIN:
            return False
        
        # Role hierarchy check - can't manage users with equal or higher roles
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        manager_level = role_hierarchy.get(manager_role, 0)
        target_level = role_hierarchy.get(target_role, 0)
        
        return manager_level > target_level
    
    @classmethod
    def _check_context_permissions(
        cls,
        user_role: UserRole,
        permission: Permission,
        context: PermissionContext
    ) -> bool:
        """Check context-specific permissions."""
        # Add custom business logic here
        # For example, check if user owns the resource, etc.
        
        # For now, just return True if basic permission check passed
        return True


class PermissionDecorator:
    """Decorator for permission-based access control."""
    
    def __init__(self, required_permission: Permission):
        self.required_permission = required_permission
    
    def __call__(self, func):
        """Decorator function."""
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependencies
            # The actual implementation would extract user from request
            # and check permissions before calling the function
            return func(*args, **kwargs)
        return wrapper


# Convenience functions for common permission checks
def require_permission(permission: Permission):
    """Decorator factory for requiring specific permission."""
    return PermissionDecorator(permission)


def can_create_user(user_role: UserRole) -> bool:
    """Check if user can create other users."""
    return RBACService.user_has_permission(user_role, Permission.USER_CREATE)


def can_read_user(user_role: UserRole) -> bool:
    """Check if user can read user information."""
    return RBACService.user_has_permission(user_role, Permission.USER_READ)


def can_update_user(user_role: UserRole) -> bool:
    """Check if user can update user information."""
    return RBACService.user_has_permission(user_role, Permission.USER_UPDATE)


def can_delete_user(user_role: UserRole) -> bool:
    """Check if user can delete users."""
    return RBACService.user_has_permission(user_role, Permission.USER_DELETE)


def can_manage_tenant(user_role: UserRole) -> bool:
    """Check if user can manage tenant settings."""
    return RBACService.user_has_permission(user_role, Permission.TENANT_UPDATE)


def can_upload_data(user_role: UserRole) -> bool:
    """Check if user can upload data."""
    return RBACService.user_has_permission(user_role, Permission.DATA_UPLOAD)


def can_train_models(user_role: UserRole) -> bool:
    """Check if user can train ML models."""
    return RBACService.user_has_permission(user_role, Permission.MODEL_TRAIN)


def can_access_billing(user_role: UserRole) -> bool:
    """Check if user can access billing information."""
    return RBACService.user_has_permission(user_role, Permission.BILLING_READ)