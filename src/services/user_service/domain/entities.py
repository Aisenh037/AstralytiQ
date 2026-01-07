"""
User service domain entities and value objects.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID
import secrets

from src.shared.domain.models import User as BaseUser, UserProfile, UserRole
from src.shared.domain.base import DomainService


class User(BaseUser):
    """Extended User entity with service-specific methods."""
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        from ..infrastructure.security import password_hasher
        return password_hasher.verify_password(password, self.password_hash)
    
    def change_password(self, new_password: str) -> None:
        """Change user password."""
        from ..infrastructure.security import password_hasher
        self.password_hash = password_hasher.hash_password(new_password)
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate user account."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        self.updated_at = datetime.utcnow()
    
    def update_profile(self, profile: UserProfile) -> None:
        """Update user profile."""
        self.profile = profile
        self.updated_at = datetime.utcnow()
    
    @classmethod
    def create_new_user(
        cls,
        email: str,
        password: str,
        role: UserRole,
        tenant_id: UUID,
        profile: Optional[UserProfile] = None
    ) -> "User":
        """Create a new user with hashed password."""
        from ..infrastructure.security import password_hasher, email_validator
        
        # Normalize email
        normalized_email = email_validator.normalize_email(email)
        
        return cls(
            email=normalized_email,
            password_hash=password_hasher.hash_password(password),
            role=role,
            tenant_id=tenant_id,
            profile=profile or UserProfile()
        )


class PasswordResetToken:
    """Password reset token value object."""
    
    def __init__(self, user_id: UUID, expires_at: datetime):
        self.user_id = user_id
        self.token = self._generate_token()
        self.expires_at = expires_at
        self.created_at = datetime.utcnow()
    
    def _generate_token(self) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(32)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    def is_valid_for_user(self, user_id: UUID) -> bool:
        """Check if token is valid for the given user."""
        return self.user_id == user_id and not self.is_expired()


class UserDomainService(DomainService):
    """Domain service for user-related business logic."""
    
    @staticmethod
    def can_user_access_tenant(user: User, tenant_id: UUID) -> bool:
        """Check if user can access the specified tenant."""
        return user.tenant_id == tenant_id and user.is_active
    
    @staticmethod
    def can_user_perform_action(user: User, required_role: UserRole) -> bool:
        """Check if user has sufficient role to perform action."""
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level and user.is_active
    
    @staticmethod
    def is_email_valid_format(email: str) -> bool:
        """Validate email format."""
        from ..infrastructure.security import email_validator
        return email_validator.validate_email_format(email)
    
    @staticmethod
    def is_password_strong(password: str) -> bool:
        """Check if password meets strength requirements."""
        from ..infrastructure.security import password_validator
        is_valid, _ = password_validator.validate_password_strength(password)
        return is_valid