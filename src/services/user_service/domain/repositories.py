"""
User service repository interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import User, PasswordResetToken
from src.shared.domain.base import Repository


class IUserRepository(Repository):
    """User repository interface."""
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        pass
    
    @abstractmethod
    async def get_by_tenant_id(self, tenant_id: UUID) -> List[User]:
        """Get all users for a tenant."""
        pass
    
    @abstractmethod
    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        pass
    
    @abstractmethod
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users with pagination."""
        pass
    
    @abstractmethod
    async def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp."""
        pass


class IPasswordResetRepository(ABC):
    """Password reset token repository interface."""
    
    @abstractmethod
    async def save_token(self, token: PasswordResetToken) -> bool:
        """Save password reset token."""
        pass
    
    @abstractmethod
    async def get_token(self, token_value: str) -> Optional[PasswordResetToken]:
        """Get password reset token by value."""
        pass
    
    @abstractmethod
    async def delete_token(self, token_value: str) -> bool:
        """Delete password reset token."""
        pass
    
    @abstractmethod
    async def delete_user_tokens(self, user_id: UUID) -> bool:
        """Delete all tokens for a user."""
        pass