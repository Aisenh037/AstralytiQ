"""
Password reset service implementation.
"""
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import redis.asyncio as redis

from ..domain.entities import PasswordResetToken
from ..domain.repositories import IPasswordResetRepository
from .security import token_generator
from src.shared.infrastructure.database import get_redis_client


class PasswordResetService:
    """Password reset service with Redis storage."""
    
    def __init__(self, redis_client: redis.Redis, token_expiry_hours: int = 1):
        self.redis = redis_client
        self.token_expiry_hours = token_expiry_hours
        self.key_prefix = "password_reset"
    
    async def create_reset_token(self, user_id: UUID) -> PasswordResetToken:
        """Create a new password reset token."""
        expires_at = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        reset_token = PasswordResetToken(user_id, expires_at)
        
        # Store token in Redis
        token_key = f"{self.key_prefix}:{reset_token.token}"
        user_key = f"{self.key_prefix}_user:{user_id}"
        
        token_data = {
            "user_id": str(user_id),
            "expires_at": expires_at.isoformat(),
            "created_at": reset_token.created_at.isoformat()
        }
        
        # Calculate TTL in seconds
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        
        # Store token data with expiration
        await self.redis.hset(token_key, mapping=token_data)
        await self.redis.expire(token_key, ttl)
        
        # Store user -> token mapping (for cleanup)
        await self.redis.set(user_key, reset_token.token, ex=ttl)
        
        return reset_token
    
    async def get_reset_token(self, token: str) -> Optional[PasswordResetToken]:
        """Get password reset token by token string."""
        token_key = f"{self.key_prefix}:{token}"
        token_data = await self.redis.hgetall(token_key)
        
        if not token_data:
            return None
        
        try:
            user_id = UUID(token_data["user_id"])
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            created_at = datetime.fromisoformat(token_data["created_at"])
            
            reset_token = PasswordResetToken(user_id, expires_at)
            reset_token.token = token
            reset_token.created_at = created_at
            
            return reset_token
        except (KeyError, ValueError):
            return None
    
    async def verify_reset_token(self, token: str, user_id: UUID) -> bool:
        """Verify that token is valid for the given user."""
        reset_token = await self.get_reset_token(token)
        
        if not reset_token:
            return False
        
        return reset_token.is_valid_for_user(user_id)
    
    async def consume_reset_token(self, token: str) -> Optional[UUID]:
        """Consume (delete) reset token and return user ID if valid."""
        reset_token = await self.get_reset_token(token)
        
        if not reset_token or reset_token.is_expired():
            return None
        
        user_id = reset_token.user_id
        
        # Delete token and user mapping
        token_key = f"{self.key_prefix}:{token}"
        user_key = f"{self.key_prefix}_user:{user_id}"
        
        await self.redis.delete(token_key)
        await self.redis.delete(user_key)
        
        return user_id
    
    async def revoke_user_tokens(self, user_id: UUID) -> bool:
        """Revoke all reset tokens for a user."""
        user_key = f"{self.key_prefix}_user:{user_id}"
        token = await self.redis.get(user_key)
        
        if token:
            token_key = f"{self.key_prefix}:{token}"
            await self.redis.delete(token_key)
            await self.redis.delete(user_key)
            return True
        
        return False
    
    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens (Redis handles this automatically with TTL)."""
        # Redis automatically removes expired keys, so this is mainly for monitoring
        pattern = f"{self.key_prefix}:*"
        keys = await self.redis.keys(pattern)
        return len(keys)


class EmailService:
    """Email service for sending password reset emails."""
    
    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 587):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
    
    async def send_password_reset_email(
        self,
        email: str,
        reset_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """Send password reset email."""
        # In a real implementation, you would:
        # 1. Create HTML email template
        # 2. Generate reset URL with token
        # 3. Send email via SMTP
        
        reset_url = f"https://yourapp.com/reset-password?token={reset_token}"
        
        # For now, just log the reset URL (in production, send actual email)
        print(f"Password reset email for {email}:")
        print(f"Reset URL: {reset_url}")
        print(f"Token expires in 1 hour")
        
        return True
    
    async def send_password_changed_notification(
        self,
        email: str,
        user_name: Optional[str] = None
    ) -> bool:
        """Send password changed notification email."""
        # In a real implementation, send security notification
        print(f"Password changed notification sent to {email}")
        return True


# Global service instances (would be injected via DI in production)
async def get_password_reset_service() -> PasswordResetService:
    """Get password reset service instance."""
    redis_client = await get_redis_client()
    return PasswordResetService(redis_client)


async def get_email_service() -> EmailService:
    """Get email service instance."""
    return EmailService()