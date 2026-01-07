"""
User service repository implementations.
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..domain.entities import User, PasswordResetToken
from ..domain.repositories import IUserRepository, IPasswordResetRepository
from src.shared.infrastructure.repositories import SQLAlchemyRepository, RedisRepository
from src.shared.infrastructure.models import UserModel


class UserRepository(SQLAlchemyRepository[User], IUserRepository):
    """SQLAlchemy implementation of user repository."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, UserModel)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self.session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if user_model:
            return self._model_to_entity(user_model)
        return None
    
    async def get_by_tenant_id(self, tenant_id: UUID) -> List[User]:
        """Get all users for a tenant."""
        stmt = select(UserModel).where(
            UserModel.tenant_id == tenant_id,
            UserModel.is_active == True
        )
        result = await self.session.execute(stmt)
        user_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in user_models]
    
    async def email_exists(self, email: str) -> bool:
        """Check if email already exists."""
        stmt = select(UserModel.id).where(UserModel.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get active users with pagination."""
        stmt = (
            select(UserModel)
            .where(UserModel.is_active == True)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        user_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in user_models]
    
    async def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp."""
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(last_login=datetime.utcnow())
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def save(self, entity: User) -> User:
        """Save user entity."""
        # Convert entity to SQLAlchemy model
        user_model = self._entity_to_model(entity)
        
        # Check if it's an update or insert
        existing = await self.session.get(UserModel, entity.id)
        if existing:
            # Update existing
            for key, value in user_model.__dict__.items():
                if not key.startswith('_'):
                    setattr(existing, key, value)
            user_model = existing
        else:
            # Insert new
            self.session.add(user_model)
        
        await self.session.flush()
        await self.session.refresh(user_model)
        
        return self._model_to_entity(user_model)
    
    def _model_to_entity(self, model: UserModel) -> User:
        """Convert SQLAlchemy model to domain entity."""
        from src.shared.domain.models import UserProfile, UserRole
        
        return User(
            id=model.id,
            email=model.email,
            password_hash=model.password_hash,
            role=UserRole(model.role),
            tenant_id=model.tenant_id,
            profile=UserProfile(**model.profile) if model.profile else UserProfile(),
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_login=model.last_login,
            email_verified=model.email_verified,
            is_active=model.is_active
        )
    
    def _entity_to_model(self, entity: User) -> UserModel:
        """Convert domain entity to SQLAlchemy model."""
        return UserModel(
            id=entity.id,
            email=entity.email,
            password_hash=entity.password_hash,
            role=entity.role.value,
            tenant_id=entity.tenant_id,
            profile=entity.profile.dict() if entity.profile else {},
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_login=entity.last_login,
            email_verified=entity.email_verified,
            is_active=entity.is_active
        )


class PasswordResetRepository(IPasswordResetRepository):
    """Redis implementation of password reset token repository."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = RedisRepository(redis_client, "password_reset")
    
    async def save_token(self, token: PasswordResetToken) -> bool:
        """Save password reset token."""
        token_data = {
            "user_id": str(token.user_id),
            "expires_at": token.expires_at.isoformat(),
            "created_at": token.created_at.isoformat()
        }
        
        # Calculate TTL in seconds
        ttl = int((token.expires_at - datetime.utcnow()).total_seconds())
        if ttl <= 0:
            return False
        
        return await self.redis.set_hash(token.token, token_data) and \
               await self.redis.set(f"user_token:{token.user_id}", token.token, ttl=ttl)
    
    async def get_token(self, token_value: str) -> Optional[PasswordResetToken]:
        """Get password reset token by value."""
        token_data = await self.redis.get_hash(token_value)
        
        if not token_data:
            return None
        
        try:
            user_id = UUID(token_data["user_id"])
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            
            token = PasswordResetToken(user_id, expires_at)
            token.token = token_value
            token.created_at = datetime.fromisoformat(token_data["created_at"])
            
            return token
        except (KeyError, ValueError):
            return None
    
    async def delete_token(self, token_value: str) -> bool:
        """Delete password reset token."""
        # Get token to find user_id
        token = await self.get_token(token_value)
        if not token:
            return False
        
        # Delete both the token hash and user mapping
        await self.redis.delete(token_value)
        await self.redis.delete(f"user_token:{token.user_id}")
        
        return True
    
    async def delete_user_tokens(self, user_id: UUID) -> bool:
        """Delete all tokens for a user."""
        user_token_key = f"user_token:{user_id}"
        token_value = await self.redis.get(user_token_key)
        
        if token_value:
            await self.redis.delete(token_value)
            await self.redis.delete(user_token_key)
            return True
        
        return False