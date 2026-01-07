"""
JWT authentication and token management.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID
import os

from jose import JWTError, jwt
from pydantic import BaseModel

from ..domain.entities import User
from src.shared.domain.models import UserRole


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    email: str
    role: str
    tenant_id: str
    exp: int
    iat: int
    token_type: str = "access"


class RefreshTokenData(BaseModel):
    """Refresh token payload data."""
    user_id: str
    token_type: str = "refresh"
    exp: int
    iat: int


class AuthTokens(BaseModel):
    """Authentication token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class JWTManager:
    """JWT token management."""
    
    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = TokenData(
            user_id=str(user.id),
            email=user.email,
            role=user.role.value,
            tenant_id=str(user.tenant_id),
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            token_type="access"
        )
        
        return jwt.encode(payload.dict(), self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token for user."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = RefreshTokenData(
            user_id=str(user.id),
            token_type="refresh",
            exp=int(expire.timestamp()),
            iat=int(now.timestamp())
        )
        
        return jwt.encode(payload.dict(), self.secret_key, algorithm=self.algorithm)
    
    def create_token_pair(self, user: User) -> AuthTokens:
        """Create access and refresh token pair."""
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return AuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60
        )
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if it's an access token
            if payload.get("token_type") != "access":
                return None
            
            return TokenData(**payload)
        except JWTError:
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[RefreshTokenData]:
        """Verify and decode refresh token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if it's a refresh token
            if payload.get("token_type") != "refresh":
                return None
            
            return RefreshTokenData(**payload)
        except JWTError:
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp")
            if exp:
                return datetime.utcnow().timestamp() > exp
            return True
        except JWTError:
            return True
    
    def get_token_claims(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token claims without verification (for debugging)."""
        try:
            return jwt.get_unverified_claims(token)
        except JWTError:
            return None


class AuthenticationService:
    """Authentication service for user login/logout."""
    
    def __init__(self, jwt_manager: JWTManager):
        self.jwt_manager = jwt_manager
    
    async def authenticate_user(
        self,
        email: str,
        password: str,
        user_repository
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        # Get user by email
        user = await user_repository.get_by_email(email)
        if not user:
            return None
        
        # Check if user is active
        if not user.is_active:
            return None
        
        # Verify password
        if not user.verify_password(password):
            return None
        
        return user
    
    async def login_user(
        self,
        email: str,
        password: str,
        user_repository
    ) -> Optional[AuthTokens]:
        """Login user and return tokens."""
        user = await self.authenticate_user(email, password, user_repository)
        if not user:
            return None
        
        # Update last login
        await user_repository.update_last_login(user.id)
        
        # Create token pair
        return self.jwt_manager.create_token_pair(user)
    
    async def refresh_access_token(
        self,
        refresh_token: str,
        user_repository
    ) -> Optional[AuthTokens]:
        """Refresh access token using refresh token."""
        # Verify refresh token
        token_data = self.jwt_manager.verify_refresh_token(refresh_token)
        if not token_data:
            return None
        
        # Get user
        user = await user_repository.get_by_id(UUID(token_data.user_id))
        if not user or not user.is_active:
            return None
        
        # Create new token pair
        return self.jwt_manager.create_token_pair(user)
    
    def logout_user(self, token: str) -> bool:
        """Logout user (in a real implementation, you'd blacklist the token)."""
        # In a production system, you would:
        # 1. Add token to blacklist in Redis
        # 2. Set expiration time for the blacklist entry
        # For now, we'll just return True
        return True


# Global instances
jwt_manager = JWTManager()
auth_service = AuthenticationService(jwt_manager)