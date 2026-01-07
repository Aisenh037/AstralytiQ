"""
Authentication and authorization middleware for API Gateway.
"""
from typing import Optional, List, Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import jwt
import httpx
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle JWT authentication."""
    
    def __init__(
        self,
        app: ASGIApp,
        jwt_secret: str = "your-secret-key",  # Should come from environment
        jwt_algorithm: str = "HS256",
        user_service_url: str = "http://localhost:8001"
    ):
        super().__init__(app)
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.user_service_url = user_service_url
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/auth/forgot-password",
            "/api/v1/auth/reset-password"
        }
        
        # Paths that require authentication
        self.protected_prefixes = [
            "/api/v1/users",
            "/api/v1/tenants", 
            "/api/v1/data",
            "/api/v1/ml"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and validate authentication."""
        
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Check if path requires authentication
        if not self._requires_auth(request.url.path):
            return await call_next(request)
        
        try:
            # Extract and validate JWT token
            token_data = await self._validate_token(request)
            
            if token_data:
                # Add user context to request
                request.state.user_id = token_data.get("user_id")
                request.state.tenant_id = token_data.get("tenant_id")
                request.state.user_role = token_data.get("role")
                request.state.user_permissions = token_data.get("permissions", [])
                
                # Add authentication headers for downstream services
                request.headers.__dict__["_list"].append(
                    (b"x-user-id", str(token_data.get("user_id")).encode())
                )
                request.headers.__dict__["_list"].append(
                    (b"x-tenant-id", str(token_data.get("tenant_id")).encode())
                )
                request.headers.__dict__["_list"].append(
                    (b"x-user-role", str(token_data.get("role")).encode())
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Authentication required",
                        "code": "AUTHENTICATION_REQUIRED"
                    }
                )
            
            return await call_next(request)
            
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "code": "AUTHENTICATION_ERROR"
                }
            )
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Authentication service error",
                    "code": "AUTHENTICATION_SERVICE_ERROR"
                }
            )
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        return path in self.public_paths
    
    def _requires_auth(self, path: str) -> bool:
        """Check if path requires authentication."""
        return any(path.startswith(prefix) for prefix in self.protected_prefixes)
    
    async def _validate_token(self, request: Request) -> Optional[dict]:
        """Validate JWT token from request."""
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            # Validate with user service (optional - for revoked tokens)
            if await self._is_token_valid(token):
                return payload
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token is invalid or revoked"
                )
                
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def _is_token_valid(self, token: str) -> bool:
        """Check token validity with user service."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.user_service_url}/api/v1/auth/validate",
                    headers={"Authorization": f"Bearer {token}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return True  # Fallback to local validation if service is down


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle role-based authorization."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        # Define role-based access rules
        self.role_permissions = {
            "admin": ["*"],  # Admin has access to everything
            "analyst": [
                "users:read",
                "tenants:read", 
                "data:*",
                "ml:*",
                "dashboards:*"
            ],
            "viewer": [
                "users:read",
                "tenants:read",
                "data:read",
                "ml:read", 
                "dashboards:read"
            ]
        }
        
        # Map endpoints to required permissions
        self.endpoint_permissions = {
            "GET:/api/v1/users": "users:read",
            "POST:/api/v1/users": "users:write",
            "PUT:/api/v1/users": "users:write",
            "DELETE:/api/v1/users": "users:delete",
            
            "GET:/api/v1/tenants": "tenants:read",
            "POST:/api/v1/tenants": "tenants:write",
            "PUT:/api/v1/tenants": "tenants:write",
            "DELETE:/api/v1/tenants": "tenants:delete",
            
            "GET:/api/v1/data": "data:read",
            "POST:/api/v1/data": "data:write",
            "PUT:/api/v1/data": "data:write",
            "DELETE:/api/v1/data": "data:delete",
            
            "GET:/api/v1/ml": "ml:read",
            "POST:/api/v1/ml": "ml:write",
            "PUT:/api/v1/ml": "ml:write",
            "DELETE:/api/v1/ml": "ml:delete"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and validate authorization."""
        
        # Skip authorization for public paths
        if not hasattr(request.state, "user_role"):
            return await call_next(request)
        
        try:
            # Check if user has required permissions
            if self._has_permission(request):
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Insufficient permissions",
                        "code": "INSUFFICIENT_PERMISSIONS"
                    }
                )
                
        except Exception as e:
            logger.error(f"Authorization middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Authorization service error",
                    "code": "AUTHORIZATION_SERVICE_ERROR"
                }
            )
    
    def _has_permission(self, request: Request) -> bool:
        """Check if user has permission for the requested endpoint."""
        
        user_role = getattr(request.state, "user_role", None)
        if not user_role:
            return False
        
        # Admin has access to everything
        if user_role == "admin":
            return True
        
        # Get required permission for endpoint
        endpoint_key = f"{request.method}:{request.url.path}"
        required_permission = self.endpoint_permissions.get(endpoint_key)
        
        if not required_permission:
            # If no specific permission defined, allow access
            return True
        
        # Check if user role has the required permission
        user_permissions = self.role_permissions.get(user_role, [])
        
        # Check for wildcard permissions
        for permission in user_permissions:
            if permission == "*":
                return True
            if permission.endswith(":*") and required_permission.startswith(permission[:-1]):
                return True
            if permission == required_permission:
                return True
        
        return False


# Dependency functions for FastAPI
async def get_current_user_id(request: Request) -> Optional[str]:
    """Get current user ID from request state."""
    return getattr(request.state, "user_id", None)


async def get_current_tenant_id(request: Request) -> Optional[str]:
    """Get current tenant ID from request state."""
    return getattr(request.state, "tenant_id", None)


async def get_current_user_role(request: Request) -> Optional[str]:
    """Get current user role from request state."""
    return getattr(request.state, "user_role", None)


async def require_authentication(request: Request) -> str:
    """Require authentication (raise exception if not authenticated)."""
    user_id = await get_current_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user_id


async def require_role(request: Request, required_role: str) -> bool:
    """Require specific role (raise exception if insufficient permissions)."""
    user_role = await get_current_user_role(request)
    if not user_role or user_role != required_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{required_role}' required"
        )
    return True