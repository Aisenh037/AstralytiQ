"""
Tenant context middleware for request routing and isolation.
"""
from typing import Optional, Callable, Dict, Any
from uuid import UUID
import re

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..domain.entities import Tenant
from ..infrastructure.repositories import TenantRepository
from src.shared.infrastructure.database import get_postgres_session


class TenantContext:
    """Tenant context for request processing."""
    
    def __init__(
        self,
        tenant_id: UUID,
        tenant: Optional[Tenant] = None,
        subdomain: Optional[str] = None,
        custom_domain: Optional[str] = None
    ):
        self.tenant_id = tenant_id
        self.tenant = tenant
        self.subdomain = subdomain
        self.custom_domain = custom_domain
    
    @property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.tenant.is_active if self.tenant else False
    
    @property
    def subscription_plan(self) -> str:
        """Get tenant subscription plan."""
        return self.tenant.subscription_plan.value if self.tenant else "free"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": str(self.tenant_id),
            "subdomain": self.subdomain,
            "custom_domain": self.custom_domain,
            "subscription_plan": self.subscription_plan,
            "is_active": self.is_active
        }


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and validate tenant context from requests."""
    
    def __init__(
        self,
        app: ASGIApp,
        default_domain: str = "localhost",
        require_tenant: bool = True
    ):
        super().__init__(app)
        self.default_domain = default_domain
        self.require_tenant = require_tenant
        
        # Paths that don't require tenant context
        self.excluded_paths = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and extract tenant context."""
        
        # Skip tenant resolution for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        try:
            # Extract tenant context
            tenant_context = await self._extract_tenant_context(request)
            
            if tenant_context:
                # Add tenant context to request state
                request.state.tenant_context = tenant_context
                
                # Validate tenant is active
                if not tenant_context.is_active:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": "Tenant account is inactive",
                            "code": "TENANT_INACTIVE"
                        }
                    )
            
            elif self.require_tenant:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Tenant context required",
                        "code": "TENANT_REQUIRED"
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add tenant context to response headers (for debugging)
            if tenant_context:
                response.headers["X-Tenant-ID"] = str(tenant_context.tenant_id)
                response.headers["X-Tenant-Plan"] = tenant_context.subscription_plan
            
            return response
            
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Failed to resolve tenant context",
                    "detail": str(e),
                    "code": "TENANT_RESOLUTION_ERROR"
                }
            )
    
    async def _extract_tenant_context(self, request: Request) -> Optional[TenantContext]:
        """Extract tenant context from request."""
        
        # Method 1: Extract from subdomain
        tenant_context = await self._extract_from_subdomain(request)
        if tenant_context:
            return tenant_context
        
        # Method 2: Extract from custom domain
        tenant_context = await self._extract_from_custom_domain(request)
        if tenant_context:
            return tenant_context
        
        # Method 3: Extract from header (for API clients)
        tenant_context = await self._extract_from_header(request)
        if tenant_context:
            return tenant_context
        
        # Method 4: Extract from JWT token (if authenticated)
        tenant_context = await self._extract_from_token(request)
        if tenant_context:
            return tenant_context
        
        return None
    
    async def _extract_from_subdomain(self, request: Request) -> Optional[TenantContext]:
        """Extract tenant from subdomain (e.g., acme.platform.com)."""
        host = request.headers.get("host", "")
        
        # Parse subdomain
        subdomain = self._parse_subdomain(host)
        if not subdomain:
            return None
        
        # Skip common subdomains
        if subdomain in ["www", "api", "admin", "app"]:
            return None
        
        # Look up tenant by subdomain (stored in domain field)
        tenant = await self._get_tenant_by_domain(f"{subdomain}.{self.default_domain}")
        if tenant:
            return TenantContext(
                tenant_id=tenant.id,
                tenant=tenant,
                subdomain=subdomain
            )
        
        return None
    
    async def _extract_from_custom_domain(self, request: Request) -> Optional[TenantContext]:
        """Extract tenant from custom domain (e.g., analytics.acme.com)."""
        host = request.headers.get("host", "")
        
        # Skip default domains
        if self.default_domain in host or "localhost" in host:
            return None
        
        # Look up tenant by custom domain
        tenant = await self._get_tenant_by_domain(host)
        if tenant:
            return TenantContext(
                tenant_id=tenant.id,
                tenant=tenant,
                custom_domain=host
            )
        
        return None
    
    async def _extract_from_header(self, request: Request) -> Optional[TenantContext]:
        """Extract tenant from X-Tenant-ID header."""
        tenant_id_header = request.headers.get("X-Tenant-ID")
        if not tenant_id_header:
            return None
        
        try:
            tenant_id = UUID(tenant_id_header)
            tenant = await self._get_tenant_by_id(tenant_id)
            if tenant:
                return TenantContext(
                    tenant_id=tenant.id,
                    tenant=tenant
                )
        except ValueError:
            pass  # Invalid UUID format
        
        return None
    
    async def _extract_from_token(self, request: Request) -> Optional[TenantContext]:
        """Extract tenant from JWT token."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        try:
            # Import here to avoid circular imports
            from src.services.user_service.infrastructure.auth import jwt_manager
            
            token = auth_header.split(" ")[1]
            token_data = jwt_manager.verify_token(token)
            
            if token_data and token_data.tenant_id:
                tenant_id = UUID(token_data.tenant_id)
                tenant = await self._get_tenant_by_id(tenant_id)
                if tenant:
                    return TenantContext(
                        tenant_id=tenant.id,
                        tenant=tenant
                    )
        except Exception:
            pass  # Token validation failed
        
        return None
    
    def _parse_subdomain(self, host: str) -> Optional[str]:
        """Parse subdomain from host header."""
        if not host:
            return None
        
        # Remove port if present
        host = host.split(":")[0]
        
        # Split by dots
        parts = host.split(".")
        
        # Need at least 3 parts for subdomain (subdomain.domain.tld)
        if len(parts) < 3:
            return None
        
        # Return first part as subdomain
        subdomain = parts[0]
        
        # Validate subdomain format
        if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$', subdomain):
            return subdomain
        
        return None
    
    async def _get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain from database."""
        try:
            # This is a simplified version - in production you'd use proper DI
            async with get_postgres_session() as session:
                tenant_repo = TenantRepository(session)
                return await tenant_repo.get_by_domain(domain)
        except Exception:
            return None
    
    async def _get_tenant_by_id(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID from database."""
        try:
            async with get_postgres_session() as session:
                tenant_repo = TenantRepository(session)
                return await tenant_repo.get_by_id(tenant_id)
        except Exception:
            return None


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce tenant data isolation."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce tenant isolation for database queries."""
        
        # Get tenant context
        tenant_context = getattr(request.state, "tenant_context", None)
        
        if tenant_context:
            # Add tenant filter to database session
            # This would be implemented with SQLAlchemy filters
            # For now, we'll just add it to request state
            request.state.tenant_id = tenant_context.tenant_id
        
        return await call_next(request)


# Dependency functions for FastAPI
async def get_tenant_context(request: Request) -> Optional[TenantContext]:
    """Get tenant context from request state."""
    return getattr(request.state, "tenant_context", None)


async def require_tenant_context(request: Request) -> TenantContext:
    """Require tenant context (raise exception if not present)."""
    tenant_context = await get_tenant_context(request)
    if not tenant_context:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required"
        )
    return tenant_context


async def get_current_tenant(request: Request) -> Optional[Tenant]:
    """Get current tenant from context."""
    tenant_context = await get_tenant_context(request)
    return tenant_context.tenant if tenant_context else None


async def require_current_tenant(request: Request) -> Tenant:
    """Require current tenant (raise exception if not present)."""
    tenant = await get_current_tenant(request)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant required"
        )
    return tenant