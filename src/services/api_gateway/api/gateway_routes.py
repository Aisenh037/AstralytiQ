"""
API Gateway routes for service proxying and management.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ..infrastructure.routing import service_registry, service_router
from ..infrastructure.versioning import version_manager, get_api_version, add_version_headers
from ..infrastructure.auth_middleware import (
    get_current_user_id, get_current_tenant_id, get_current_user_role,
    require_authentication
)

router = APIRouter()


# Service proxy routes
@router.api_route(
    "/api/v1/users/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["user-service"]
)
async def proxy_user_service(
    request: Request,
    path: str,
    user_id: str = Depends(require_authentication)
):
    """Proxy requests to User Management Service."""
    full_path = f"/api/v1/users/{path}" if path else "/api/v1/users"
    response = await service_router.route_request(request, "user", full_path)
    
    # Add version headers
    api_version = get_api_version(request)
    add_version_headers(response, api_version)
    
    return response


@router.api_route(
    "/api/v1/tenants/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["tenant-service"]
)
async def proxy_tenant_service(
    request: Request,
    path: str,
    user_id: str = Depends(require_authentication)
):
    """Proxy requests to Tenant Management Service."""
    full_path = f"/api/v1/tenants/{path}" if path else "/api/v1/tenants"
    response = await service_router.route_request(request, "tenant", full_path)
    
    # Add version headers
    api_version = get_api_version(request)
    add_version_headers(response, api_version)
    
    return response


@router.api_route(
    "/api/v1/data/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["data-service"]
)
async def proxy_data_service(
    request: Request,
    path: str,
    user_id: str = Depends(require_authentication)
):
    """Proxy requests to Data Processing Service."""
    full_path = f"/api/v1/data/{path}" if path else "/api/v1/data"
    response = await service_router.route_request(request, "data", full_path)
    
    # Add version headers
    api_version = get_api_version(request)
    add_version_headers(response, api_version)
    
    return response


@router.api_route(
    "/api/v1/ml/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["ml-service"]
)
async def proxy_ml_service(
    request: Request,
    path: str,
    user_id: str = Depends(require_authentication)
):
    """Proxy requests to ML/Analytics Service."""
    full_path = f"/api/v1/ml/{path}" if path else "/api/v1/ml"
    response = await service_router.route_request(request, "ml", full_path)
    
    # Add version headers
    api_version = get_api_version(request)
    add_version_headers(response, api_version)
    
    return response


# Authentication routes (proxy to user service)
@router.post("/api/v1/auth/login", tags=["authentication"])
async def login(request: Request):
    """Proxy login request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/login")
    return response


@router.post("/api/v1/auth/register", tags=["authentication"])
async def register(request: Request):
    """Proxy registration request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/register")
    return response


@router.post("/api/v1/auth/refresh", tags=["authentication"])
async def refresh_token(request: Request):
    """Proxy token refresh request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/refresh")
    return response


@router.post("/api/v1/auth/logout", tags=["authentication"])
async def logout(request: Request, user_id: str = Depends(require_authentication)):
    """Proxy logout request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/logout")
    return response


@router.post("/api/v1/auth/forgot-password", tags=["authentication"])
async def forgot_password(request: Request):
    """Proxy forgot password request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/forgot-password")
    return response


@router.post("/api/v1/auth/reset-password", tags=["authentication"])
async def reset_password(request: Request):
    """Proxy reset password request to User Service."""
    response = await service_router.route_request(request, "user", "/api/v1/auth/reset-password")
    return response


# Gateway management endpoints
@router.get("/api/v1/gateway/services", tags=["gateway-management"])
async def list_services(user_id: str = Depends(require_authentication)) -> Dict[str, Any]:
    """List all registered services and their health status."""
    services = service_registry.list_services()
    health_status = await service_registry.get_health_status()
    
    service_info = {}
    for service_name, config in services.items():
        service_info[service_name] = {
            "base_url": config["base_url"],
            "prefix": config["prefix"],
            "healthy": health_status.get(service_name, False),
            "health_endpoint": config["health_endpoint"]
        }
    
    return {
        "services": service_info,
        "total_services": len(services),
        "healthy_services": sum(1 for status in health_status.values() if status)
    }


@router.get("/api/v1/gateway/health", tags=["gateway-management"])
async def gateway_health_check() -> Dict[str, Any]:
    """Comprehensive health check for the gateway and all services."""
    health_status = await service_registry.get_health_status()
    
    overall_status = "healthy" if all(health_status.values()) else "degraded"
    if not any(health_status.values()):
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "gateway": "healthy",
        "services": health_status,
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }


@router.get("/api/v1/gateway/versions", tags=["gateway-management"])
async def get_api_versions() -> Dict[str, Any]:
    """Get information about supported API versions."""
    return version_manager.get_version_info()


@router.get("/api/v1/gateway/stats", tags=["gateway-management"])
async def get_gateway_stats(
    user_id: str = Depends(require_authentication),
    user_role: str = Depends(get_current_user_role)
) -> Dict[str, Any]:
    """Get gateway statistics (admin only)."""
    if user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # In a real implementation, these would come from metrics collection
    return {
        "total_requests": 0,
        "requests_per_service": {
            "user": 0,
            "tenant": 0,
            "data": 0,
            "ml": 0
        },
        "average_response_time": 0.0,
        "error_rate": 0.0,
        "rate_limit_hits": 0,
        "active_connections": 0
    }


# User context endpoints
@router.get("/api/v1/gateway/me", tags=["user-context"])
async def get_current_user_context(
    request: Request,
    user_id: str = Depends(require_authentication),
    tenant_id: str = Depends(get_current_tenant_id),
    user_role: str = Depends(get_current_user_role)
) -> Dict[str, Any]:
    """Get current user context information."""
    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": user_role,
        "permissions": getattr(request.state, "user_permissions", []),
        "api_version": str(get_api_version(request))
    }


# Error handling endpoints
@router.get("/api/v1/gateway/errors/test", tags=["testing"])
async def test_error_handling():
    """Test endpoint for error handling (development only)."""
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="This is a test error"
    )


# Catch-all route for unmatched API paths
@router.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["fallback"]
)
async def api_fallback(request: Request, path: str):
    """Fallback route for unmatched API paths."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "API endpoint not found",
            "code": "ENDPOINT_NOT_FOUND",
            "path": f"/api/{path}",
            "method": request.method,
            "available_versions": [str(v) for v in version_manager.supported_versions]
        }
    )