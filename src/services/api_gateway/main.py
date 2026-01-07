"""
API Gateway Service - Main FastAPI application.
"""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import uuid
import logging

from ...shared.infrastructure.database import db_manager
from ...shared.infrastructure.container import get_configured_container
from .infrastructure.auth_middleware import AuthenticationMiddleware, AuthorizationMiddleware
from .infrastructure.rate_limiting import RateLimitingMiddleware
from .infrastructure.routing import service_router
from .infrastructure.versioning import version_manager
from .api.gateway_routes import router as gateway_router
from .api.openapi_config import setup_openapi_documentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting API Gateway...")
    container = get_configured_container()
    
    # Initialize services
    logger.info("API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    await service_router.close()
    await db_manager.close_connections()
    logger.info("API Gateway shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Enterprise SaaS Platform - API Gateway",
        description="API Gateway for the Enterprise SaaS Analytics Platform with authentication, rate limiting, and service routing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "authentication",
                "description": "User authentication and authorization endpoints"
            },
            {
                "name": "user-service",
                "description": "User management service endpoints"
            },
            {
                "name": "tenant-service", 
                "description": "Tenant management service endpoints"
            },
            {
                "name": "data-service",
                "description": "Data processing service endpoints"
            },
            {
                "name": "ml-service",
                "description": "ML/Analytics service endpoints"
            },
            {
                "name": "gateway-management",
                "description": "Gateway management and monitoring endpoints"
            },
            {
                "name": "user-context",
                "description": "User context and session information"
            }
        ]
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to each request."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add rate limiting middleware
    app.add_middleware(
        RateLimitingMiddleware,
        redis_url="redis://localhost:6379",
        default_rate_limit=100,
        default_window=3600
    )
    
    # Add authorization middleware
    app.add_middleware(AuthorizationMiddleware)
    
    # Add authentication middleware
    app.add_middleware(
        AuthenticationMiddleware,
        jwt_secret="your-secret-key",  # Should come from environment
        user_service_url="http://localhost:8001"
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with request tracking."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Unhandled exception for request {request_id}: {exc}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                }
            }
        )
    
    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent format."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.detail if isinstance(exc.detail, str) else "HTTP_ERROR",
                    "message": exc.detail,
                    "request_id": request_id
                }
            }
        )
    
    # Include API routes
    app.include_router(gateway_router)
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "api-gateway",
            "version": "1.0.0"
        }
    
    # Root endpoint with API information
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Enterprise SaaS Platform API Gateway",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "supported_versions": [str(v) for v in version_manager.supported_versions],
            "services": {
                "user_management": "/api/v1/users",
                "tenant_management": "/api/v1/tenants",
                "data_processing": "/api/v1/data",
                "ml_analytics": "/api/v1/ml",
                "authentication": "/api/v1/auth"
            }
        }
    
    # API version information endpoint
    @app.get("/api", tags=["api-info"])
    async def api_info():
        """API version and service information."""
        return {
            "api_version": "1.0.0",
            "supported_versions": version_manager.get_version_info(),
            "services": {
                "user": {
                    "description": "User management and authentication",
                    "base_path": "/api/v1/users",
                    "documentation": "/docs#/user-service"
                },
                "tenant": {
                    "description": "Multi-tenant organization management",
                    "base_path": "/api/v1/tenants",
                    "documentation": "/docs#/tenant-service"
                },
                "data": {
                    "description": "Data processing and ETL pipelines",
                    "base_path": "/api/v1/data",
                    "documentation": "/docs#/data-service"
                },
                "ml": {
                    "description": "Machine learning and analytics",
                    "base_path": "/api/v1/ml",
                    "documentation": "/docs#/ml-service"
                }
            },
            "authentication": {
                "login": "/api/v1/auth/login",
                "register": "/api/v1/auth/register",
                "refresh": "/api/v1/auth/refresh"
            }
        }
    
    # Setup enhanced OpenAPI documentation
    setup_openapi_documentation(app)
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )