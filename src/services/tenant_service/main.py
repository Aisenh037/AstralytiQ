"""
Tenant Management Service - Main FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from ...shared.infrastructure.database import db_manager
from ...shared.infrastructure.container import get_configured_container
from .api.tenant_routes import router as tenant_router
from .infrastructure.middleware import TenantContextMiddleware, TenantIsolationMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    container = get_configured_container()
    yield
    # Shutdown
    await db_manager.close_connections()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Tenant Management Service",
        description="Multi-tenant organization and resource management",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware (order matters!)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add tenant context middleware
    app.add_middleware(
        TenantContextMiddleware,
        default_domain="localhost",
        require_tenant=False  # Allow health checks without tenant
    )
    
    # Add tenant isolation middleware
    app.add_middleware(TenantIsolationMiddleware)
    
    # Include routers
    app.include_router(tenant_router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "tenant-service",
            "version": "1.0.0"
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )