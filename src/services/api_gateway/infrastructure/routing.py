"""
API Gateway routing and service discovery.
"""
import httpx
from typing import Dict, Optional, Any
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import asyncio
from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Service registry for microservice discovery."""
    
    def __init__(self):
        self._services = {
            "user": {
                "base_url": "http://localhost:8001",
                "health_endpoint": "/health",
                "prefix": "/api/v1/users"
            },
            "tenant": {
                "base_url": "http://localhost:8002", 
                "health_endpoint": "/health",
                "prefix": "/api/v1/tenants"
            },
            "data": {
                "base_url": "http://localhost:8003",
                "health_endpoint": "/health", 
                "prefix": "/api/v1/data"
            },
            "ml": {
                "base_url": "http://localhost:8004",
                "health_endpoint": "/health",
                "prefix": "/api/v1/ml"
            }
        }
        self._health_status = {}
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get base URL for a service."""
        service = self._services.get(service_name)
        return service["base_url"] if service else None
    
    def get_service_prefix(self, service_name: str) -> Optional[str]:
        """Get API prefix for a service."""
        service = self._services.get(service_name)
        return service["prefix"] if service else None
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services."""
        return self._services.copy()
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        service = self._services.get(service_name)
        if not service:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = urljoin(service["base_url"], service["health_endpoint"])
                response = await client.get(health_url)
                is_healthy = response.status_code == 200
                self._health_status[service_name] = is_healthy
                return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            self._health_status[service_name] = False
            return False
    
    async def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all services."""
        tasks = [
            self.check_service_health(service_name) 
            for service_name in self._services.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return self._health_status.copy()


class ServiceRouter:
    """Routes requests to appropriate microservices."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def route_request(
        self,
        request: Request,
        service_name: str,
        path: str,
        **kwargs
    ) -> Response:
        """Route request to a microservice."""
        
        # Get service URL
        service_url = self.service_registry.get_service_url(service_name)
        if not service_url:
            raise HTTPException(
                status_code=404,
                detail=f"Service '{service_name}' not found"
            )
        
        # Check service health
        is_healthy = await self.service_registry.check_service_health(service_name)
        if not is_healthy:
            raise HTTPException(
                status_code=503,
                detail=f"Service '{service_name}' is unavailable"
            )
        
        # Build target URL
        target_url = urljoin(service_url, path)
        
        # Prepare headers (exclude host and content-length)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        
        # Add gateway headers
        headers["X-Gateway-Request-ID"] = getattr(request.state, "request_id", "unknown")
        headers["X-Forwarded-For"] = request.client.host if request.client else "unknown"
        
        try:
            # Forward request
            if request.method == "GET":
                response = await self.client.get(
                    target_url,
                    headers=headers,
                    params=dict(request.query_params)
                )
            elif request.method == "POST":
                body = await request.body()
                response = await self.client.post(
                    target_url,
                    headers=headers,
                    params=dict(request.query_params),
                    content=body
                )
            elif request.method == "PUT":
                body = await request.body()
                response = await self.client.put(
                    target_url,
                    headers=headers,
                    params=dict(request.query_params),
                    content=body
                )
            elif request.method == "DELETE":
                response = await self.client.delete(
                    target_url,
                    headers=headers,
                    params=dict(request.query_params)
                )
            elif request.method == "PATCH":
                body = await request.body()
                response = await self.client.patch(
                    target_url,
                    headers=headers,
                    params=dict(request.query_params),
                    content=body
                )
            else:
                raise HTTPException(
                    status_code=405,
                    detail=f"Method {request.method} not supported"
                )
            
            # Prepare response headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers["X-Gateway-Service"] = service_name
            
            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
            
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Service '{service_name}' request timeout"
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to service '{service_name}'"
            )
        except Exception as e:
            logger.error(f"Error routing to {service_name}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Bad gateway response from service '{service_name}'"
            )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Global instances
service_registry = ServiceRegistry()
service_router = ServiceRouter(service_registry)