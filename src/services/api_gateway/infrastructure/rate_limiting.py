"""
Rate limiting middleware for API Gateway.
"""
import time
import asyncio
from typing import Dict, Optional, Callable, Tuple
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis-based rate limiter with sliding window algorithm."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed based on rate limit.
        Returns (is_allowed, rate_limit_info)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]
        
        # Calculate rate limit info
        rate_limit_info = {
            "limit": limit,
            "remaining": max(0, limit - current_count - 1),
            "reset_time": int(now + window_seconds),
            "retry_after": window_seconds if current_count >= limit else 0
        }
        
        is_allowed = current_count < limit
        
        if not is_allowed:
            # Remove the request we just added since it's not allowed
            await self.redis.zrem(key, str(now))
        
        return is_allowed, rate_limit_info


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware to implement rate limiting."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_url: str = "redis://localhost:6379",
        default_rate_limit: int = 100,
        default_window: int = 3600  # 1 hour
    ):
        super().__init__(app)
        self.redis_url = redis_url
        self.default_rate_limit = default_rate_limit
        self.default_window = default_window
        self.redis_client = None
        self.rate_limiter = None
        
        # Rate limit rules by endpoint pattern
        self.endpoint_limits = {
            # Authentication endpoints - more restrictive
            "/api/v1/auth/login": {"limit": 10, "window": 900},  # 10 per 15 min
            "/api/v1/auth/register": {"limit": 5, "window": 3600},  # 5 per hour
            "/api/v1/auth/forgot-password": {"limit": 3, "window": 3600},  # 3 per hour
            
            # Data upload endpoints - moderate limits
            "/api/v1/data/upload": {"limit": 20, "window": 3600},  # 20 per hour
            "/api/v1/ml/train": {"limit": 10, "window": 3600},  # 10 per hour
            
            # General API endpoints
            "/api/v1/users": {"limit": 200, "window": 3600},  # 200 per hour
            "/api/v1/tenants": {"limit": 100, "window": 3600},  # 100 per hour
            "/api/v1/data": {"limit": 500, "window": 3600},  # 500 per hour
            "/api/v1/ml": {"limit": 100, "window": 3600},  # 100 per hour
        }
        
        # Role-based rate limits
        self.role_multipliers = {
            "admin": 5.0,      # 5x the base limit
            "analyst": 2.0,    # 2x the base limit  
            "viewer": 1.0,     # 1x the base limit
            "free": 0.5        # 0.5x the base limit
        }
        
        # Paths exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and apply rate limiting."""
        
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Initialize Redis client if needed
        if not self.redis_client:
            await self._init_redis()
        
        try:
            # Generate rate limit key
            rate_limit_key = self._generate_key(request)
            
            # Get rate limit for endpoint
            limit, window = self._get_rate_limit(request)
            
            # Check rate limit
            is_allowed, rate_info = await self.rate_limiter.is_allowed(
                rate_limit_key, limit, window
            )
            
            if not is_allowed:
                # Rate limit exceeded
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "limit": rate_info["limit"],
                        "retry_after": rate_info["retry_after"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": str(rate_info["remaining"]),
                        "X-RateLimit-Reset": str(rate_info["reset_time"]),
                        "Retry-After": str(rate_info["retry_after"])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Continue without rate limiting if there's an error
            return await call_next(request)
    
    async def _init_redis(self):
        """Initialize Redis client and rate limiter."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.rate_limiter = RateLimiter(self.redis_client)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Rate limiting Redis client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis for rate limiting: {e}")
            # Create a no-op rate limiter
            self.rate_limiter = NoOpRateLimiter()
    
    def _generate_key(self, request: Request) -> str:
        """Generate rate limit key for request."""
        
        # Use user ID if authenticated, otherwise use IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            identifier = f"user:{user_id}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            identifier = f"ip:{client_ip}"
        
        # Include endpoint in key for endpoint-specific limits
        endpoint = self._normalize_endpoint(request.url.path)
        
        return f"rate_limit:{identifier}:{endpoint}"
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for rate limiting."""
        
        # Remove trailing slashes
        path = path.rstrip("/")
        
        # Replace path parameters with placeholders
        parts = path.split("/")
        normalized_parts = []
        
        for part in parts:
            # Replace UUIDs and numeric IDs with placeholder
            if self._is_uuid_or_id(part):
                normalized_parts.append("{id}")
            else:
                normalized_parts.append(part)
        
        return "/".join(normalized_parts)
    
    def _is_uuid_or_id(self, part: str) -> bool:
        """Check if path part is a UUID or numeric ID."""
        if not part:
            return False
        
        # Check for UUID pattern
        if len(part) == 36 and part.count("-") == 4:
            return True
        
        # Check for numeric ID
        if part.isdigit():
            return True
        
        return False
    
    def _get_rate_limit(self, request: Request) -> Tuple[int, int]:
        """Get rate limit and window for request."""
        
        endpoint = self._normalize_endpoint(request.url.path)
        
        # Get base limit for endpoint
        endpoint_config = None
        for pattern, config in self.endpoint_limits.items():
            if endpoint.startswith(pattern):
                endpoint_config = config
                break
        
        if endpoint_config:
            base_limit = endpoint_config["limit"]
            window = endpoint_config["window"]
        else:
            base_limit = self.default_rate_limit
            window = self.default_window
        
        # Apply role-based multiplier
        user_role = getattr(request.state, "user_role", "free")
        multiplier = self.role_multipliers.get(user_role, 1.0)
        
        final_limit = int(base_limit * multiplier)
        
        return final_limit, window


class NoOpRateLimiter:
    """No-op rate limiter for fallback when Redis is unavailable."""
    
    async def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict[str, int]]:
        """Always allow requests."""
        return True, {
            "limit": limit,
            "remaining": limit,
            "reset_time": int(time.time() + window_seconds),
            "retry_after": 0
        }


class RateLimitConfig:
    """Configuration for rate limiting."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        burst_limit: Optional[int] = None
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit or limit * 2


# Utility functions
async def get_rate_limit_info(request: Request) -> Dict[str, int]:
    """Get current rate limit information for request."""
    # This would be populated by the middleware
    return {
        "limit": getattr(request.state, "rate_limit", 0),
        "remaining": getattr(request.state, "rate_remaining", 0),
        "reset_time": getattr(request.state, "rate_reset", 0)
    }


def create_rate_limit_response(rate_info: Dict[str, int]) -> JSONResponse:
    """Create standardized rate limit exceeded response."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "code": "RATE_LIMIT_EXCEEDED",
            "limit": rate_info["limit"],
            "retry_after": rate_info["retry_after"]
        },
        headers={
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset_time"]),
            "Retry-After": str(rate_info["retry_after"])
        }
    )