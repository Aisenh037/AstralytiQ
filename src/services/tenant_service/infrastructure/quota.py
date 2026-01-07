"""
Resource quota management and enforcement system.
"""
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum

from fastapi import HTTPException, status
import redis.asyncio as redis

from ..domain.entities import Tenant, TenantQuota
from ..infrastructure.repositories import TenantUsageRepository
from src.shared.infrastructure.repositories import RedisRepository


class QuotaViolationType(str, Enum):
    """Types of quota violations."""
    USER_LIMIT = "user_limit"
    DATASET_LIMIT = "dataset_limit"
    MODEL_LIMIT = "model_limit"
    STORAGE_LIMIT = "storage_limit"
    API_LIMIT = "api_limit"
    CONCURRENT_JOB_LIMIT = "concurrent_job_limit"


class QuotaViolation:
    """Represents a quota violation."""
    
    def __init__(
        self,
        violation_type: QuotaViolationType,
        current_usage: int,
        limit: int,
        message: str
    ):
        self.violation_type = violation_type
        self.current_usage = current_usage
        self.limit = limit
        self.message = message
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.violation_type.value,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class QuotaEnforcer:
    """Enforces resource quotas for tenants."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = RedisRepository(redis_client, "quota")
    
    async def check_quota(
        self,
        tenant: Tenant,
        resource_type: str,
        requested_quantity: int = 1,
        current_usage: Optional[Dict[str, int]] = None
    ) -> Tuple[bool, Optional[QuotaViolation]]:
        """Check if resource creation would violate quota."""
        
        quota = tenant.get_quota()
        
        # Get current usage if not provided
        if current_usage is None:
            current_usage = await self._get_current_usage(tenant.id)
        
        # Check specific resource limits
        if resource_type == "user":
            current = current_usage.get("user", 0)
            if current + requested_quantity > quota.max_users:
                return False, QuotaViolation(
                    QuotaViolationType.USER_LIMIT,
                    current,
                    quota.max_users,
                    f"User limit exceeded: {current + requested_quantity}/{quota.max_users}"
                )
        
        elif resource_type == "dataset":
            current = current_usage.get("dataset", 0)
            if current + requested_quantity > quota.max_datasets:
                return False, QuotaViolation(
                    QuotaViolationType.DATASET_LIMIT,
                    current,
                    quota.max_datasets,
                    f"Dataset limit exceeded: {current + requested_quantity}/{quota.max_datasets}"
                )
        
        elif resource_type == "model":
            current = current_usage.get("model", 0)
            if current + requested_quantity > quota.max_models:
                return False, QuotaViolation(
                    QuotaViolationType.MODEL_LIMIT,
                    current,
                    quota.max_models,
                    f"Model limit exceeded: {current + requested_quantity}/{quota.max_models}"
                )
        
        elif resource_type == "storage":
            current = current_usage.get("storage_gb", 0)
            if current + requested_quantity > quota.max_storage_gb:
                return False, QuotaViolation(
                    QuotaViolationType.STORAGE_LIMIT,
                    current,
                    quota.max_storage_gb,
                    f"Storage limit exceeded: {current + requested_quantity}GB/{quota.max_storage_gb}GB"
                )
        
        elif resource_type == "concurrent_job":
            current = current_usage.get("concurrent_jobs", 0)
            if current + requested_quantity > quota.max_concurrent_jobs:
                return False, QuotaViolation(
                    QuotaViolationType.CONCURRENT_JOB_LIMIT,
                    current,
                    quota.max_concurrent_jobs,
                    f"Concurrent job limit exceeded: {current + requested_quantity}/{quota.max_concurrent_jobs}"
                )
        
        return True, None
    
    async def check_api_rate_limit(
        self,
        tenant_id: UUID,
        quota: TenantQuota,
        requested_calls: int = 1
    ) -> Tuple[bool, Optional[QuotaViolation]]:
        """Check API rate limit for tenant."""
        
        # Get current month's API usage
        current_month = datetime.utcnow().strftime("%Y-%m")
        api_key = f"api_calls:{tenant_id}:{current_month}"
        
        current_calls = await self.redis.get(api_key)
        current_calls = int(current_calls) if current_calls else 0
        
        if current_calls + requested_calls > quota.max_api_calls_per_month:
            return False, QuotaViolation(
                QuotaViolationType.API_LIMIT,
                current_calls,
                quota.max_api_calls_per_month,
                f"API call limit exceeded: {current_calls + requested_calls}/{quota.max_api_calls_per_month}"
            )
        
        return True, None
    
    async def record_api_usage(self, tenant_id: UUID, calls: int = 1) -> None:
        """Record API usage for tenant."""
        current_month = datetime.utcnow().strftime("%Y-%m")
        api_key = f"api_calls:{tenant_id}:{current_month}"
        
        # Increment counter
        await self.redis.increment(api_key, calls)
        
        # Set expiration to end of next month
        next_month = datetime.utcnow().replace(day=1) + timedelta(days=32)
        next_month = next_month.replace(day=1)
        ttl = int((next_month - datetime.utcnow()).total_seconds())
        
        await self.redis.redis.expire(self.redis._make_key(api_key), ttl)
    
    async def get_usage_percentage(
        self,
        tenant: Tenant,
        current_usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """Get usage as percentage of quota for all resources."""
        
        quota = tenant.get_quota()
        
        if current_usage is None:
            current_usage = await self._get_current_usage(tenant.id)
        
        percentages = {}
        
        # Calculate percentages
        if quota.max_users > 0:
            percentages["users"] = min((current_usage.get("user", 0) / quota.max_users) * 100, 100)
        
        if quota.max_datasets > 0:
            percentages["datasets"] = min((current_usage.get("dataset", 0) / quota.max_datasets) * 100, 100)
        
        if quota.max_models > 0:
            percentages["models"] = min((current_usage.get("model", 0) / quota.max_models) * 100, 100)
        
        if quota.max_storage_gb > 0:
            percentages["storage"] = min((current_usage.get("storage_gb", 0) / quota.max_storage_gb) * 100, 100)
        
        if quota.max_concurrent_jobs > 0:
            percentages["concurrent_jobs"] = min((current_usage.get("concurrent_jobs", 0) / quota.max_concurrent_jobs) * 100, 100)
        
        # API calls for current month
        current_month = datetime.utcnow().strftime("%Y-%m")
        api_key = f"api_calls:{tenant.id}:{current_month}"
        api_calls = await self.redis.get(api_key)
        api_calls = int(api_calls) if api_calls else 0
        
        if quota.max_api_calls_per_month > 0:
            percentages["api_calls"] = min((api_calls / quota.max_api_calls_per_month) * 100, 100)
        
        return percentages
    
    async def get_quota_warnings(
        self,
        tenant: Tenant,
        warning_threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """Get quota warnings for resources approaching limits."""
        
        percentages = await self.get_usage_percentage(tenant)
        warnings = []
        
        for resource, percentage in percentages.items():
            if percentage >= warning_threshold:
                warnings.append({
                    "resource": resource,
                    "usage_percentage": round(percentage, 1),
                    "threshold": warning_threshold,
                    "severity": "critical" if percentage >= 95 else "warning"
                })
        
        return warnings
    
    async def _get_current_usage(self, tenant_id: UUID) -> Dict[str, int]:
        """Get current usage from cache or calculate."""
        # This would typically use the TenantUsageRepository
        # For now, return empty dict
        return {}
    
    async def enforce_quota_middleware(
        self,
        tenant: Tenant,
        resource_type: str,
        requested_quantity: int = 1
    ) -> None:
        """Middleware function to enforce quotas (raises HTTPException on violation)."""
        
        can_proceed, violation = await self.check_quota(
            tenant, resource_type, requested_quantity
        )
        
        if not can_proceed and violation:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Quota exceeded",
                    "violation": violation.to_dict(),
                    "upgrade_url": "/billing/upgrade"
                }
            )


class QuotaMonitor:
    """Monitors quota usage and sends alerts."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = RedisRepository(redis_client, "quota_monitor")
        self.enforcer = QuotaEnforcer(redis_client)
    
    async def check_all_tenants(self) -> Dict[str, List[Dict[str, Any]]]:
        """Check quota usage for all tenants."""
        # This would iterate through all active tenants
        # and check their quota usage
        results = {
            "warnings": [],
            "violations": [],
            "healthy": []
        }
        
        # Implementation would go here
        return results
    
    async def send_quota_alerts(self, tenant_id: UUID, warnings: List[Dict[str, Any]]) -> None:
        """Send quota alerts to tenant administrators."""
        # Implementation would send emails/notifications
        pass
    
    async def get_tenant_quota_report(self, tenant: Tenant) -> Dict[str, Any]:
        """Get comprehensive quota report for tenant."""
        
        quota = tenant.get_quota()
        percentages = await self.enforcer.get_usage_percentage(tenant)
        warnings = await self.enforcer.get_quota_warnings(tenant)
        
        return {
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "subscription_plan": tenant.subscription_plan.value,
            "quota_limits": {
                "max_users": quota.max_users,
                "max_datasets": quota.max_datasets,
                "max_models": quota.max_models,
                "max_storage_gb": quota.max_storage_gb,
                "max_api_calls_per_month": quota.max_api_calls_per_month,
                "max_concurrent_jobs": quota.max_concurrent_jobs
            },
            "usage_percentages": percentages,
            "warnings": warnings,
            "status": "healthy" if not warnings else ("critical" if any(w["severity"] == "critical" for w in warnings) else "warning")
        }


# Dependency functions for FastAPI
async def get_quota_enforcer() -> QuotaEnforcer:
    """Get quota enforcer instance."""
    from src.shared.infrastructure.database import get_redis_client
    redis_client = await get_redis_client()
    return QuotaEnforcer(redis_client)


async def get_quota_monitor() -> QuotaMonitor:
    """Get quota monitor instance."""
    from src.shared.infrastructure.database import get_redis_client
    redis_client = await get_redis_client()
    return QuotaMonitor(redis_client)


def require_quota(resource_type: str, quantity: int = 1):
    """Decorator factory for quota enforcement."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be used with FastAPI dependencies
            # to check quotas before executing the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator