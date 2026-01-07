"""
Tenant service repository interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from .entities import Tenant
from src.shared.domain.base import Repository
from src.shared.domain.models import SubscriptionPlan


class ITenantRepository(Repository):
    """Tenant repository interface."""
    
    @abstractmethod
    async def get_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        pass
    
    @abstractmethod
    async def get_by_owner_id(self, owner_id: UUID) -> List[Tenant]:
        """Get tenants owned by a user."""
        pass
    
    @abstractmethod
    async def domain_exists(self, domain: str) -> bool:
        """Check if domain already exists."""
        pass
    
    @abstractmethod
    async def get_active_tenants(self, limit: int = 100, offset: int = 0) -> List[Tenant]:
        """Get active tenants with pagination."""
        pass
    
    @abstractmethod
    async def get_by_subscription_plan(self, plan: SubscriptionPlan) -> List[Tenant]:
        """Get tenants by subscription plan."""
        pass
    
    @abstractmethod
    async def search_tenants(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """Search tenants by name or domain."""
        pass
    
    @abstractmethod
    async def get_tenant_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get tenant usage statistics."""
        pass
    
    @abstractmethod
    async def update_last_activity(self, tenant_id: UUID) -> bool:
        """Update tenant's last activity timestamp."""
        pass


class ITenantUsageRepository(ABC):
    """Tenant usage tracking repository interface."""
    
    @abstractmethod
    async def record_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        action: str,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record usage event."""
        pass
    
    @abstractmethod
    async def get_usage_summary(
        self,
        tenant_id: UUID,
        resource_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage summary for tenant."""
        pass
    
    @abstractmethod
    async def get_current_usage(self, tenant_id: UUID) -> Dict[str, int]:
        """Get current resource usage counts."""
        pass
    
    @abstractmethod
    async def check_quota_violation(
        self,
        tenant_id: UUID,
        resource_type: str,
        requested_quantity: int = 1
    ) -> tuple[bool, Optional[str]]:
        """Check if action would violate quota."""
        pass