"""
Tenant service repository implementations.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..domain.entities import Tenant
from ..domain.repositories import ITenantRepository, ITenantUsageRepository
from src.shared.infrastructure.repositories import SQLAlchemyRepository, RedisRepository
from src.shared.infrastructure.models import TenantModel, UsageMetricModel
from src.shared.domain.models import SubscriptionPlan, TenantSettings


class TenantRepository(SQLAlchemyRepository[Tenant], ITenantRepository):
    """SQLAlchemy implementation of tenant repository."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TenantModel)
    
    async def get_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain."""
        stmt = select(TenantModel).where(TenantModel.domain == domain)
        result = await self.session.execute(stmt)
        tenant_model = result.scalar_one_or_none()
        
        if tenant_model:
            return self._model_to_entity(tenant_model)
        return None
    
    async def get_by_owner_id(self, owner_id: UUID) -> List[Tenant]:
        """Get tenants owned by a user."""
        stmt = select(TenantModel).where(
            TenantModel.owner_id == owner_id,
            TenantModel.is_active == True
        )
        result = await self.session.execute(stmt)
        tenant_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in tenant_models]
    
    async def domain_exists(self, domain: str) -> bool:
        """Check if domain already exists."""
        stmt = select(TenantModel.id).where(TenantModel.domain == domain)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    async def get_active_tenants(self, limit: int = 100, offset: int = 0) -> List[Tenant]:
        """Get active tenants with pagination."""
        stmt = (
            select(TenantModel)
            .where(TenantModel.is_active == True)
            .limit(limit)
            .offset(offset)
            .order_by(TenantModel.created_at.desc())
        )
        result = await self.session.execute(stmt)
        tenant_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in tenant_models]
    
    async def get_by_subscription_plan(self, plan: SubscriptionPlan) -> List[Tenant]:
        """Get tenants by subscription plan."""
        stmt = select(TenantModel).where(
            TenantModel.subscription_plan == plan.value,
            TenantModel.is_active == True
        )
        result = await self.session.execute(stmt)
        tenant_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in tenant_models]
    
    async def search_tenants(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """Search tenants by name or domain."""
        search_term = f"%{query}%"
        stmt = (
            select(TenantModel)
            .where(
                or_(
                    TenantModel.name.ilike(search_term),
                    TenantModel.domain.ilike(search_term)
                ),
                TenantModel.is_active == True
            )
            .limit(limit)
            .offset(offset)
            .order_by(TenantModel.name)
        )
        result = await self.session.execute(stmt)
        tenant_models = result.scalars().all()
        
        return [self._model_to_entity(model) for model in tenant_models]
    
    async def get_tenant_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get tenant usage statistics."""
        # Get user count
        from src.shared.infrastructure.models import UserModel
        user_count_stmt = select(func.count(UserModel.id)).where(
            UserModel.tenant_id == tenant_id,
            UserModel.is_active == True
        )
        user_count_result = await self.session.execute(user_count_stmt)
        user_count = user_count_result.scalar() or 0
        
        # Get dataset count
        from src.shared.infrastructure.models import DatasetModel
        dataset_count_stmt = select(func.count(DatasetModel.id)).where(
            DatasetModel.tenant_id == tenant_id,
            DatasetModel.is_active == True
        )
        dataset_count_result = await self.session.execute(dataset_count_stmt)
        dataset_count = dataset_count_result.scalar() or 0
        
        # Get model count
        from src.shared.infrastructure.models import MLModelModel
        model_count_stmt = select(func.count(MLModelModel.id)).where(
            MLModelModel.tenant_id == tenant_id,
            MLModelModel.is_active == True
        )
        model_count_result = await self.session.execute(model_count_stmt)
        model_count = model_count_result.scalar() or 0
        
        # Get storage usage (sum of dataset sizes)
        storage_stmt = select(func.coalesce(func.sum(DatasetModel.size_bytes), 0)).where(
            DatasetModel.tenant_id == tenant_id,
            DatasetModel.is_active == True
        )
        storage_result = await self.session.execute(storage_stmt)
        storage_bytes = storage_result.scalar() or 0
        storage_gb = storage_bytes / (1024 ** 3)  # Convert to GB
        
        return {
            "user_count": user_count,
            "dataset_count": dataset_count,
            "model_count": model_count,
            "storage_gb": round(storage_gb, 2),
            "storage_bytes": storage_bytes
        }
    
    async def update_last_activity(self, tenant_id: UUID) -> bool:
        """Update tenant's last activity timestamp."""
        stmt = (
            update(TenantModel)
            .where(TenantModel.id == tenant_id)
            .values(updated_at=datetime.utcnow())
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def save(self, entity: Tenant) -> Tenant:
        """Save tenant entity."""
        # Convert entity to SQLAlchemy model
        tenant_model = self._entity_to_model(entity)
        
        # Check if it's an update or insert
        existing = await self.session.get(TenantModel, entity.id)
        if existing:
            # Update existing
            for key, value in tenant_model.__dict__.items():
                if not key.startswith('_'):
                    setattr(existing, key, value)
            tenant_model = existing
        else:
            # Insert new
            self.session.add(tenant_model)
        
        await self.session.flush()
        await self.session.refresh(tenant_model)
        
        return self._model_to_entity(tenant_model)
    
    def _model_to_entity(self, model: TenantModel) -> Tenant:
        """Convert SQLAlchemy model to domain entity."""
        return Tenant(
            id=model.id,
            name=model.name,
            domain=model.domain,
            subscription_plan=SubscriptionPlan(model.subscription_plan),
            settings=TenantSettings(**model.settings) if model.settings else TenantSettings(),
            owner_id=model.owner_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            is_active=model.is_active
        )
    
    def _entity_to_model(self, entity: Tenant) -> TenantModel:
        """Convert domain entity to SQLAlchemy model."""
        return TenantModel(
            id=entity.id,
            name=entity.name,
            domain=entity.domain,
            subscription_plan=entity.subscription_plan.value,
            settings=entity.settings.dict() if entity.settings else {},
            owner_id=entity.owner_id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            is_active=entity.is_active
        )


class TenantUsageRepository(ITenantUsageRepository):
    """Redis and PostgreSQL implementation of tenant usage repository."""
    
    def __init__(self, session: AsyncSession, redis_client: redis.Redis):
        self.session = session
        self.redis = RedisRepository(redis_client, "tenant_usage")
    
    async def record_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        action: str,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record usage event."""
        # Store in PostgreSQL for long-term analytics
        usage_metric = UsageMetricModel(
            tenant_id=tenant_id,
            metric_type=f"{resource_type}_{action}",
            metric_value=quantity,
            usage_metadata=metadata or {}
        )
        
        self.session.add(usage_metric)
        await self.session.flush()
        
        # Update current usage counters in Redis
        current_key = f"current:{tenant_id}:{resource_type}"
        await self.redis.increment(current_key, quantity)
        
        # Set expiration for Redis keys (30 days)
        await self.redis.redis.expire(self.redis._make_key(current_key), 30 * 24 * 3600)
        
        return True
    
    async def get_usage_summary(
        self,
        tenant_id: UUID,
        resource_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage summary for tenant."""
        # Build query
        stmt = select(
            UsageMetricModel.metric_type,
            func.sum(UsageMetricModel.metric_value).label('total'),
            func.count(UsageMetricModel.id).label('events')
        ).where(UsageMetricModel.tenant_id == tenant_id)
        
        if resource_type:
            stmt = stmt.where(UsageMetricModel.metric_type.like(f"{resource_type}_%"))
        
        if start_date:
            stmt = stmt.where(UsageMetricModel.recorded_at >= start_date)
        
        if end_date:
            stmt = stmt.where(UsageMetricModel.recorded_at <= end_date)
        
        stmt = stmt.group_by(UsageMetricModel.metric_type)
        
        result = await self.session.execute(stmt)
        usage_data = result.fetchall()
        
        summary = {}
        for row in usage_data:
            summary[row.metric_type] = {
                "total": row.total,
                "events": row.events
            }
        
        return summary
    
    async def get_current_usage(self, tenant_id: UUID) -> Dict[str, int]:
        """Get current resource usage counts."""
        # Get from Redis first (faster)
        pattern = f"current:{tenant_id}:*"
        keys = await self.redis.redis.keys(self.redis._make_key(pattern))
        
        usage = {}
        for key in keys:
            # Extract resource type from key
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            parts = key_str.split(':')
            if len(parts) >= 3:
                resource_type = parts[-1]  # Last part is resource type
                count = await self.redis.redis.get(key)
                if count:
                    usage[resource_type] = int(count)
        
        # If Redis is empty, calculate from database
        if not usage:
            # Get current counts from database
            from src.shared.infrastructure.models import UserModel, DatasetModel, MLModelModel
            
            # User count
            user_stmt = select(func.count(UserModel.id)).where(
                UserModel.tenant_id == tenant_id,
                UserModel.is_active == True
            )
            user_result = await self.session.execute(user_stmt)
            usage["user"] = user_result.scalar() or 0
            
            # Dataset count
            dataset_stmt = select(func.count(DatasetModel.id)).where(
                DatasetModel.tenant_id == tenant_id,
                DatasetModel.is_active == True
            )
            dataset_result = await self.session.execute(dataset_stmt)
            usage["dataset"] = dataset_result.scalar() or 0
            
            # Model count
            model_stmt = select(func.count(MLModelModel.id)).where(
                MLModelModel.tenant_id == tenant_id,
                MLModelModel.is_active == True
            )
            model_result = await self.session.execute(model_stmt)
            usage["model"] = model_result.scalar() or 0
            
            # Update Redis cache
            for resource_type, count in usage.items():
                current_key = f"current:{tenant_id}:{resource_type}"
                await self.redis.set(current_key, str(count), ttl=3600)  # 1 hour TTL
        
        return usage
    
    async def check_quota_violation(
        self,
        tenant_id: UUID,
        resource_type: str,
        requested_quantity: int = 1
    ) -> tuple[bool, Optional[str]]:
        """Check if action would violate quota."""
        # Get tenant to check quota
        tenant_stmt = select(TenantModel).where(TenantModel.id == tenant_id)
        tenant_result = await self.session.execute(tenant_stmt)
        tenant_model = tenant_result.scalar_one_or_none()
        
        if not tenant_model:
            return False, "Tenant not found"
        
        # Convert to domain entity to get quota
        tenant = Tenant(
            id=tenant_model.id,
            name=tenant_model.name,
            domain=tenant_model.domain,
            subscription_plan=SubscriptionPlan(tenant_model.subscription_plan),
            settings=TenantSettings(**tenant_model.settings) if tenant_model.settings else TenantSettings(),
            owner_id=tenant_model.owner_id,
            created_at=tenant_model.created_at,
            updated_at=tenant_model.updated_at,
            is_active=tenant_model.is_active
        )
        
        quota = tenant.get_quota()
        current_usage = await self.get_current_usage(tenant_id)
        current_count = current_usage.get(resource_type, 0)
        
        # Check limits
        limits = {
            "user": quota.max_users,
            "dataset": quota.max_datasets,
            "model": quota.max_models
        }
        
        limit = limits.get(resource_type)
        if limit is None:
            return True, None  # No limit for this resource type
        
        if current_count + requested_quantity > limit:
            return False, f"{resource_type.title()} limit would be exceeded ({current_count + requested_quantity}/{limit})"
        
        return True, None