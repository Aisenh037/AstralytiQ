"""
Base repository implementations for different database types.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from motor.motor_asyncio import AsyncIOMotorDatabase
import redis.asyncio as redis

from ..domain.base import Entity, Repository

T = TypeVar('T', bound=Entity)


class SQLAlchemyRepository(Repository, Generic[T]):
    """Base SQLAlchemy repository implementation."""
    
    def __init__(self, session: AsyncSession, model_class: Type[T]):
        self.session = session
        self.model_class = model_class
    
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID."""
        stmt = select(self.model_class).where(self.model_class.id == entity_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def save(self, entity: T) -> T:
        """Save entity."""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        stmt = delete(self.model_class).where(self.model_class.id == entity_id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List entities with pagination."""
        stmt = (
            select(self.model_class)
            .where(self.model_class.is_active == True)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def find_by_field(self, field_name: str, value: Any) -> List[T]:
        """Find entities by field value."""
        field = getattr(self.model_class, field_name)
        stmt = select(self.model_class).where(field == value)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def update_by_id(self, entity_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update entity by ID."""
        stmt = (
            update(self.model_class)
            .where(self.model_class.id == entity_id)
            .values(**updates)
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0


class MongoRepository(Repository, Generic[T]):
    """Base MongoDB repository implementation."""
    
    def __init__(self, database: AsyncIOMotorDatabase, collection_name: str, model_class: Type[T]):
        self.database = database
        self.collection = database[collection_name]
        self.model_class = model_class
    
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID."""
        doc = await self.collection.find_one({"_id": str(entity_id)})
        if doc:
            doc["id"] = doc.pop("_id")
            return self.model_class(**doc)
        return None
    
    async def save(self, entity: T) -> T:
        """Save entity."""
        doc = entity.dict()
        doc["_id"] = str(doc.pop("id"))
        
        await self.collection.replace_one(
            {"_id": doc["_id"]}, 
            doc, 
            upsert=True
        )
        return entity
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        result = await self.collection.delete_one({"_id": str(entity_id)})
        return result.deleted_count > 0
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List entities with pagination."""
        cursor = self.collection.find({"is_active": True}).skip(offset).limit(limit)
        entities = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            entities.append(self.model_class(**doc))
        return entities
    
    async def find_by_field(self, field_name: str, value: Any) -> List[T]:
        """Find entities by field value."""
        cursor = self.collection.find({field_name: value})
        entities = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            entities.append(self.model_class(**doc))
        return entities


class RedisRepository:
    """Redis-based repository for caching and session management."""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = ""):
        self.redis = redis_client
        self.key_prefix = key_prefix
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}:{key}" if self.key_prefix else key
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self.redis.get(self._make_key(key))
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        return await self.redis.set(self._make_key(key), value, ex=ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        result = await self.redis.delete(self._make_key(key))
        return result > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.redis.exists(self._make_key(key)) > 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        return await self.redis.incrby(self._make_key(key), amount)
    
    async def set_hash(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set hash fields."""
        return await self.redis.hset(self._make_key(key), mapping=mapping)
    
    async def get_hash(self, key: str) -> Dict[str, str]:
        """Get all hash fields."""
        return await self.redis.hgetall(self._make_key(key))
    
    async def add_to_set(self, key: str, *values: str) -> int:
        """Add values to set."""
        return await self.redis.sadd(self._make_key(key), *values)
    
    async def get_set_members(self, key: str) -> List[str]:
        """Get all set members."""
        members = await self.redis.smembers(self._make_key(key))
        return list(members)


class UnitOfWork:
    """Unit of Work pattern implementation."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._repositories: Dict[str, Any] = {}
    
    def get_repository(self, name: str, repository_class: Type, *args, **kwargs):
        """Get or create repository instance."""
        if name not in self._repositories:
            self._repositories[name] = repository_class(self.session, *args, **kwargs)
        return self._repositories[name]
    
    async def commit(self):
        """Commit transaction."""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback transaction."""
        await self.session.rollback()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()