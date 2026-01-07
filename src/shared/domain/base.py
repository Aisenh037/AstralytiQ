"""
Base domain models and entities for the enterprise SaaS platform.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Base entity class with common fields."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ValueObject(BaseModel):
    """Base value object class."""
    
    class Config:
        frozen = True
        from_attributes = True


class AggregateRoot(Entity):
    """Base aggregate root for domain-driven design."""
    
    def __init__(self, **data):
        super().__init__(**data)
        self._domain_events: List[Any] = []
    
    def add_domain_event(self, event: Any) -> None:
        """Add a domain event to be published."""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> List[Any]:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events


class Repository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """Save entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        """List entities with pagination."""
        pass


class DomainService(ABC):
    """Base domain service interface."""
    pass


class ApplicationService(ABC):
    """Base application service interface."""
    pass