"""
Dependency injection container for the enterprise SaaS platform.
"""
from typing import Any, Dict, Type, TypeVar, Callable, Optional
from functools import lru_cache
import inspect

T = TypeVar('T')


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        key = self._get_key(interface)
        self._singletons[key] = implementation
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service."""
        key = self._get_key(interface)
        self._services[key] = implementation
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function."""
        key = self._get_key(interface)
        self._factories[key] = factory
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        key = self._get_key(interface)
        self._singletons[key] = instance
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        key = self._get_key(interface)
        
        # Check for registered instance first
        if key in self._singletons:
            service = self._singletons[key]
            if not inspect.isclass(service):
                return service  # Already an instance
            
            # Create singleton instance
            instance = self._create_instance(service)
            self._singletons[key] = instance
            return instance
        
        # Check for factory
        if key in self._factories:
            return self._factories[key]()
        
        # Check for transient service
        if key in self._services:
            return self._create_instance(self._services[key])
        
        # Try to create instance directly
        if inspect.isclass(interface):
            return self._create_instance(interface)
        
        raise ValueError(f"Service {interface} not registered")
    
    def _create_instance(self, service_class: Type[T]) -> T:
        """Create service instance with dependency injection."""
        signature = inspect.signature(service_class.__init__)
        kwargs = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency {param.annotation} for {service_class}")
        
        return service_class(**kwargs)
    
    def _get_key(self, interface: Type) -> str:
        """Get string key for interface."""
        return f"{interface.__module__}.{interface.__name__}"
    
    # Convenience methods for common services
    def dataset_repository(self):
        """Get dataset repository."""
        try:
            from ..services.data_service.domain.repositories import DatasetRepository
            return self.resolve(DatasetRepository)
        except (ImportError, ValueError):
            return None
    
    def data_processing_job_repository(self):
        """Get data processing job repository."""
        try:
            from ..services.data_service.domain.repositories import DataProcessingJobRepository
            return self.resolve(DataProcessingJobRepository)
        except (ImportError, ValueError):
            return None
    
    def file_storage_repository(self):
        """Get file storage repository."""
        try:
            from ..services.data_service.domain.repositories import FileStorageRepository
            return self.resolve(FileStorageRepository)
        except (ImportError, ValueError):
            return None


# Global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return container


# Decorator for automatic service registration
def service(interface: Optional[Type] = None, singleton: bool = False):
    """Decorator to automatically register services."""
    def decorator(cls):
        target_interface = interface or cls
        if singleton:
            container.register_singleton(target_interface, cls)
        else:
            container.register_transient(target_interface, cls)
        return cls
    return decorator


def singleton(interface: Optional[Type] = None):
    """Decorator to register singleton services."""
    return service(interface, singleton=True)


# Configuration for dependency injection
class DIConfig:
    """Dependency injection configuration."""
    
    @staticmethod
    def configure_container():
        """Configure the dependency injection container."""
        from .database import DatabaseManager, DatabaseSettings
        from .repositories import SQLAlchemyRepository, MongoRepository, RedisRepository
        
        # Register database components
        container.register_singleton(DatabaseSettings, DatabaseSettings)
        container.register_singleton(DatabaseManager, DatabaseManager)
        
        # Register repository base classes
        container.register_transient(SQLAlchemyRepository, SQLAlchemyRepository)
        container.register_transient(MongoRepository, MongoRepository)
        container.register_transient(RedisRepository, RedisRepository)
        
        # Register data service repositories
        try:
            from ..services.data_service.domain.repositories import (
                DatasetRepository, DataProcessingJobRepository, FileStorageRepository
            )
            from ..services.data_service.infrastructure.repositories import (
                SQLDatasetRepository, InMemoryDataProcessingJobRepository, LocalFileStorageRepository
            )
            
            container.register_transient(DatasetRepository, SQLDatasetRepository)
            container.register_singleton(DataProcessingJobRepository, InMemoryDataProcessingJobRepository)
            container.register_singleton(FileStorageRepository, LocalFileStorageRepository)
        except ImportError:
            # Data service not available
            pass


# Initialize container configuration
@lru_cache()
def get_configured_container() -> Container:
    """Get configured container (cached)."""
    DIConfig.configure_container()
    return container