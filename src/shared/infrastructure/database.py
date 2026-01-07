"""
Database configuration and connection management.
"""
import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_db: str = "enterprise_saas"
    
    # MongoDB settings
    mongodb_host: str = "localhost"
    mongodb_port: int = 27017
    mongodb_user: Optional[str] = None
    mongodb_password: Optional[str] = None
    mongodb_db: str = "enterprise_saas"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL."""
        if self.mongodb_user and self.mongodb_password:
            return (
                f"mongodb://{self.mongodb_user}:{self.mongodb_password}"
                f"@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}"
            )
        return f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_db}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self._postgres_engine = None
        self._postgres_session_factory = None
        self._mongodb_client = None
        self._redis_client = None
    
    def get_postgres_engine(self):
        """Get PostgreSQL engine."""
        if self._postgres_engine is None:
            self._postgres_engine = create_async_engine(
                self.settings.postgres_url,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                echo=False  # Set to True for SQL logging in development
            )
        return self._postgres_engine
    
    def get_postgres_session_factory(self):
        """Get PostgreSQL session factory."""
        if self._postgres_session_factory is None:
            engine = self.get_postgres_engine()
            self._postgres_session_factory = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
        return self._postgres_session_factory
    
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get PostgreSQL session context manager."""
        session_factory = self.get_postgres_session_factory()
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_mongodb_client(self) -> AsyncIOMotorClient:
        """Get MongoDB client."""
        if self._mongodb_client is None:
            self._mongodb_client = AsyncIOMotorClient(self.settings.mongodb_url)
        return self._mongodb_client
    
    def get_mongodb_database(self):
        """Get MongoDB database."""
        client = self.get_mongodb_client()
        return client[self.settings.mongodb_db]
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    async def close_connections(self):
        """Close all database connections."""
        if self._postgres_engine:
            await self._postgres_engine.dispose()
        
        if self._mongodb_client:
            self._mongodb_client.close()
        
        if self._redis_client:
            await self._redis_client.close()


# Global database manager instance
db_settings = DatabaseSettings()
db_manager = DatabaseManager(db_settings)


# Dependency injection functions for FastAPI
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for PostgreSQL session."""
    async with db_manager.get_postgres_session() as session:
        yield session


async def get_mongodb_database():
    """FastAPI dependency for MongoDB database."""
    return db_manager.get_mongodb_database()


async def get_redis_client():
    """FastAPI dependency for Redis client."""
    return await db_manager.get_redis_client()