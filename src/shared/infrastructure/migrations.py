"""
Database migration utilities.
"""
from sqlalchemy.ext.asyncio import create_async_engine
from .database import DatabaseSettings, Base
from .models import *  # Import all models to ensure they're registered


async def create_tables():
    """Create all database tables."""
    settings = DatabaseSettings()
    engine = create_async_engine(settings.postgres_url)
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("Database tables created successfully!")


async def drop_tables():
    """Drop all database tables."""
    settings = DatabaseSettings()
    engine = create_async_engine(settings.postgres_url)
    
    async with engine.begin() as conn:
        # Drop all tables
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()
    print("Database tables dropped successfully!")


if __name__ == "__main__":
    import asyncio
    
    # Create tables
    asyncio.run(create_tables())