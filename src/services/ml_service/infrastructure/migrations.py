"""
ML service database migrations.
"""
from src.shared.infrastructure.database import db_manager


async def create_ml_tables():
    """Create ML service database tables."""
    
    # ML Models table
    ml_models_table = """
    CREATE TABLE IF NOT EXISTS ml_models (
        id UUID PRIMARY KEY,
        tenant_id UUID NOT NULL,
        name VARCHAR(255) NOT NULL,
        ml_model_type VARCHAR(50) NOT NULL,
        version VARCHAR(50) NOT NULL,
        parameters JSONB DEFAULT '{}',
        metrics JSONB,
        artifact_path TEXT,
        status VARCHAR(50) NOT NULL DEFAULT 'trained',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        metadata JSONB,
        UNIQUE(name, tenant_id, version)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ml_models_tenant_id ON ml_models(tenant_id);
    CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
    CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(ml_model_type);
    CREATE INDEX IF NOT EXISTS idx_ml_models_created_at ON ml_models(created_at);
    """
    
    # Training Jobs table
    training_jobs_table = """
    CREATE TABLE IF NOT EXISTS training_jobs (
        id UUID PRIMARY KEY,
        tenant_id UUID NOT NULL,
        created_by UUID NOT NULL,
        config JSONB NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'pending',
        started_at TIMESTAMP WITH TIME ZONE,
        completed_at TIMESTAMP WITH TIME ZONE,
        error_message TEXT,
        metrics JSONB,
        model_id UUID,
        logs JSONB DEFAULT '[]',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        FOREIGN KEY (model_id) REFERENCES ml_models(id) ON DELETE SET NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_training_jobs_tenant_id ON training_jobs(tenant_id);
    CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
    CREATE INDEX IF NOT EXISTS idx_training_jobs_created_by ON training_jobs(created_by);
    CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at);
    """
    
    # Model Evaluations table
    model_evaluations_table = """
    CREATE TABLE IF NOT EXISTS model_evaluations (
        evaluation_id UUID PRIMARY KEY,
        model_id UUID NOT NULL,
        dataset_id UUID NOT NULL,
        metrics JSONB NOT NULL,
        confusion_matrix JSONB,
        feature_importance JSONB,
        predictions_sample JSONB,
        evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        FOREIGN KEY (model_id) REFERENCES ml_models(id) ON DELETE CASCADE
    );
    
    CREATE INDEX IF NOT EXISTS idx_model_evaluations_model_id ON model_evaluations(model_id);
    CREATE INDEX IF NOT EXISTS idx_model_evaluations_dataset_id ON model_evaluations(dataset_id);
    CREATE INDEX IF NOT EXISTS idx_model_evaluations_date ON model_evaluations(evaluation_date);
    """
    
    # Execute migrations
    async with db_manager.get_connection() as conn:
        await conn.execute(ml_models_table)
        await conn.execute(training_jobs_table)
        await conn.execute(model_evaluations_table)
        await conn.commit()
    
    print("✅ ML service database tables created successfully")


async def drop_ml_tables():
    """Drop ML service database tables (for testing)."""
    
    drop_tables = """
    DROP TABLE IF EXISTS model_evaluations CASCADE;
    DROP TABLE IF EXISTS training_jobs CASCADE;
    DROP TABLE IF EXISTS ml_models CASCADE;
    """
    
    async with db_manager.get_connection() as conn:
        await conn.execute(drop_tables)
        await conn.commit()
    
    print("✅ ML service database tables dropped successfully")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        await create_ml_tables()
    
    asyncio.run(main())