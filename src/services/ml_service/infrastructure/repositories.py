"""
ML service repository implementations.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
import json
from datetime import datetime
from abc import ABC, abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..domain.repositories import MLModelRepository, TrainingJobRepository, ModelEvaluationRepository
from ..domain.entities import MLModel, TrainingJob, ModelEvaluation, TrainingStatus
from src.shared.domain.models import ModelStatus


class BaseMLRepository(ABC):
    """Base repository for ML service with raw SQL support."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def execute_query(self, query: str, values: Dict[str, Any] = None):
        """Execute a raw SQL query."""
        result = await self.session.execute(text(query), values or {})
        return result
    
    async def fetch_one(self, query: str, values: Dict[str, Any] = None):
        """Fetch one row from query."""
        result = await self.execute_query(query, values)
        return result.fetchone()
    
    async def fetch_all(self, query: str, values: Dict[str, Any] = None):
        """Fetch all rows from query."""
        result = await self.execute_query(query, values)
        return result.fetchall()


class SQLMLModelRepository(BaseMLRepository, MLModelRepository):
    """SQL implementation of ML model repository."""
    
    async def create(self, model: MLModel) -> MLModel:
        """Create a new ML model."""
        query = """
        INSERT INTO ml_models (
            id, tenant_id, name, ml_model_type, version, parameters, 
            metrics, artifact_path, status, created_at, updated_at, metadata
        ) VALUES (
            :id, :tenant_id, :name, :ml_model_type, :version, :parameters,
            :metrics, :artifact_path, :status, :created_at, :updated_at, :metadata
        )
        """
        
        values = {
            "id": model.id,
            "tenant_id": model.tenant_id,
            "name": model.name,
            "ml_model_type": model.ml_model_type.value,
            "version": model.version,
            "parameters": json.dumps(model.parameters),
            "metrics": json.dumps(model.metrics.dict()) if model.metrics else None,
            "artifact_path": model.artifact_path,
            "status": model.status.value,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "metadata": json.dumps(model.metadata) if model.metadata else None
        }
        
        await self.execute_query(query, values)
        return model
    
    async def get_by_id(self, model_id: UUID) -> Optional[MLModel]:
        """Get ML model by ID."""
        query = """
        SELECT * FROM ml_models WHERE id = :model_id
        """
        
        result = await self.fetch_one(query, {"model_id": model_id})
        return self._map_to_model(result) if result else None
    
    async def get_by_tenant(self, tenant_id: UUID) -> List[MLModel]:
        """Get all models for a tenant."""
        query = """
        SELECT * FROM ml_models 
        WHERE tenant_id = :tenant_id 
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {"tenant_id": tenant_id})
        return [self._map_to_model(row) for row in results]
    
    async def get_by_name_and_tenant(self, name: str, tenant_id: UUID) -> Optional[MLModel]:
        """Get model by name and tenant."""
        query = """
        SELECT * FROM ml_models 
        WHERE name = :name AND tenant_id = :tenant_id
        """
        
        result = await self.fetch_one(query, {"name": name, "tenant_id": tenant_id})
        return self._map_to_model(result) if result else None
    
    async def update(self, model: MLModel) -> MLModel:
        """Update ML model."""
        query = """
        UPDATE ml_models SET
            name = :name,
            ml_model_type = :ml_model_type,
            version = :version,
            parameters = :parameters,
            metrics = :metrics,
            artifact_path = :artifact_path,
            status = :status,
            updated_at = :updated_at,
            metadata = :metadata
        WHERE id = :id
        """
        
        values = {
            "id": model.id,
            "name": model.name,
            "ml_model_type": model.ml_model_type.value,
            "version": model.version,
            "parameters": json.dumps(model.parameters),
            "metrics": json.dumps(model.metrics.dict()) if model.metrics else None,
            "artifact_path": model.artifact_path,
            "status": model.status.value,
            "updated_at": datetime.utcnow(),
            "metadata": json.dumps(model.metadata) if model.metadata else None
        }
        
        await self.execute_query(query, values)
        return model
    
    async def delete(self, model_id: UUID) -> bool:
        """Delete ML model."""
        query = "DELETE FROM ml_models WHERE id = :model_id"
        result = await self.execute_query(query, {"model_id": model_id})
        return result.rowcount > 0
    
    async def get_deployed_models(self, tenant_id: UUID) -> List[MLModel]:
        """Get deployed models for a tenant."""
        query = """
        SELECT * FROM ml_models 
        WHERE tenant_id = :tenant_id AND status = :status
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {
            "tenant_id": tenant_id,
            "status": ModelStatus.DEPLOYED.value
        })
        return [self._map_to_model(row) for row in results]
    
    async def get_models_by_type(self, tenant_id: UUID, model_type: str) -> List[MLModel]:
        """Get models by type for a tenant."""
        query = """
        SELECT * FROM ml_models 
        WHERE tenant_id = :tenant_id AND ml_model_type = :model_type
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {
            "tenant_id": tenant_id,
            "model_type": model_type
        })
        return [self._map_to_model(row) for row in results]
    
    def _map_to_model(self, row: Dict[str, Any]) -> MLModel:
        """Map database row to MLModel entity."""
        from src.shared.domain.models import ModelType, ModelStatus, ModelMetrics
        
        metrics = None
        if row.get("metrics"):
            metrics_data = json.loads(row["metrics"])
            metrics = ModelMetrics(**metrics_data)
        
        return MLModel(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            ml_model_type=ModelType(row["ml_model_type"]),
            version=row["version"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            metrics=metrics,
            artifact_path=row.get("artifact_path"),
            status=ModelStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row.get("metadata") else None
        )


class SQLTrainingJobRepository(BaseMLRepository, TrainingJobRepository):
    """SQL implementation of training job repository."""
    
    async def create(self, job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        query = """
        INSERT INTO training_jobs (
            id, tenant_id, created_by, config, status, started_at, 
            completed_at, error_message, metrics, model_id, logs, created_at
        ) VALUES (
            :id, :tenant_id, :created_by, :config, :status, :started_at,
            :completed_at, :error_message, :metrics, :model_id, :logs, :created_at
        )
        """
        
        values = {
            "id": job.id,
            "tenant_id": job.tenant_id,
            "created_by": job.created_by,
            "config": json.dumps(job.config.dict()),
            "status": job.status.value,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "metrics": json.dumps(job.metrics.dict()) if job.metrics else None,
            "model_id": job.model_id,
            "logs": json.dumps(job.logs),
            "created_at": datetime.utcnow()
        }
        
        await self.execute_query(query, values)
        return job
    
    async def get_by_id(self, job_id: UUID) -> Optional[TrainingJob]:
        """Get training job by ID."""
        query = "SELECT * FROM training_jobs WHERE id = :job_id"
        result = await self.fetch_one(query, {"job_id": job_id})
        return self._map_to_job(result) if result else None
    
    async def get_by_tenant(self, tenant_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a tenant."""
        query = """
        SELECT * FROM training_jobs 
        WHERE tenant_id = :tenant_id 
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {"tenant_id": tenant_id})
        return [self._map_to_job(row) for row in results]
    
    async def update(self, job: TrainingJob) -> TrainingJob:
        """Update training job."""
        query = """
        UPDATE training_jobs SET
            status = :status,
            started_at = :started_at,
            completed_at = :completed_at,
            error_message = :error_message,
            metrics = :metrics,
            model_id = :model_id,
            logs = :logs
        WHERE id = :id
        """
        
        values = {
            "id": job.id,
            "status": job.status.value,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "metrics": json.dumps(job.metrics.dict()) if job.metrics else None,
            "model_id": job.model_id,
            "logs": json.dumps(job.logs)
        }
        
        await self.execute_query(query, values)
        return job
    
    async def delete(self, job_id: UUID) -> bool:
        """Delete training job."""
        query = "DELETE FROM training_jobs WHERE id = :job_id"
        result = await self.execute_query(query, {"job_id": job_id})
        return result.rowcount > 0
    
    async def get_running_jobs(self, tenant_id: UUID) -> List[TrainingJob]:
        """Get running training jobs for a tenant."""
        query = """
        SELECT * FROM training_jobs 
        WHERE tenant_id = :tenant_id AND status = :status
        ORDER BY started_at DESC
        """
        
        results = await self.fetch_all(query, {
            "tenant_id": tenant_id,
            "status": TrainingStatus.RUNNING.value
        })
        return [self._map_to_job(row) for row in results]
    
    async def get_jobs_by_status(self, tenant_id: UUID, status: str) -> List[TrainingJob]:
        """Get training jobs by status."""
        query = """
        SELECT * FROM training_jobs 
        WHERE tenant_id = :tenant_id AND status = :status
        ORDER BY created_at DESC
        """
        
        results = await self.fetch_all(query, {
            "tenant_id": tenant_id,
            "status": status
        })
        return [self._map_to_job(row) for row in results]
    
    def _map_to_job(self, row: Dict[str, Any]) -> TrainingJob:
        """Map database row to TrainingJob entity."""
        from ..domain.entities import TrainingConfig, HyperparameterConfig
        from src.shared.domain.models import ModelType, ModelMetrics
        
        # Parse config
        config_data = json.loads(row["config"])
        hyperparams_data = config_data.get("hyperparameters", {})
        
        hyperparams = HyperparameterConfig(
            parameters=hyperparams_data.get("parameters", {}),
            search_space=hyperparams_data.get("search_space"),
            optimization_method=hyperparams_data.get("optimization_method", "grid_search"),
            max_trials=hyperparams_data.get("max_trials", 10)
        )
        
        config = TrainingConfig(
            dataset_id=UUID(config_data["dataset_id"]),
            target_column=config_data["target_column"],
            feature_columns=config_data["feature_columns"],
            model_type=ModelType(config_data["model_type"]),
            framework=config_data["framework"],
            hyperparameters=hyperparams,
            validation_split=config_data.get("validation_split", 0.2),
            cross_validation_folds=config_data.get("cross_validation_folds", 5),
            random_seed=config_data.get("random_seed", 42)
        )
        
        # Parse metrics
        metrics = None
        if row.get("metrics"):
            metrics_data = json.loads(row["metrics"])
            metrics = ModelMetrics(**metrics_data)
        
        return TrainingJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            created_by=row["created_by"],
            config=config,
            status=TrainingStatus(row["status"]),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            error_message=row.get("error_message"),
            metrics=metrics,
            model_id=row.get("model_id"),
            logs=json.loads(row["logs"]) if row.get("logs") else []
        )


class SQLModelEvaluationRepository(BaseMLRepository, ModelEvaluationRepository):
    """SQL implementation of model evaluation repository."""
    
    async def create(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        """Create a new model evaluation."""
        query = """
        INSERT INTO model_evaluations (
            evaluation_id, model_id, dataset_id, metrics, confusion_matrix,
            feature_importance, predictions_sample, evaluation_date
        ) VALUES (
            :evaluation_id, :model_id, :dataset_id, :metrics, :confusion_matrix,
            :feature_importance, :predictions_sample, :evaluation_date
        )
        """
        
        values = {
            "evaluation_id": evaluation.evaluation_id,
            "model_id": evaluation.model_id,
            "dataset_id": evaluation.dataset_id,
            "metrics": json.dumps(evaluation.metrics.dict()),
            "confusion_matrix": json.dumps(evaluation.confusion_matrix) if evaluation.confusion_matrix else None,
            "feature_importance": json.dumps(evaluation.feature_importance) if evaluation.feature_importance else None,
            "predictions_sample": json.dumps(evaluation.predictions_sample) if evaluation.predictions_sample else None,
            "evaluation_date": evaluation.evaluation_date
        }
        
        await self.execute_query(query, values)
        return evaluation
    
    async def get_by_id(self, evaluation_id: UUID) -> Optional[ModelEvaluation]:
        """Get evaluation by ID."""
        query = "SELECT * FROM model_evaluations WHERE evaluation_id = :evaluation_id"
        result = await self.fetch_one(query, {"evaluation_id": evaluation_id})
        return self._map_to_evaluation(result) if result else None
    
    async def get_by_model(self, model_id: UUID) -> List[ModelEvaluation]:
        """Get all evaluations for a model."""
        query = """
        SELECT * FROM model_evaluations 
        WHERE model_id = :model_id 
        ORDER BY evaluation_date DESC
        """
        
        results = await self.fetch_all(query, {"model_id": model_id})
        return [self._map_to_evaluation(row) for row in results]
    
    async def get_latest_evaluation(self, model_id: UUID) -> Optional[ModelEvaluation]:
        """Get latest evaluation for a model."""
        query = """
        SELECT * FROM model_evaluations 
        WHERE model_id = :model_id 
        ORDER BY evaluation_date DESC 
        LIMIT 1
        """
        
        result = await self.fetch_one(query, {"model_id": model_id})
        return self._map_to_evaluation(result) if result else None
    
    async def update(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        """Update model evaluation."""
        query = """
        UPDATE model_evaluations SET
            metrics = :metrics,
            confusion_matrix = :confusion_matrix,
            feature_importance = :feature_importance,
            predictions_sample = :predictions_sample,
            evaluation_date = :evaluation_date
        WHERE evaluation_id = :evaluation_id
        """
        
        values = {
            "evaluation_id": evaluation.evaluation_id,
            "metrics": json.dumps(evaluation.metrics.dict()),
            "confusion_matrix": json.dumps(evaluation.confusion_matrix) if evaluation.confusion_matrix else None,
            "feature_importance": json.dumps(evaluation.feature_importance) if evaluation.feature_importance else None,
            "predictions_sample": json.dumps(evaluation.predictions_sample) if evaluation.predictions_sample else None,
            "evaluation_date": evaluation.evaluation_date
        }
        
        await self.execute_query(query, values)
        return evaluation
    
    async def delete(self, evaluation_id: UUID) -> bool:
        """Delete model evaluation."""
        query = "DELETE FROM model_evaluations WHERE evaluation_id = :evaluation_id"
        result = await self.execute_query(query, {"evaluation_id": evaluation_id})
        return result.rowcount > 0
    
    def _map_to_evaluation(self, row: Dict[str, Any]) -> ModelEvaluation:
        """Map database row to ModelEvaluation entity."""
        from src.shared.domain.models import ModelMetrics
        
        metrics_data = json.loads(row["metrics"])
        metrics = ModelMetrics(**metrics_data)
        
        return ModelEvaluation(
            model_id=row["model_id"],
            evaluation_id=row["evaluation_id"],
            dataset_id=row["dataset_id"],
            metrics=metrics,
            confusion_matrix=json.loads(row["confusion_matrix"]) if row.get("confusion_matrix") else None,
            feature_importance=json.loads(row["feature_importance"]) if row.get("feature_importance") else None,
            predictions_sample=json.loads(row["predictions_sample"]) if row.get("predictions_sample") else None,
            evaluation_date=row["evaluation_date"]
        )