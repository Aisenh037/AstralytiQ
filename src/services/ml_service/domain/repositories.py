"""
ML service repository interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from .entities import MLModel, TrainingJob, ModelEvaluation, ModelDeployment, ABTest


class MLModelRepository(ABC):
    """Repository interface for ML models."""
    
    @abstractmethod
    async def create(self, model: MLModel) -> MLModel:
        """Create a new ML model."""
        pass
    
    @abstractmethod
    async def get_by_id(self, model_id: UUID) -> Optional[MLModel]:
        """Get ML model by ID."""
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID) -> List[MLModel]:
        """Get all models for a tenant."""
        pass
    
    @abstractmethod
    async def get_by_name_and_tenant(self, name: str, tenant_id: UUID) -> Optional[MLModel]:
        """Get model by name and tenant."""
        pass
    
    @abstractmethod
    async def update(self, model: MLModel) -> MLModel:
        """Update ML model."""
        pass
    
    @abstractmethod
    async def delete(self, model_id: UUID) -> bool:
        """Delete ML model."""
        pass
    
    @abstractmethod
    async def get_deployed_models(self, tenant_id: UUID) -> List[MLModel]:
        """Get deployed models for a tenant."""
        pass
    
    @abstractmethod
    async def get_models_by_type(self, tenant_id: UUID, model_type: str) -> List[MLModel]:
        """Get models by type for a tenant."""
        pass


class TrainingJobRepository(ABC):
    """Repository interface for training jobs."""
    
    @abstractmethod
    async def create(self, job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        pass
    
    @abstractmethod
    async def get_by_id(self, job_id: UUID) -> Optional[TrainingJob]:
        """Get training job by ID."""
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a tenant."""
        pass
    
    @abstractmethod
    async def update(self, job: TrainingJob) -> TrainingJob:
        """Update training job."""
        pass
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
        """Delete training job."""
        pass
    
    @abstractmethod
    async def get_running_jobs(self, tenant_id: UUID) -> List[TrainingJob]:
        """Get running training jobs for a tenant."""
        pass
    
    @abstractmethod
    async def get_jobs_by_status(self, tenant_id: UUID, status: str) -> List[TrainingJob]:
        """Get training jobs by status."""
        pass


class ModelEvaluationRepository(ABC):
    """Repository interface for model evaluations."""
    
    @abstractmethod
    async def create(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        """Create a new model evaluation."""
        pass
    
    @abstractmethod
    async def get_by_id(self, evaluation_id: UUID) -> Optional[ModelEvaluation]:
        """Get evaluation by ID."""
        pass
    
    @abstractmethod
    async def get_by_model(self, model_id: UUID) -> List[ModelEvaluation]:
        """Get all evaluations for a model."""
        pass
    
    @abstractmethod
    async def get_latest_evaluation(self, model_id: UUID) -> Optional[ModelEvaluation]:
        """Get latest evaluation for a model."""
        pass
    
    @abstractmethod
    async def update(self, evaluation: ModelEvaluation) -> ModelEvaluation:
        """Update model evaluation."""
        pass
    
    @abstractmethod
    async def delete(self, evaluation_id: UUID) -> bool:
        """Delete model evaluation."""
        pass


class ModelDeploymentRepository(ABC):
    """Repository interface for model deployments."""
    
    @abstractmethod
    async def create(self, deployment: ModelDeployment) -> ModelDeployment:
        """Create a new model deployment."""
        pass
    
    @abstractmethod
    async def get_by_id(self, deployment_id: UUID) -> Optional[ModelDeployment]:
        """Get deployment by ID."""
        pass
    
    @abstractmethod
    async def get_by_model(self, model_id: UUID) -> List[ModelDeployment]:
        """Get all deployments for a model."""
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID) -> List[ModelDeployment]:
        """Get all deployments for a tenant."""
        pass
    
    @abstractmethod
    async def get_active_deployments(self, tenant_id: UUID) -> List[ModelDeployment]:
        """Get active deployments for a tenant."""
        pass
    
    @abstractmethod
    async def update(self, deployment: ModelDeployment) -> ModelDeployment:
        """Update model deployment."""
        pass
    
    @abstractmethod
    async def delete(self, deployment_id: UUID) -> bool:
        """Delete model deployment."""
        pass


class ABTestRepository(ABC):
    """Repository interface for A/B tests."""
    
    @abstractmethod
    async def create(self, test: ABTest) -> ABTest:
        """Create a new A/B test."""
        pass
    
    @abstractmethod
    async def get_by_id(self, test_id: UUID) -> Optional[ABTest]:
        """Get A/B test by ID."""
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID) -> List[ABTest]:
        """Get all A/B tests for a tenant."""
        pass
    
    @abstractmethod
    async def get_running_tests(self, tenant_id: UUID) -> List[ABTest]:
        """Get running A/B tests for a tenant."""
        pass
    
    @abstractmethod
    async def update(self, test: ABTest) -> ABTest:
        """Update A/B test."""
        pass
    
    @abstractmethod
    async def delete(self, test_id: UUID) -> bool:
        """Delete A/B test."""
        pass
    
    @abstractmethod
    async def get_tests_by_model(self, model_id: UUID) -> List[ABTest]:
        """Get A/B tests involving a specific model."""
        pass