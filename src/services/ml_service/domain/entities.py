"""
ML service domain entities and business logic.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from enum import Enum
import json

from src.shared.domain.models import MLModel as BaseMLModel, ModelType, ModelStatus, ModelMetrics
from src.shared.domain.base import DomainService, ValueObject


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelFramework(str, Enum):
    """Supported ML frameworks."""
    SCIKIT_LEARN = "scikit_learn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class HyperparameterConfig(ValueObject):
    """Hyperparameter configuration for model training."""
    parameters: Dict[str, Any]
    search_space: Optional[Dict[str, Any]] = None
    optimization_method: str = "grid_search"  # grid_search, random_search, bayesian
    max_trials: int = 10
    
    def get_parameter_value(self, param_name: str) -> Any:
        """Get parameter value."""
        return self.parameters.get(param_name)
    
    def update_parameter(self, param_name: str, value: Any) -> None:
        """Update parameter value."""
        self.parameters[param_name] = value


class TrainingConfig(ValueObject):
    """Training configuration."""
    dataset_id: UUID
    target_column: str
    feature_columns: List[str]
    model_type: ModelType
    framework: ModelFramework
    hyperparameters: HyperparameterConfig
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_seed: int = 42
    
    def validate_config(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        if not self.target_column:
            errors.append("Target column is required")
        
        if not self.feature_columns:
            errors.append("At least one feature column is required")
        
        if self.validation_split <= 0 or self.validation_split >= 1:
            errors.append("Validation split must be between 0 and 1")
        
        if self.cross_validation_folds < 2:
            errors.append("Cross validation folds must be at least 2")
        
        return errors


class TrainingJob(ValueObject):
    """Training job entity."""
    id: UUID
    tenant_id: UUID
    created_by: UUID
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    model_id: Optional[UUID] = None
    logs: List[str] = []
    
    def start_training(self) -> None:
        """Mark training as started."""
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.add_log("Training started")
    
    def complete_training(self, model_id: UUID, metrics: ModelMetrics) -> None:
        """Mark training as completed."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.model_id = model_id
        self.metrics = metrics
        self.add_log("Training completed successfully")
    
    def fail_training(self, error_message: str) -> None:
        """Mark training as failed."""
        self.status = TrainingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.add_log(f"Training failed: {error_message}")
    
    def cancel_training(self) -> None:
        """Cancel training job."""
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.add_log("Training cancelled")
    
    def add_log(self, message: str) -> None:
        """Add log message."""
        timestamp = datetime.utcnow().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def get_duration(self) -> Optional[float]:
        """Get training duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class MLModel(BaseMLModel):
    """Extended ML Model entity with business logic."""
    
    def update_metrics(self, metrics: ModelMetrics) -> None:
        """Update model metrics."""
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def deploy(self) -> None:
        """Deploy model."""
        self.status = ModelStatus.DEPLOYED
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive model."""
        self.status = ModelStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    def get_performance_score(self) -> Optional[float]:
        """Get primary performance score based on model type."""
        if not self.metrics:
            return None
        
        # Return appropriate metric based on model type
        if self.ml_model_type in [ModelType.LINEAR_REGRESSION]:
            return self.metrics.r2_score
        elif self.ml_model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST]:
            return self.metrics.accuracy or self.metrics.f1_score
        else:
            return self.metrics.accuracy
    
    def is_better_than(self, other_model: "MLModel") -> bool:
        """Compare model performance with another model."""
        self_score = self.get_performance_score()
        other_score = other_model.get_performance_score()
        
        if self_score is None or other_score is None:
            return False
        
        return self_score > other_score
    
    @classmethod
    def create_new_model(
        cls,
        name: str,
        tenant_id: UUID,
        model_type: ModelType,
        version: str,
        parameters: Dict[str, Any],
        training_job_id: UUID
    ) -> "MLModel":
        """Create a new ML model."""
        return cls(
            name=name,
            tenant_id=tenant_id,
            ml_model_type=model_type,
            version=version,
            parameters=parameters,
            status=ModelStatus.TRAINED,
            metadata={
                "training_job_id": str(training_job_id),
                "created_timestamp": datetime.utcnow().isoformat()
            }
        )


class ModelEvaluation(ValueObject):
    """Model evaluation results."""
    model_id: UUID
    evaluation_id: UUID
    dataset_id: UUID
    metrics: ModelMetrics
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    predictions_sample: Optional[List[Dict[str, Any]]] = None
    evaluation_date: datetime
    
    def get_classification_report(self) -> Dict[str, Any]:
        """Get classification report."""
        if not self.metrics:
            return {}
        
        return {
            "accuracy": self.metrics.accuracy,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1_score": self.metrics.f1_score
        }
    
    def get_regression_report(self) -> Dict[str, Any]:
        """Get regression report."""
        if not self.metrics:
            return {}
        
        return {
            "mae": self.metrics.mae,
            "mse": self.metrics.mse,
            "r2_score": self.metrics.r2_score
        }


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class ABTestStatus(str, Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentConfig(ValueObject):
    """Model deployment configuration."""
    model_id: UUID
    deployment_name: str
    environment: str = "production"  # staging, production
    replicas: int = 1
    cpu_limit: str = "500m"
    memory_limit: str = "1Gi"
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    health_check_path: str = "/health"
    rollback_on_failure: bool = True
    
    def validate_config(self) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if self.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        if self.auto_scaling and self.min_replicas >= self.max_replicas:
            errors.append("Min replicas must be less than max replicas for auto-scaling")
        
        if not self.deployment_name:
            errors.append("Deployment name is required")
        
        return errors


class ABTestConfig(ValueObject):
    """A/B test configuration."""
    test_name: str
    control_model_id: UUID
    treatment_model_id: UUID
    traffic_split: float = 0.5  # 0.0 to 1.0, percentage for treatment
    duration_hours: int = 24
    success_metric: str = "accuracy"  # accuracy, precision, recall, f1_score
    minimum_sample_size: int = 100
    confidence_level: float = 0.95
    auto_promote_winner: bool = False
    
    def validate_config(self) -> List[str]:
        """Validate A/B test configuration."""
        errors = []
        
        if not 0.0 <= self.traffic_split <= 1.0:
            errors.append("Traffic split must be between 0.0 and 1.0")
        
        if self.duration_hours < 1:
            errors.append("Duration must be at least 1 hour")
        
        if self.minimum_sample_size < 10:
            errors.append("Minimum sample size must be at least 10")
        
        if not 0.5 <= self.confidence_level <= 0.99:
            errors.append("Confidence level must be between 0.5 and 0.99")
        
        if not self.test_name:
            errors.append("Test name is required")
        
        return errors


class ModelDeployment(ValueObject):
    """Model deployment entity."""
    id: UUID
    tenant_id: UUID
    model_id: UUID
    config: DeploymentConfig
    status: DeploymentStatus = DeploymentStatus.PENDING
    deployment_url: Optional[str] = None
    created_at: datetime
    deployed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    version: str = "1.0.0"
    rollback_version: Optional[str] = None
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    
    def start_deployment(self) -> None:
        """Start deployment process."""
        self.status = DeploymentStatus.DEPLOYING
        self.deployed_at = datetime.utcnow()
    
    def complete_deployment(self, deployment_url: str) -> None:
        """Complete deployment successfully."""
        self.status = DeploymentStatus.DEPLOYED
        self.deployment_url = deployment_url
        self.health_status = "healthy"
    
    def fail_deployment(self, error_message: str) -> None:
        """Mark deployment as failed."""
        self.status = DeploymentStatus.FAILED
        self.failed_at = datetime.utcnow()
        self.error_message = error_message
        self.health_status = "unhealthy"
    
    def start_rollback(self, rollback_version: str) -> None:
        """Start rollback process."""
        self.status = DeploymentStatus.ROLLING_BACK
        self.rollback_version = rollback_version
    
    def complete_rollback(self) -> None:
        """Complete rollback successfully."""
        self.status = DeploymentStatus.ROLLED_BACK
        self.version = self.rollback_version or "previous"
        self.health_status = "healthy"
    
    def update_health_status(self, status: str) -> None:
        """Update health status."""
        self.health_status = status
    
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return self.status == DeploymentStatus.DEPLOYED and self.health_status == "healthy"
    
    @classmethod
    def create_new_deployment(
        cls,
        tenant_id: UUID,
        model_id: UUID,
        config: DeploymentConfig
    ) -> "ModelDeployment":
        """Create a new model deployment."""
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            model_id=model_id,
            config=config,
            created_at=datetime.utcnow()
        )


class ABTest(ValueObject):
    """A/B test entity."""
    id: UUID
    tenant_id: UUID
    config: ABTestConfig
    status: ABTestStatus = ABTestStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    control_deployment_id: UUID
    treatment_deployment_id: UUID
    control_requests: int = 0
    treatment_requests: int = 0
    control_successes: int = 0
    treatment_successes: int = 0
    winner_model_id: Optional[UUID] = None
    confidence_score: Optional[float] = None
    statistical_significance: Optional[bool] = None
    results: Optional[Dict[str, Any]] = None
    
    def start_test(self) -> None:
        """Start A/B test."""
        self.status = ABTestStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def record_control_request(self, success: bool = True) -> None:
        """Record a control group request."""
        self.control_requests += 1
        if success:
            self.control_successes += 1
    
    def record_treatment_request(self, success: bool = True) -> None:
        """Record a treatment group request."""
        self.treatment_requests += 1
        if success:
            self.treatment_successes += 1
    
    def calculate_results(self) -> Dict[str, Any]:
        """Calculate A/B test results."""
        if self.control_requests == 0 or self.treatment_requests == 0:
            return {"error": "Insufficient data for analysis"}
        
        control_rate = self.control_successes / self.control_requests
        treatment_rate = self.treatment_successes / self.treatment_requests
        
        # Simple statistical significance calculation
        # In production, use proper statistical tests
        improvement = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        
        # Mock confidence calculation
        total_samples = self.control_requests + self.treatment_requests
        confidence = min(0.99, total_samples / self.config.minimum_sample_size * 0.8)
        
        results = {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "improvement": improvement,
            "confidence": confidence,
            "control_samples": self.control_requests,
            "treatment_samples": self.treatment_requests,
            "is_significant": confidence >= self.config.confidence_level and abs(improvement) > 0.05
        }
        
        self.results = results
        self.confidence_score = confidence
        self.statistical_significance = results["is_significant"]
        
        # Determine winner
        if results["is_significant"]:
            if treatment_rate > control_rate:
                self.winner_model_id = self.config.treatment_model_id
            else:
                self.winner_model_id = self.config.control_model_id
        
        return results
    
    def complete_test(self) -> None:
        """Complete A/B test."""
        self.status = ABTestStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.calculate_results()
    
    def fail_test(self, error_message: str) -> None:
        """Mark test as failed."""
        self.status = ABTestStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.results = {"error": error_message}
    
    def cancel_test(self) -> None:
        """Cancel A/B test."""
        self.status = ABTestStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def should_route_to_treatment(self, request_hash: float) -> bool:
        """Determine if request should go to treatment group."""
        return request_hash < self.config.traffic_split
    
    def is_complete(self) -> bool:
        """Check if test has enough data to be complete."""
        if not self.started_at:
            return False
        
        # Check duration
        duration = datetime.utcnow() - self.started_at
        if duration.total_seconds() >= self.config.duration_hours * 3600:
            return True
        
        # Check minimum sample size
        total_samples = self.control_requests + self.treatment_requests
        return total_samples >= self.config.minimum_sample_size
    
    @classmethod
    def create_new_test(
        cls,
        tenant_id: UUID,
        config: ABTestConfig,
        control_deployment_id: UUID,
        treatment_deployment_id: UUID
    ) -> "ABTest":
        """Create a new A/B test."""
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            config=config,
            control_deployment_id=control_deployment_id,
            treatment_deployment_id=treatment_deployment_id
        )


class MLDomainService(DomainService):
    """Domain service for ML-related business logic."""
    
    @staticmethod
    def validate_training_data(dataset_config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate training data configuration."""
        errors = []
        
        if not dataset_config.get("target_column"):
            errors.append("Target column is required")
        
        if not dataset_config.get("feature_columns"):
            errors.append("Feature columns are required")
        
        feature_columns = dataset_config.get("feature_columns", [])
        target_column = dataset_config.get("target_column")
        
        if target_column in feature_columns:
            errors.append("Target column cannot be in feature columns")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def suggest_model_type(dataset_info: Dict[str, Any]) -> ModelType:
        """Suggest model type based on dataset characteristics."""
        target_type = dataset_info.get("target_type", "").lower()
        num_samples = dataset_info.get("num_samples", 0)
        num_features = dataset_info.get("num_features", 0)
        
        # Simple heuristics for model suggestion
        if target_type in ["continuous", "numeric", "float"]:
            if num_samples > 10000 and num_features > 50:
                return ModelType.XGBOOST
            else:
                return ModelType.LINEAR_REGRESSION
        else:  # Classification
            if num_samples > 5000:
                return ModelType.RANDOM_FOREST
            else:
                return ModelType.RANDOM_FOREST
    
    @staticmethod
    def generate_default_hyperparameters(model_type: ModelType) -> HyperparameterConfig:
        """Generate default hyperparameters for model type."""
        defaults = {
            ModelType.LINEAR_REGRESSION: {
                "fit_intercept": True,
                "normalize": False
            },
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            ModelType.XGBOOST: {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "random_state": 42
            }
        }
        
        parameters = defaults.get(model_type, {})
        
        return HyperparameterConfig(
            parameters=parameters,
            optimization_method="grid_search",
            max_trials=10
        )
    
    @staticmethod
    def calculate_model_complexity(model_type: ModelType, parameters: Dict[str, Any]) -> float:
        """Calculate model complexity score (0-1)."""
        if model_type == ModelType.LINEAR_REGRESSION:
            return 0.1  # Simple model
        elif model_type == ModelType.RANDOM_FOREST:
            n_estimators = parameters.get("n_estimators", 100)
            max_depth = parameters.get("max_depth", 10) or 10
            return min(0.9, (n_estimators * max_depth) / 10000)
        elif model_type == ModelType.XGBOOST:
            n_estimators = parameters.get("n_estimators", 100)
            max_depth = parameters.get("max_depth", 6)
            return min(0.9, (n_estimators * max_depth) / 5000)
        else:
            return 0.5  # Default complexity