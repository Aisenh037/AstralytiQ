"""
ML service API schemas.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from src.shared.domain.models import ModelType, ModelStatus, ModelMetrics


class HyperparameterConfigRequest(BaseModel):
    """Hyperparameter configuration request."""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    search_space: Optional[Dict[str, Any]] = None
    optimization_method: str = Field(default="grid_search", pattern="^(grid_search|random_search|bayesian)$")
    max_trials: int = Field(default=10, ge=1, le=100)


class TrainingConfigRequest(BaseModel):
    """Training configuration request."""
    dataset_id: UUID
    target_column: str = Field(..., min_length=1)
    feature_columns: List[str] = Field(..., min_items=1)
    model_type: ModelType
    framework: str = Field(default="scikit_learn")
    hyperparameters: HyperparameterConfigRequest = Field(default_factory=HyperparameterConfigRequest)
    validation_split: float = Field(default=0.2, gt=0, lt=1)
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    random_seed: int = Field(default=42)


class StartTrainingRequest(BaseModel):
    """Start training request."""
    config: TrainingConfigRequest
    model_name: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """Training job response."""
    id: UUID
    tenant_id: UUID
    created_by: UUID
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    model_id: Optional[UUID] = None
    logs: List[str] = Field(default_factory=list)
    created_at: datetime
    duration_seconds: Optional[float] = None
    
    class Config:
        from_attributes = True


class MLModelResponse(BaseModel):
    """ML model response."""
    id: UUID
    tenant_id: UUID
    name: str
    ml_model_type: ModelType
    version: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[ModelMetrics] = None
    artifact_path: Optional[str] = None
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    performance_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class ModelEvaluationResponse(BaseModel):
    """Model evaluation response."""
    evaluation_id: UUID
    model_id: UUID
    dataset_id: UUID
    metrics: ModelMetrics
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    predictions_sample: Optional[List[Dict[str, Any]]] = None
    evaluation_date: datetime
    classification_report: Optional[Dict[str, Any]] = None
    regression_report: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Prediction request."""
    input_data: List[Dict[str, Any]] = Field(..., min_items=1)
    model_version: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response."""
    predictions: List[Dict[str, Any]]
    model_id: UUID
    model_version: str
    timestamp: datetime
    processing_time_ms: Optional[float] = None


class ModelDeploymentRequest(BaseModel):
    """Model deployment request."""
    model_id: UUID
    deployment_config: Optional[Dict[str, Any]] = None


class ModelDeploymentResponse(BaseModel):
    """Model deployment response."""
    model_id: UUID
    status: ModelStatus
    deployment_url: Optional[str] = None
    deployed_at: datetime


class ModelComparisonRequest(BaseModel):
    """Model comparison request."""
    model_ids: List[UUID] = Field(..., min_items=2, max_items=10)
    comparison_metrics: Optional[List[str]] = None


class ModelComparisonResponse(BaseModel):
    """Model comparison response."""
    models: List[MLModelResponse]
    comparison_table: Dict[str, Dict[str, Any]]
    best_model_id: Optional[UUID] = None
    recommendation: Optional[str] = None


class HyperparameterTuningRequest(BaseModel):
    """Hyperparameter tuning request."""
    base_config: TrainingConfigRequest
    tuning_config: HyperparameterConfigRequest
    budget_minutes: Optional[int] = Field(default=60, ge=1, le=1440)  # Max 24 hours


class ModelMetricsHistoryResponse(BaseModel):
    """Model metrics history response."""
    model_id: UUID
    metrics_history: List[Dict[str, Any]]
    trend_analysis: Optional[Dict[str, Any]] = None


class TrainingJobListResponse(BaseModel):
    """Training job list response."""
    jobs: List[TrainingJobResponse]
    total_count: int
    running_count: int
    completed_count: int
    failed_count: int


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[MLModelResponse]
    total_count: int
    deployed_count: int
    training_count: int
    failed_count: int


class ModelSuggestionsRequest(BaseModel):
    """Model suggestions request."""
    dataset_id: UUID
    problem_description: Optional[str] = None
    performance_requirements: Optional[Dict[str, Any]] = None


class ModelSuggestionsResponse(BaseModel):
    """Model suggestions response."""
    suggested_models: List[Dict[str, Any]]
    dataset_analysis: Dict[str, Any]
    recommendations: List[str]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    model_id: UUID
    dataset_id: UUID
    output_format: str = Field(default="json", pattern="^(json|csv)$")
    batch_size: int = Field(default=1000, ge=1, le=10000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    job_id: UUID
    status: str
    total_records: int
    processed_records: int
    output_location: Optional[str] = None
    started_at: datetime
    estimated_completion: Optional[datetime] = None


class ModelPerformanceRequest(BaseModel):
    """Model performance monitoring request."""
    model_id: UUID
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metrics: Optional[List[str]] = None


class ModelPerformanceResponse(BaseModel):
    """Model performance monitoring response."""
    model_id: UUID
    performance_data: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    drift_detection: Optional[Dict[str, Any]] = None
    recommendations: List[str]


class AutoMLRequest(BaseModel):
    """AutoML training request."""
    dataset_id: UUID
    target_column: str
    problem_type: Optional[str] = None  # auto-detect if None
    time_budget_minutes: int = Field(default=60, ge=5, le=1440)
    quality_metric: Optional[str] = None
    feature_selection: bool = Field(default=True)


class AutoMLResponse(BaseModel):
    """AutoML training response."""
    job_id: UUID
    status: str
    best_model_id: Optional[UUID] = None
    leaderboard: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]] = None
    model_explanations: Optional[Dict[str, Any]] = None