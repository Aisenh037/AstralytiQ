"""
ML service API routes.
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer

from ..domain.repositories import MLModelRepository, TrainingJobRepository, ModelEvaluationRepository
from ..infrastructure.training_service import ModelTrainingService
from ..infrastructure.repositories import SQLMLModelRepository, SQLTrainingJobRepository, SQLModelEvaluationRepository
from .schemas import (
    StartTrainingRequest, TrainingJobResponse, MLModelResponse, ModelEvaluationResponse,
    PredictionRequest, PredictionResponse, ModelDeploymentRequest, ModelDeploymentResponse,
    ModelComparisonRequest, ModelComparisonResponse, TrainingJobListResponse, ModelListResponse,
    BatchPredictionRequest, BatchPredictionResponse, ModelPerformanceRequest, ModelPerformanceResponse,
    AutoMLRequest, AutoMLResponse, ModelSuggestionsRequest, ModelSuggestionsResponse
)
from src.shared.infrastructure.container import get_configured_container


router = APIRouter(prefix="/api/v1/ml", tags=["ML/Analytics"])
security = HTTPBearer()


def get_ml_service() -> ModelTrainingService:
    """Get ML training service instance."""
    container = get_configured_container()
    
    # Get repositories
    model_repo = SQLMLModelRepository(container.db_session())
    job_repo = SQLTrainingJobRepository(container.db_session())
    evaluation_repo = SQLModelEvaluationRepository(container.db_session())
    
    return ModelTrainingService(model_repo, job_repo, evaluation_repo)


def get_current_user_id() -> UUID:
    """Get current user ID from JWT token."""
    # In a real implementation, this would decode the JWT token
    # For now, return a dummy user ID
    from uuid import uuid4
    return uuid4()


def get_current_tenant_id() -> UUID:
    """Get current tenant ID from context."""
    # In a real implementation, this would get tenant from middleware
    # For now, return a dummy tenant ID
    from uuid import uuid4
    return uuid4()


@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training_job(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    current_user: UUID = Depends(get_current_user_id),
    tenant_id: UUID = Depends(get_current_tenant_id),
    token: str = Depends(security)
):
    """Start a new model training job."""
    try:
        job = await ml_service.start_training_job(
            config=request.config,
            tenant_id=tenant_id,
            created_by=current_user
        )
        
        return TrainingJobResponse(
            id=job.id,
            tenant_id=job.tenant_id,
            created_by=job.created_by,
            status=job.status.value,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            metrics=job.metrics,
            model_id=job.model_id,
            logs=job.logs,
            created_at=datetime.utcnow(),
            duration_seconds=job.get_duration()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start training job: {str(e)}"
        )


@router.get("/training/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    ml_service: ModelTrainingService = Depends(get_ml_service),
    tenant_id: UUID = Depends(get_current_tenant_id),
    token: str = Depends(security)
):
    """List all training jobs for the current tenant."""
    try:
        jobs = await ml_service.get_tenant_training_jobs(tenant_id)
        
        job_responses = []
        running_count = completed_count = failed_count = 0
        
        for job in jobs:
            job_responses.append(TrainingJobResponse(
                id=job.id,
                tenant_id=job.tenant_id,
                created_by=job.created_by,
                status=job.status.value,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
                metrics=job.metrics,
                model_id=job.model_id,
                logs=job.logs,
                created_at=datetime.utcnow(),
                duration_seconds=job.get_duration()
            ))
            
            if job.status.value == "running":
                running_count += 1
            elif job.status.value == "completed":
                completed_count += 1
            elif job.status.value == "failed":
                failed_count += 1
        
        return TrainingJobListResponse(
            jobs=job_responses,
            total_count=len(jobs),
            running_count=running_count,
            completed_count=completed_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list training jobs: {str(e)}"
        )


@router.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: UUID,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Get training job details."""
    try:
        job = await ml_service.get_training_job_status(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found"
            )
        
        return TrainingJobResponse(
            id=job.id,
            tenant_id=job.tenant_id,
            created_by=job.created_by,
            status=job.status.value,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            metrics=job.metrics,
            model_id=job.model_id,
            logs=job.logs,
            created_at=datetime.utcnow(),
            duration_seconds=job.get_duration()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training job: {str(e)}"
        )


@router.post("/training/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: UUID,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Cancel a running training job."""
    try:
        success = await ml_service.cancel_training_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel job (not found or not running)"
            )
        
        return {"message": "Training job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel training job: {str(e)}"
        )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    ml_service: ModelTrainingService = Depends(get_ml_service),
    tenant_id: UUID = Depends(get_current_tenant_id),
    token: str = Depends(security)
):
    """List all models for the current tenant."""
    try:
        models = await ml_service.get_tenant_models(tenant_id)
        
        model_responses = []
        deployed_count = training_count = failed_count = 0
        
        for model in models:
            model_responses.append(MLModelResponse(
                id=model.id,
                tenant_id=model.tenant_id,
                name=model.name,
                ml_model_type=model.ml_model_type,
                version=model.version,
                parameters=model.parameters,
                metrics=model.metrics,
                artifact_path=model.artifact_path,
                status=model.status,
                created_at=model.created_at,
                updated_at=model.updated_at,
                metadata=model.metadata,
                performance_score=model.get_performance_score()
            ))
            
            if model.status.value == "deployed":
                deployed_count += 1
            elif model.status.value == "training":
                training_count += 1
            elif model.status.value == "failed":
                failed_count += 1
        
        return ModelListResponse(
            models=model_responses,
            total_count=len(models),
            deployed_count=deployed_count,
            training_count=training_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=MLModelResponse)
async def get_model(
    model_id: UUID,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Get model details."""
    try:
        model = await ml_service.model_repo.get_by_id(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return MLModelResponse(
            id=model.id,
            tenant_id=model.tenant_id,
            name=model.name,
            ml_model_type=model.ml_model_type,
            version=model.version,
            parameters=model.parameters,
            metrics=model.metrics,
            artifact_path=model.artifact_path,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata,
            performance_score=model.get_performance_score()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.post("/models/{model_id}/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(
    model_id: UUID,
    request: ModelDeploymentRequest,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Deploy a model for serving."""
    try:
        success = await ml_service.deploy_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Get updated model
        model = await ml_service.model_repo.get_by_id(model_id)
        
        return ModelDeploymentResponse(
            model_id=model_id,
            status=model.status,
            deployment_url=f"/api/v1/ml/models/{model_id}/predict",
            deployed_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy model: {str(e)}"
        )


@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(
    model_id: UUID,
    request: PredictionRequest,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Get predictions from a deployed model."""
    try:
        start_time = datetime.utcnow()
        
        predictions = await ml_service.get_model_predictions(
            model_id=model_id,
            input_data=request.input_data
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Get model for version info
        model = await ml_service.model_repo.get_by_id(model_id)
        
        return PredictionResponse(
            predictions=predictions,
            model_id=model_id,
            model_version=model.version if model else "unknown",
            timestamp=end_time,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}"
        )


@router.post("/models/{model_id}/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model(
    model_id: UUID,
    dataset_id: UUID,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Evaluate a model on a dataset."""
    try:
        evaluation = await ml_service.evaluate_model(
            model_id=model_id,
            dataset_id=dataset_id
        )
        
        return ModelEvaluationResponse(
            evaluation_id=evaluation.evaluation_id,
            model_id=evaluation.model_id,
            dataset_id=evaluation.dataset_id,
            metrics=evaluation.metrics,
            confusion_matrix=evaluation.confusion_matrix,
            feature_importance=evaluation.feature_importance,
            predictions_sample=evaluation.predictions_sample,
            evaluation_date=evaluation.evaluation_date,
            classification_report=evaluation.get_classification_report(),
            regression_report=evaluation.get_regression_report()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate model: {str(e)}"
        )


@router.post("/models/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: ModelComparisonRequest,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Compare multiple models."""
    try:
        models = []
        for model_id in request.model_ids:
            model = await ml_service.model_repo.get_by_id(model_id)
            if model:
                models.append(model)
        
        if len(models) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 valid models required for comparison"
            )
        
        # Create comparison table
        comparison_table = {}
        best_model = None
        best_score = -1
        
        for model in models:
            model_data = {
                "name": model.name,
                "type": model.ml_model_type.value,
                "version": model.version,
                "status": model.status.value,
                "performance_score": model.get_performance_score()
            }
            
            if model.metrics:
                model_data.update({
                    "accuracy": model.metrics.accuracy,
                    "precision": model.metrics.precision,
                    "recall": model.metrics.recall,
                    "f1_score": model.metrics.f1_score,
                    "r2_score": model.metrics.r2_score,
                    "mae": model.metrics.mae,
                    "mse": model.metrics.mse
                })
            
            comparison_table[str(model.id)] = model_data
            
            # Find best model
            score = model.get_performance_score()
            if score and score > best_score:
                best_score = score
                best_model = model
        
        # Generate recommendation
        recommendation = None
        if best_model:
            recommendation = f"Model '{best_model.name}' has the highest performance score ({best_score:.3f})"
        
        model_responses = [
            MLModelResponse(
                id=model.id,
                tenant_id=model.tenant_id,
                name=model.name,
                ml_model_type=model.ml_model_type,
                version=model.version,
                parameters=model.parameters,
                metrics=model.metrics,
                artifact_path=model.artifact_path,
                status=model.status,
                created_at=model.created_at,
                updated_at=model.updated_at,
                metadata=model.metadata,
                performance_score=model.get_performance_score()
            )
            for model in models
        ]
        
        return ModelComparisonResponse(
            models=model_responses,
            comparison_table=comparison_table,
            best_model_id=best_model.id if best_model else None,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


# Additional endpoints for advanced features

@router.post("/suggestions", response_model=ModelSuggestionsResponse)
async def get_model_suggestions(
    request: ModelSuggestionsRequest,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Get model suggestions based on dataset characteristics."""
    try:
        from ..domain.entities import MLDomainService
        
        # Mock dataset analysis - in reality, this would analyze the actual dataset
        dataset_analysis = {
            "num_samples": 1000,
            "num_features": 10,
            "target_type": "continuous",
            "missing_values": 0.05,
            "categorical_features": 3,
            "numerical_features": 7
        }
        
        # Get model suggestion
        suggested_type = MLDomainService.suggest_model_type(dataset_analysis)
        
        suggested_models = [
            {
                "model_type": suggested_type.value,
                "confidence": 0.85,
                "reason": "Best fit for dataset characteristics",
                "expected_performance": "High"
            }
        ]
        
        recommendations = [
            "Consider feature scaling for better performance",
            "Cross-validation recommended for model selection",
            "Monitor for overfitting with this dataset size"
        ]
        
        return ModelSuggestionsResponse(
            suggested_models=suggested_models,
            dataset_analysis=dataset_analysis,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model suggestions: {str(e)}"
        )


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    ml_service: ModelTrainingService = Depends(get_ml_service),
    token: str = Depends(security)
):
    """Start batch prediction job."""
    try:
        from uuid import uuid4
        
        job_id = uuid4()
        
        # In a real implementation, this would start a background job
        # For now, return a mock response
        
        return BatchPredictionResponse(
            job_id=job_id,
            status="started",
            total_records=1000,  # Mock value
            processed_records=0,
            started_at=datetime.utcnow(),
            estimated_completion=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start batch prediction: {str(e)}"
        )