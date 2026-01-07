"""
ML model training service implementation.
"""
import asyncio
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import logging
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..domain.entities import (
    TrainingJob, TrainingConfig, MLModel, ModelEvaluation,
    TrainingStatus, ModelFramework, HyperparameterConfig
)
from ..domain.repositories import MLModelRepository, TrainingJobRepository, ModelEvaluationRepository
from src.shared.domain.models import ModelType, ModelStatus, ModelMetrics


logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Service for training ML models."""
    
    def __init__(
        self,
        model_repo: MLModelRepository,
        job_repo: TrainingJobRepository,
        evaluation_repo: ModelEvaluationRepository,
        storage_path: str = "storage/models"
    ):
        self.model_repo = model_repo
        self.job_repo = job_repo
        self.evaluation_repo = evaluation_repo
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry = {
            ModelType.LINEAR_REGRESSION: {
                "regression": LinearRegression,
                "classification": LogisticRegression
            },
            ModelType.RANDOM_FOREST: {
                "regression": RandomForestRegressor,
                "classification": RandomForestClassifier
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.model_registry[ModelType.XGBOOST] = {
                "regression": xgb.XGBRegressor,
                "classification": xgb.XGBClassifier
            }
    
    async def start_training_job(
        self,
        config: TrainingConfig,
        tenant_id: UUID,
        created_by: UUID
    ) -> TrainingJob:
        """Start a new training job."""
        job = TrainingJob(
            id=uuid4(),
            tenant_id=tenant_id,
            created_by=created_by,
            config=config
        )
        
        # Validate configuration
        validation_errors = config.validate_config()
        if validation_errors:
            job.fail_training(f"Configuration validation failed: {', '.join(validation_errors)}")
            await self.job_repo.create(job)
            return job
        
        # Create job record
        await self.job_repo.create(job)
        
        # Start training asynchronously
        asyncio.create_task(self._execute_training(job))
        
        return job
    
    async def _execute_training(self, job: TrainingJob) -> None:
        """Execute the training job."""
        try:
            job.start_training()
            await self.job_repo.update(job)
            
            # Load and prepare data
            job.add_log("Loading training data...")
            X_train, X_test, y_train, y_test, feature_names = await self._prepare_data(job.config)
            
            # Determine problem type
            problem_type = self._determine_problem_type(y_train)
            job.add_log(f"Detected problem type: {problem_type}")
            
            # Get model class
            model_class = self._get_model_class(job.config.model_type, problem_type)
            if not model_class:
                raise ValueError(f"Model type {job.config.model_type} not supported for {problem_type}")
            
            # Hyperparameter optimization
            job.add_log("Starting hyperparameter optimization...")
            best_model, best_params, cv_scores = await self._optimize_hyperparameters(
                model_class, X_train, y_train, job.config.hyperparameters, problem_type
            )
            
            job.add_log(f"Best parameters: {best_params}")
            job.add_log(f"Cross-validation scores: {cv_scores}")
            
            # Train final model
            job.add_log("Training final model...")
            final_model = model_class(**best_params)
            final_model.fit(X_train, y_train)
            
            # Evaluate model
            job.add_log("Evaluating model...")
            metrics = self._evaluate_model(final_model, X_test, y_test, problem_type)
            
            # Save model
            job.add_log("Saving model...")
            model_path = await self._save_model(final_model, job.id, feature_names)
            
            # Create ML model record
            ml_model = MLModel.create_new_model(
                name=f"Model_{job.config.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tenant_id=job.tenant_id,
                model_type=job.config.model_type,
                version="1.0.0",
                parameters=best_params,
                training_job_id=job.id
            )
            ml_model.artifact_path = str(model_path)
            ml_model.metrics = metrics
            
            saved_model = await self.model_repo.create(ml_model)
            
            # Complete training job
            job.complete_training(saved_model.id, metrics)
            await self.job_repo.update(job)
            
            job.add_log("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training job {job.id} failed: {str(e)}")
            job.fail_training(str(e))
            await self.job_repo.update(job)
    
    async def _prepare_data(self, config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from dataset."""
        # In a real implementation, this would load data from the data service
        # For now, we'll create sample data based on the configuration
        
        # This is a placeholder - in reality, you'd fetch from the data service
        np.random.seed(config.random_seed)
        n_samples = 1000
        n_features = len(config.feature_columns)
        
        # Generate sample data
        X = np.random.randn(n_samples, n_features)
        
        # Generate target based on model type
        if config.model_type in [ModelType.LINEAR_REGRESSION]:
            # Regression target
            y = X.sum(axis=1) + np.random.randn(n_samples) * 0.1
        else:
            # Classification target
            y = (X.sum(axis=1) > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.validation_split, random_state=config.random_seed
        )
        
        return X_train, X_test, y_train, y_test, config.feature_columns
    
    def _determine_problem_type(self, y: np.ndarray) -> str:
        """Determine if this is a regression or classification problem."""
        if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
            return "classification"
        else:
            return "regression"
    
    def _get_model_class(self, model_type: ModelType, problem_type: str):
        """Get the appropriate model class."""
        if model_type not in self.model_registry:
            return None
        
        return self.model_registry[model_type].get(problem_type)
    
    async def _optimize_hyperparameters(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparams: HyperparameterConfig,
        problem_type: str
    ) -> Tuple[Any, Dict[str, Any], List[float]]:
        """Optimize hyperparameters using the specified method."""
        
        if hyperparams.search_space:
            param_grid = hyperparams.search_space
        else:
            param_grid = self._get_default_param_grid(model_class, problem_type)
        
        # Choose optimization method
        if hyperparams.optimization_method == "grid_search":
            search = GridSearchCV(
                model_class(),
                param_grid,
                cv=5,
                scoring=self._get_scoring_metric(problem_type),
                n_jobs=-1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                model_class(),
                param_grid,
                n_iter=hyperparams.max_trials,
                cv=5,
                scoring=self._get_scoring_metric(problem_type),
                n_jobs=-1,
                random_state=42
            )
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(
            search.best_estimator_, X_train, y_train, cv=5,
            scoring=self._get_scoring_metric(problem_type)
        )
        
        return search.best_estimator_, search.best_params_, cv_scores.tolist()
    
    def _get_default_param_grid(self, model_class, problem_type: str) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter optimization."""
        class_name = model_class.__name__
        
        if "LinearRegression" in class_name:
            return {"fit_intercept": [True, False]}
        elif "LogisticRegression" in class_name:
            return {"C": [0.1, 1.0, 10.0], "max_iter": [100, 1000]}
        elif "RandomForest" in class_name:
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        elif "XGB" in class_name:
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        else:
            return {}
    
    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for the problem type."""
        if problem_type == "regression":
            return "r2"
        else:
            return "accuracy"
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, problem_type: str) -> ModelMetrics:
        """Evaluate the trained model."""
        y_pred = model.predict(X_test)
        
        metrics = ModelMetrics()
        
        if problem_type == "regression":
            metrics.mae = float(mean_absolute_error(y_test, y_pred))
            metrics.mse = float(mean_squared_error(y_test, y_pred))
            metrics.r2_score = float(r2_score(y_test, y_pred))
        else:
            metrics.accuracy = float(accuracy_score(y_test, y_pred))
            metrics.precision = float(precision_score(y_test, y_pred, average='weighted'))
            metrics.recall = float(recall_score(y_test, y_pred, average='weighted'))
            metrics.f1_score = float(f1_score(y_test, y_pred, average='weighted'))
        
        return metrics
    
    async def _save_model(self, model, job_id: UUID, feature_names: List[str]) -> Path:
        """Save the trained model to disk."""
        model_dir = self.storage_path / str(job_id)
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "model.pkl"
        
        # Save model and metadata
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "training_timestamp": datetime.utcnow().isoformat(),
            "job_id": str(job_id)
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        return model_path
    
    async def get_training_job_status(self, job_id: UUID) -> Optional[TrainingJob]:
        """Get training job status."""
        return await self.job_repo.get_by_id(job_id)
    
    async def cancel_training_job(self, job_id: UUID) -> bool:
        """Cancel a training job."""
        job = await self.job_repo.get_by_id(job_id)
        if not job or job.status != TrainingStatus.RUNNING:
            return False
        
        job.cancel_training()
        await self.job_repo.update(job)
        return True
    
    async def get_tenant_training_jobs(self, tenant_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a tenant."""
        return await self.job_repo.get_by_tenant(tenant_id)
    
    async def get_tenant_models(self, tenant_id: UUID) -> List[MLModel]:
        """Get all models for a tenant."""
        return await self.model_repo.get_by_tenant(tenant_id)
    
    async def evaluate_model(
        self,
        model_id: UUID,
        dataset_id: UUID,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> ModelEvaluation:
        """Evaluate a model on a dataset."""
        model = await self.model_repo.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Load model
        loaded_model = await self._load_model(model.artifact_path)
        
        # Load evaluation data (placeholder)
        # In reality, this would load from the data service
        np.random.seed(42)
        X_eval = np.random.randn(100, 5)  # Sample evaluation data
        y_eval = (X_eval.sum(axis=1) > 0).astype(int)  # Sample labels
        
        # Make predictions
        y_pred = loaded_model.predict(X_eval)
        
        # Calculate metrics
        problem_type = "classification" if len(np.unique(y_eval)) <= 10 else "regression"
        metrics = self._evaluate_model(loaded_model, X_eval, y_eval, problem_type)
        
        # Create evaluation record
        evaluation = ModelEvaluation(
            model_id=model_id,
            evaluation_id=uuid4(),
            dataset_id=dataset_id,
            metrics=metrics,
            evaluation_date=datetime.utcnow()
        )
        
        # Add confusion matrix for classification
        if problem_type == "classification":
            cm = confusion_matrix(y_eval, y_pred)
            evaluation.confusion_matrix = cm.tolist()
        
        # Save evaluation
        await self.evaluation_repo.create(evaluation)
        
        return evaluation
    
    async def _load_model(self, model_path: str):
        """Load a saved model."""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        return model_data["model"]
    
    async def deploy_model(self, model_id: UUID) -> bool:
        """Deploy a model for serving."""
        model = await self.model_repo.get_by_id(model_id)
        if not model:
            return False
        
        model.deploy()
        await self.model_repo.update(model)
        return True
    
    async def get_model_predictions(
        self,
        model_id: UUID,
        input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get predictions from a deployed model."""
        model = await self.model_repo.get_by_id(model_id)
        if not model or model.status != ModelStatus.DEPLOYED:
            raise ValueError("Model not found or not deployed")
        
        # Load model
        loaded_model = await self._load_model(model.artifact_path)
        
        # Convert input data to numpy array
        # This is a simplified version - in reality, you'd need proper feature engineering
        X = np.array([[list(row.values())] for row in input_data]).reshape(len(input_data), -1)
        
        # Make predictions
        predictions = loaded_model.predict(X)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "input": input_data[i],
                "prediction": float(pred) if isinstance(pred, (int, float, np.number)) else pred.tolist(),
                "model_id": str(model_id),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return results