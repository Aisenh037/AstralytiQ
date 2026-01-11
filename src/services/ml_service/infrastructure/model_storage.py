"""
Model Storage Service
Handles saving and loading trained forecasting models
"""
import pickle
from pathlib import Path
from typing import Optional, List, Dict
from uuid import UUID
import os


class ModelStorage:
    """Manage storage and retrieval of trained ML models."""
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize model storage.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.forecasts_path = self.base_path / "forecasts"
        self.datasets_path = self.base_path / "datasets" / "uploads"
        
        # Create directories if they don't exist
        self.forecasts_path.mkdir(parents=True, exist_ok=True)
        self.datasets_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_id: UUID, model_type: str = "prophet") -> Path:
        """
        Get absolute path to model file.
        
        Args:
            model_id: UUID of the model
            model_type: Type of model ('prophet' or 'arima')
            
        Returns:
            Path object for the model file
        """
        filename = f"{model_type}_{model_id}.pkl"
        return self.forecasts_path / filename
    
    def save_model(self, model: any, model_id: UUID, model_type: str = "prophet") -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object (ProphetForecaster or ARIMAForecaster)
            model_id: UUID of the model
            model_type: Type of model ('prophet' or 'arima')
            
        Returns:
            String path to saved model
        """
        model_path = self.get_model_path(model_id, model_type)
        
        # Save using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)
    
    def load_model(self, model_id: UUID, model_type: str = "prophet") -> Optional[any]:
        """
        Load a trained model from disk.
        
        Args:
            model_id: UUID of the model
            model_type: Type of model ('prophet' or 'arima')
            
        Returns:
            Loaded model object or None if not found
        """
        model_path = self.get_model_path(model_id, model_type)
        
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def delete_model(self, model_id: UUID, model_type: str = "prophet") -> bool:
        """
        Delete a model file from disk.
        
        Args:
            model_id: UUID of the model
            model_type: Type of model
            
        Returns:
            True if deleted, False if not found
        """
        model_path = self.get_model_path(model_id, model_type)
        
        if model_path.exists():
            model_path.unlink()
            return True
        
        return False
    
    def model_exists(self, model_id: UUID, model_type: str = "prophet") -> bool:
        """Check if a model file exists."""
        model_path = self.get_model_path(model_id, model_type)
        return model_path.exists()
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all saved models.
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of dictionaries with model info
        """
        models = []
        
        for model_file in self.forecasts_path.glob("*.pkl"):
            file_parts = model_file.stem.split('_', 1)
            
            if len(file_parts) == 2:
                m_type, m_id = file_parts
                
                if model_type is None or m_type == model_type:
                    models.append({
                        'model_id': m_id,
                        'model_type': m_type,
                        'file_path': str(model_file),
                        'file_size_mb': model_file.stat().st_size / (1024 * 1024),
                        'created_at': model_file.stat().st_mtime
                    })
        
        return models
    
    def get_dataset_path(self, dataset_id: UUID) -> Path:
        """Get path for uploaded dataset."""
        return self.datasets_path / f"{dataset_id}.csv"
    
    def save_dataset(self, file_path: str, dataset_id: UUID) -> str:
        """
        Save uploaded dataset file.
        
        Args:
            file_path: Path to temporary uploaded file
            dataset_id: UUID for the dataset
            
        Returns:
            Path to saved dataset
        """
        import shutil
        
        dest_path = self.get_dataset_path(dataset_id)
        shutil.copy2(file_path, dest_path)
        
        return str(dest_path)
    
    def load_dataset(self, dataset_id: UUID):
        """Load dataset from storage."""
        import pandas as pd
        
        dataset_path = self.get_dataset_path(dataset_id)
        
        if not dataset_path.exists():
            return None
        
        return pd.read_csv(dataset_path)
    
    def delete_dataset(self, dataset_id: UUID) -> bool:
        """Delete a dataset file."""
        dataset_path = self.get_dataset_path(dataset_id)
        
        if dataset_path.exists():
            dataset_path.unlink()
            return True
        
        return False
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics."""
        total_models = len(list(self.forecasts_path.glob("*.pkl")))
        total_datasets = len(list(self.datasets_path.glob("*.csv")))
        
        # Calculate total size
        models_size = sum(f.stat().st_size for f in self.forecasts_path.glob("*.pkl"))
        datasets_size = sum(f.stat().st_size for f in self.datasets_path.glob("*.csv"))
        
        return {
            'total_models': total_models,
            'total_datasets': total_datasets,
            'models_size_mb': models_size / (1024 * 1024),
            'datasets_size_mb': datasets_size / (1024 * 1024),
            'total_size_mb': (models_size + datasets_size) / (1024 * 1024)
        }
