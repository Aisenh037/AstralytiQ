"""
Data transformation service that integrates with the data processing pipeline.
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from uuid import UUID, uuid4
from datetime import datetime

from ..domain.entities import (
    Dataset, DataTransformation, TransformationType, DataLineage,
    DataProcessingJob, DataDomainService
)
from ..domain.repositories import (
    DatasetRepository, DataProcessingJobRepository, FileStorageRepository
)
from .transformations import TransformationEngine, TransformationResult
from .processors import DataFormatProcessor

if TYPE_CHECKING:
    from .lineage_service import DataLineageService


class DataTransformationService:
    """Service for managing data transformations."""
    
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        job_repo: DataProcessingJobRepository,
        file_storage: FileStorageRepository,
        lineage_service: Optional['DataLineageService'] = None
    ):
        self.dataset_repo = dataset_repo
        self.job_repo = job_repo
        self.file_storage = file_storage
        self.transformation_engine = TransformationEngine()
        self._lineage_service = lineage_service
    
    async def apply_transformations(
        self,
        dataset_id: UUID,
        transformations: List[Dict[str, Any]],
        save_result: bool = True,
        create_new_dataset: bool = False
    ) -> Tuple[TransformationResult, Optional[Dataset]]:
        """Apply transformations to a dataset."""
        
        # Get original dataset
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load original data
        file_content = await self.file_storage.download_file(dataset.file_path)
        file_format = dataset.get_file_format()
        
        original_df = await DataFormatProcessor.process_file(file_content, file_format)
        
        # Execute transformations
        transformed_df, execution_summary = await self.transformation_engine.execute_transformation_pipeline(
            original_df, transformations
        )
        
        # Create transformation result
        result = TransformationResult(
            original_df=original_df,
            transformed_df=transformed_df,
            execution_summary=execution_summary,
            transformation_config=transformations
        )
        
        new_dataset = None
        
        if save_result:
            if create_new_dataset:
                # Create new dataset with transformed data
                new_dataset = await self._create_transformed_dataset(
                    original_dataset=dataset,
                    transformed_df=transformed_df,
                    transformations=transformations,
                    execution_summary=execution_summary
                )
            else:
                # Update existing dataset
                await self._update_dataset_with_transformations(
                    dataset=dataset,
                    transformed_df=transformed_df,
                    transformations=transformations,
                    execution_summary=execution_summary
                )
        
        return result, new_dataset
    
    async def _create_transformed_dataset(
        self,
        original_dataset: Dataset,
        transformed_df: pd.DataFrame,
        transformations: List[Dict[str, Any]],
        execution_summary: List[Dict[str, Any]]
    ) -> Dataset:
        """Create a new dataset with transformed data."""
        
        # Generate new dataset name
        transformation_suffix = f"_transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_name = f"{original_dataset.name}{transformation_suffix}"
        
        # Save transformed data to file
        file_format = original_dataset.get_file_format()
        
        # Convert DataFrame back to file content
        if file_format.value == "csv":
            file_content = transformed_df.to_csv(index=False).encode('utf-8')
            new_filename = f"{new_name}.csv"
        elif file_format.value == "json":
            file_content = transformed_df.to_json(orient='records').encode('utf-8')
            new_filename = f"{new_name}.json"
        else:
            # Default to CSV for other formats
            file_content = transformed_df.to_csv(index=False).encode('utf-8')
            new_filename = f"{new_name}.csv"
        
        # Upload transformed file
        new_file_path = await self.file_storage.upload_file(
            file_content, new_filename, original_dataset.tenant_id
        )
        
        # Create new dataset
        new_dataset = Dataset.create_new_dataset(
            name=new_name,
            description=f"Transformed version of '{original_dataset.name}' with {len(transformations)} transformations",
            tenant_id=original_dataset.tenant_id,
            created_by=original_dataset.created_by,
            file_path=new_file_path,
            file_format=file_format,
            file_size=len(file_content)
        )
        
        # Add transformation metadata
        if not new_dataset.metadata:
            new_dataset.metadata = {}
        
        new_dataset.metadata.update({
            "source_dataset_id": str(original_dataset.id),
            "transformations_applied": transformations,
            "transformation_summary": execution_summary,
            "transformation_timestamp": datetime.utcnow().isoformat()
        })
        
        # Save new dataset
        saved_dataset = await self.dataset_repo.save(new_dataset)
        
        # Record lineage if lineage service is available
        if self._lineage_service:
            # Convert transformation configs to DataTransformation objects
            transformation_objects = []
            for trans_config in transformations:
                transformation = DataTransformation(
                    transformation_type=TransformationType(trans_config["step"]),
                    parameters=trans_config.get("parameters", {}),
                    description=trans_config.get("description", f"Applied {trans_config['step']}")
                )
                transformation_objects.append(transformation)
            
            # Record transformation lineage
            await self._lineage_service.record_transformation_lineage(
                source_dataset_id=original_dataset.id,
                target_dataset_id=saved_dataset.id,
                transformations=transformation_objects,
                created_by=original_dataset.created_by
            )
        
        return saved_dataset
    
    async def _update_dataset_with_transformations(
        self,
        dataset: Dataset,
        transformed_df: pd.DataFrame,
        transformations: List[Dict[str, Any]],
        execution_summary: List[Dict[str, Any]]
    ) -> None:
        """Update existing dataset with transformation results."""
        
        # Save transformed data back to original file
        file_format = dataset.get_file_format()
        
        # Convert DataFrame back to file content
        if file_format.value == "csv":
            file_content = transformed_df.to_csv(index=False).encode('utf-8')
        elif file_format.value == "json":
            file_content = transformed_df.to_json(orient='records').encode('utf-8')
        else:
            # Default to CSV for other formats
            file_content = transformed_df.to_csv(index=False).encode('utf-8')
        
        # Update file in storage
        await self.file_storage.update_file(dataset.file_path, file_content)
        
        # Update dataset metadata
        if not dataset.metadata:
            dataset.metadata = {}
        
        # Keep history of transformations
        if "transformation_history" not in dataset.metadata:
            dataset.metadata["transformation_history"] = []
        
        dataset.metadata["transformation_history"].append({
            "transformations": transformations,
            "execution_summary": execution_summary,
            "timestamp": datetime.utcnow().isoformat(),
            "rows_before": execution_summary[0].get("rows_before") if execution_summary else None,
            "rows_after": execution_summary[-1].get("rows_after") if execution_summary else None
        })
        
        dataset.metadata["last_transformation"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "transformations_count": len(transformations)
        }
        
        # Update file size
        dataset.file_size = len(file_content)
        
        # Save updated dataset
        await self.dataset_repo.save(dataset)
    
    async def get_transformation_suggestions(
        self,
        dataset_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get transformation suggestions based on data quality analysis."""
        
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        suggestions = []
        
        # Get quality report
        quality_report = dataset.get_quality_report()
        if quality_report:
            # Suggest transformations based on quality issues
            for issue in quality_report.issues:
                if issue.issue_type.value == "missing_values":
                    suggestions.append({
                        "transformation": "fill_missing_values",
                        "reason": f"Found missing values in columns: {', '.join(issue.affected_columns or [])}",
                        "suggested_config": {
                            "step": "fill_missing_values",
                            "parameters": {
                                "strategy": "mean",
                                "columns": issue.affected_columns
                            }
                        },
                        "priority": "high" if issue.severity == "high" else "medium"
                    })
                
                elif issue.issue_type.value == "duplicate_rows":
                    suggestions.append({
                        "transformation": "remove_duplicates",
                        "reason": "Found duplicate rows in the dataset",
                        "suggested_config": {
                            "step": "remove_duplicates",
                            "parameters": {
                                "keep": "first"
                            }
                        },
                        "priority": "medium"
                    })
                
                elif issue.issue_type.value == "inconsistent_types":
                    if issue.affected_columns:
                        for column in issue.affected_columns:
                            suggestions.append({
                                "transformation": "standardize_text",
                                "reason": f"Inconsistent data types in column '{column}'",
                                "suggested_config": {
                                    "step": "standardize_text",
                                    "parameters": {
                                        "operations": ["trim", "remove_extra_spaces"],
                                        "columns": [column]
                                    }
                                },
                                "priority": "low"
                            })
        
        # Load data for additional analysis
        try:
            file_content = await self.file_storage.download_file(dataset.file_path)
            file_format = dataset.get_file_format()
            df = await DataFormatProcessor.process_file(file_content, file_format)
            
            # Suggest normalization for numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                # Check if values have very different scales
                scales = []
                for col in numeric_columns:
                    col_max = df[col].max()
                    col_min = df[col].min()
                    if col_max != col_min:
                        scales.append(col_max - col_min)
                
                if scales and max(scales) / min(scales) > 100:  # Large scale difference
                    suggestions.append({
                        "transformation": "min_max_scaling",
                        "reason": "Numeric columns have very different scales",
                        "suggested_config": {
                            "step": "min_max_scaling",
                            "parameters": {
                                "columns": numeric_columns
                            }
                        },
                        "priority": "medium"
                    })
            
            # Suggest text standardization for string columns
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if text_columns:
                # Check for mixed case or extra spaces
                needs_standardization = False
                for col in text_columns[:3]:  # Check first 3 text columns
                    sample_values = df[col].dropna().astype(str).head(100)
                    if len(sample_values) > 0:
                        has_mixed_case = any(v != v.lower() and v != v.upper() for v in sample_values)
                        has_extra_spaces = any('  ' in v or v != v.strip() for v in sample_values)
                        if has_mixed_case or has_extra_spaces:
                            needs_standardization = True
                            break
                
                if needs_standardization:
                    suggestions.append({
                        "transformation": "standardize_text",
                        "reason": "Text columns contain mixed case or extra spaces",
                        "suggested_config": {
                            "step": "standardize_text",
                            "parameters": {
                                "operations": ["lowercase", "trim", "remove_extra_spaces"],
                                "columns": text_columns
                            }
                        },
                        "priority": "low"
                    })
        
        except Exception:
            # If we can't load the data, just return quality-based suggestions
            pass
        
        return suggestions
    
    async def validate_transformation_pipeline(
        self,
        dataset_id: UUID,
        transformations: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate a transformation pipeline for a specific dataset."""
        
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Get dataset columns for validation
        try:
            file_content = await self.file_storage.download_file(dataset.file_path)
            file_format = dataset.get_file_format()
            df = await DataFormatProcessor.process_file(file_content, file_format)
            columns = df.columns.tolist()
        except Exception:
            columns = None
        
        return await self.transformation_engine.validate_transformation_pipeline(
            transformations, columns
        )
    
    async def get_available_transformations(self) -> Dict[str, Any]:
        """Get information about available transformations."""
        return self.transformation_engine.get_available_transformations()
    
    async def create_transformation_job(
        self,
        dataset_id: UUID,
        transformations: List[Dict[str, Any]],
        job_parameters: Dict[str, Any] = None
    ) -> DataProcessingJob:
        """Create a background job for data transformation."""
        
        job = DataProcessingJob(
            id=uuid4(),
            dataset_id=dataset_id,
            job_type="transformation",
            parameters={
                "transformations": transformations,
                **(job_parameters or {})
            },
            created_at=datetime.utcnow()
        )
        
        return await self.job_repo.save(job)
    
    async def execute_transformation_job(
        self,
        job_id: UUID
    ) -> TransformationResult:
        """Execute a transformation job."""
        
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            # Start job
            job.start()
            await self.job_repo.save(job)
            
            # Get parameters
            transformations = job.parameters.get("transformations", [])
            create_new_dataset = job.parameters.get("create_new_dataset", False)
            
            # Execute transformations
            result, new_dataset = await self.apply_transformations(
                dataset_id=job.dataset_id,
                transformations=transformations,
                save_result=True,
                create_new_dataset=create_new_dataset
            )
            
            # Complete job
            job_result = {
                "transformation_summary": result.get_transformation_summary(),
                "new_dataset_id": str(new_dataset.id) if new_dataset else None
            }
            job.complete(job_result)
            await self.job_repo.save(job)
            
            return result
            
        except Exception as e:
            # Fail job
            job.fail(str(e))
            await self.job_repo.save(job)
            raise
    
    async def get_transformation_history(
        self,
        dataset_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get transformation history for a dataset."""
        
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if not dataset.metadata or "transformation_history" not in dataset.metadata:
            return []
        
        return dataset.metadata["transformation_history"]
    
    async def create_data_lineage(
        self,
        source_dataset_ids: List[UUID],
        transformations: List[DataTransformation],
        created_by: UUID
    ) -> DataLineage:
        """Create data lineage record."""
        
        lineage = DataLineage(
            source_datasets=source_dataset_ids,
            transformations=transformations,
            created_at=datetime.utcnow(),
            created_by=created_by
        )
        
        return lineage
    
    async def preview_transformation(
        self,
        dataset_id: UUID,
        transformations: List[Dict[str, Any]],
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Preview transformation results on a sample of data."""
        
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load sample data
        file_content = await self.file_storage.download_file(dataset.file_path)
        file_format = dataset.get_file_format()
        
        full_df = await DataFormatProcessor.process_file(file_content, file_format)
        
        # Take sample
        if len(full_df) > sample_size:
            sample_df = full_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = full_df
        
        # Execute transformations on sample
        transformed_df, execution_summary = await self.transformation_engine.execute_transformation_pipeline(
            sample_df, transformations
        )
        
        # Create preview result
        preview = {
            "original_sample": {
                "shape": sample_df.shape,
                "columns": sample_df.columns.tolist(),
                "data": sample_df.head(10).to_dict('records'),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()}
            },
            "transformed_sample": {
                "shape": transformed_df.shape,
                "columns": transformed_df.columns.tolist(),
                "data": transformed_df.head(10).to_dict('records'),
                "dtypes": {col: str(dtype) for col, dtype in transformed_df.dtypes.items()}
            },
            "execution_summary": execution_summary,
            "changes": {
                "rows_changed": transformed_df.shape[0] - sample_df.shape[0],
                "columns_changed": transformed_df.shape[1] - sample_df.shape[1],
                "columns_added": list(set(transformed_df.columns) - set(sample_df.columns)),
                "columns_removed": list(set(sample_df.columns) - set(transformed_df.columns))
            }
        }
        
        return preview