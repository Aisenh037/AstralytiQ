"""
Unified data processing pipeline for multi-format data processing.
"""
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import pandas as pd
from io import BytesIO

from ..domain.entities import (
    Dataset, DataFormat, DataSchema, DataQualityReport, 
    DataTransformation, DataLineage, DataProcessingJob,
    DataDomainService
)
from ..domain.repositories import DatasetRepository, DataProcessingJobRepository, FileStorageRepository
from .processors import DataFormatProcessor, DataValidator, SchemaDetector, DataProfiler


class DataProcessingPipeline:
    """Unified data processing pipeline for all supported formats."""
    
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        job_repo: DataProcessingJobRepository,
        file_storage: FileStorageRepository
    ):
        self.dataset_repo = dataset_repo
        self.job_repo = job_repo
        self.file_storage = file_storage
    
    async def process_dataset(
        self,
        dataset_id: UUID,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> DataProcessingJob:
        """Process a dataset through the complete pipeline."""
        # Create processing job
        job = DataProcessingJob(
            id=UUID(),
            dataset_id=dataset_id,
            job_type="full_processing",
            parameters=processing_options or {},
            created_at=datetime.utcnow()
        )
        
        # Save job
        await self.job_repo.save(job)
        
        try:
            # Start processing
            job.start()
            await self.job_repo.save(job)
            
            # Get dataset
            dataset = await self.dataset_repo.get_by_id(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Mark dataset as processing
            dataset.mark_processing()
            await self.dataset_repo.save(dataset)
            
            # Execute pipeline stages
            result = await self._execute_pipeline_stages(dataset, processing_options or {})
            
            # Complete job
            job.complete(result)
            await self.job_repo.save(job)
            
            # Mark dataset as processed
            dataset.mark_processed()
            await self.dataset_repo.save(dataset)
            
            return job
            
        except Exception as e:
            # Handle failure
            job.fail(str(e))
            await self.job_repo.save(job)
            
            if 'dataset' in locals():
                dataset.mark_failed(str(e))
                await self.dataset_repo.save(dataset)
            
            raise
    
    async def _execute_pipeline_stages(
        self,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute all pipeline stages."""
        result = {
            "dataset_id": str(dataset.id),
            "stages_completed": [],
            "processing_time": {},
            "errors": []
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Stage 1: Load and parse data
            stage_start = datetime.utcnow()
            df, parse_info = await self._stage_load_and_parse(dataset, options)
            result["stages_completed"].append("load_and_parse")
            result["processing_time"]["load_and_parse"] = (datetime.utcnow() - stage_start).total_seconds()
            result["parse_info"] = parse_info
            
            # Stage 2: Schema detection and validation
            stage_start = datetime.utcnow()
            schema_info = await self._stage_schema_detection(dataset, df, options)
            result["stages_completed"].append("schema_detection")
            result["processing_time"]["schema_detection"] = (datetime.utcnow() - stage_start).total_seconds()
            result["schema_info"] = schema_info
            
            # Stage 3: Data quality assessment
            stage_start = datetime.utcnow()
            quality_info = await self._stage_quality_assessment(dataset, df, options)
            result["stages_completed"].append("quality_assessment")
            result["processing_time"]["quality_assessment"] = (datetime.utcnow() - stage_start).total_seconds()
            result["quality_info"] = quality_info
            
            # Stage 4: Data profiling (optional)
            if options.get("enable_profiling", True):
                stage_start = datetime.utcnow()
                profile_info = await self._stage_data_profiling(dataset, df, options)
                result["stages_completed"].append("data_profiling")
                result["processing_time"]["data_profiling"] = (datetime.utcnow() - stage_start).total_seconds()
                result["profile_info"] = profile_info
            
            # Stage 5: Format-specific processing
            stage_start = datetime.utcnow()
            format_info = await self._stage_format_specific_processing(dataset, df, options)
            result["stages_completed"].append("format_specific")
            result["processing_time"]["format_specific"] = (datetime.utcnow() - stage_start).total_seconds()
            result["format_info"] = format_info
            
            # Calculate total processing time
            result["total_processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["total_processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            raise
        
        return result
    
    async def _stage_load_and_parse(
        self,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 1: Load and parse data from file."""
        # Get file content
        file_content = await self.file_storage.download_file(dataset.file_path)
        
        # Determine format
        file_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        # Parse with format-specific options
        parse_options = options.get("parse_options", {})
        
        try:
            df = await DataFormatProcessor.process_file(
                file_content, 
                file_format, 
                **parse_options
            )
            
            parse_info = {
                "format": file_format.value,
                "rows_loaded": len(df),
                "columns_loaded": len(df.columns),
                "column_names": list(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "parse_options_used": parse_options
            }
            
            return df, parse_info
            
        except Exception as e:
            raise ValueError(f"Failed to parse {file_format.value} file: {str(e)}")
    
    async def _stage_schema_detection(
        self,
        dataset: Dataset,
        df: pd.DataFrame,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: Detect and validate schema."""
        # Detect schema
        schema = await SchemaDetector.detect_schema(df, options.get("schema_sample_size", 1000))
        
        # Update dataset with schema
        dataset.update_schema(schema)
        await self.dataset_repo.save(dataset)
        
        # Get schema suggestions
        suggestions = await SchemaDetector.suggest_schema_improvements(df, schema)
        
        schema_info = {
            "columns_detected": len(schema.columns),
            "schema": schema.dict(),
            "suggestions": suggestions,
            "primary_key": schema.primary_key,
            "constraints": schema.constraints
        }
        
        return schema_info
    
    async def _stage_quality_assessment(
        self,
        dataset: Dataset,
        df: pd.DataFrame,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: Assess data quality."""
        # Get current schema
        schema = None
        if dataset.schema:
            schema = DataSchema(**dataset.schema)
        
        # Validate data quality
        quality_report = await DataValidator.validate_data(df, schema)
        
        # Update dataset with quality report
        dataset.update_quality_report(quality_report)
        await self.dataset_repo.save(dataset)
        
        quality_info = {
            "quality_score": quality_report.quality_score,
            "total_issues": len(quality_report.issues),
            "critical_issues": len(quality_report.get_issues_by_severity("critical")),
            "high_issues": len(quality_report.get_issues_by_severity("high")),
            "medium_issues": len(quality_report.get_issues_by_severity("medium")),
            "low_issues": len(quality_report.get_issues_by_severity("low")),
            "missing_values": quality_report.missing_values_count,
            "duplicate_rows": quality_report.duplicate_rows_count,
            "has_critical_issues": quality_report.has_critical_issues()
        }
        
        return quality_info
    
    async def _stage_data_profiling(
        self,
        dataset: Dataset,
        df: pd.DataFrame,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 4: Generate data profile."""
        profile = await DataProfiler.profile_data(df)
        
        # Store profile in dataset metadata
        if not dataset.metadata:
            dataset.metadata = {}
        dataset.metadata["data_profile"] = profile
        await self.dataset_repo.save(dataset)
        
        profile_info = {
            "columns_profiled": len(profile["columns"]),
            "basic_stats": profile["basic_stats"],
            "profile_generated": True
        }
        
        return profile_info
    
    async def _stage_format_specific_processing(
        self,
        dataset: Dataset,
        df: pd.DataFrame,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 5: Format-specific processing and validation."""
        file_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        format_processor = FormatSpecificProcessor()
        format_info = await format_processor.process_format_specific(
            df, file_format, dataset, options
        )
        
        return format_info


class FormatSpecificProcessor:
    """Handles format-specific processing and validation."""
    
    async def process_format_specific(
        self,
        df: pd.DataFrame,
        file_format: DataFormat,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process format-specific requirements."""
        
        processors = {
            DataFormat.CSV: self._process_csv_specific,
            DataFormat.EXCEL: self._process_excel_specific,
            DataFormat.JSON: self._process_json_specific,
            DataFormat.XML: self._process_xml_specific,
            DataFormat.TSV: self._process_tsv_specific,
            DataFormat.PARQUET: self._process_parquet_specific
        }
        
        processor = processors.get(file_format, self._process_generic)
        return await processor(df, dataset, options)
    
    async def _process_csv_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """CSV-specific processing."""
        info = {
            "format": "csv",
            "delimiter_detected": ",",  # Could be enhanced to detect actual delimiter
            "encoding_issues": [],
            "quote_character": '"',
            "header_row": 0
        }
        
        # Check for common CSV issues
        issues = []
        
        # Check for mixed delimiters
        if any(',' in str(col) and ';' in str(col) for col in df.columns):
            issues.append("Mixed delimiter usage detected in column names")
        
        # Check for embedded newlines
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.contains('\n').any():
                issues.append(f"Embedded newlines found in column '{col}'")
        
        # Check for inconsistent quoting
        for col in df.select_dtypes(include=['object']).columns:
            values = df[col].dropna().astype(str)
            if len(values) > 0:
                quoted_count = values.str.startswith('"').sum()
                if 0 < quoted_count < len(values):
                    issues.append(f"Inconsistent quoting in column '{col}'")
        
        info["csv_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_excel_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Excel-specific processing."""
        info = {
            "format": "excel",
            "sheet_name": options.get("sheet_name", "Sheet1"),
            "has_merged_cells": False,  # Would need openpyxl to detect
            "has_formulas": False,
            "has_charts": False
        }
        
        issues = []
        
        # Check for Excel-specific data types
        for col in df.columns:
            # Check for Excel date issues (dates as numbers)
            if df[col].dtype == 'float64':
                # Excel dates are stored as numbers since 1900-01-01
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Check if values could be Excel dates (between 1 and 50000)
                    if sample_values.between(1, 50000).all():
                        issues.append(f"Column '{col}' may contain Excel date values stored as numbers")
        
        # Check for Excel error values
        for col in df.select_dtypes(include=['object']).columns:
            error_values = ['#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!', '#REF!', '#VALUE!']
            for error_val in error_values:
                if df[col].astype(str).str.contains(error_val, na=False).any():
                    issues.append(f"Excel error value '{error_val}' found in column '{col}'")
        
        info["excel_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_json_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """JSON-specific processing."""
        info = {
            "format": "json",
            "structure_type": "array_of_objects",  # Default assumption
            "nested_fields": [],
            "array_fields": []
        }
        
        issues = []
        
        # Check for nested structures that were flattened
        for col in df.columns:
            if '.' in col:
                info["nested_fields"].append(col)
            
            # Check for JSON strings that weren't parsed
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(10)
                if len(sample_values) > 0:
                    # Check if values look like JSON
                    json_like = sample_values.str.startswith(('{', '[')).any()
                    if json_like:
                        issues.append(f"Column '{col}' contains unparsed JSON strings")
        
        # Check for array-like data
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().astype(str).head(10)
            if len(sample_values) > 0:
                if sample_values.str.startswith('[').any():
                    info["array_fields"].append(col)
        
        info["json_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_xml_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """XML-specific processing."""
        info = {
            "format": "xml",
            "root_element": "root",  # Default assumption
            "namespaces_detected": [],
            "attributes_as_columns": []
        }
        
        issues = []
        
        # Check for XML attributes that became columns
        for col in df.columns:
            if col.startswith('@'):
                info["attributes_as_columns"].append(col)
        
        # Check for CDATA sections
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.contains('<!\[CDATA\[', na=False).any():
                issues.append(f"CDATA sections found in column '{col}'")
        
        # Check for mixed content (text and elements)
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().astype(str).head(10)
            if len(sample_values) > 0:
                if sample_values.str.contains('<[^>]+>', na=False).any():
                    issues.append(f"Mixed content (text and XML elements) in column '{col}'")
        
        info["xml_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_tsv_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """TSV-specific processing."""
        info = {
            "format": "tsv",
            "delimiter": "\t",
            "tab_issues": []
        }
        
        issues = []
        
        # Check for embedded tabs in data
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.contains('\t').any():
                issues.append(f"Embedded tab characters found in column '{col}'")
        
        # Check for mixed whitespace
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().astype(str).head(100)
            if len(sample_values) > 0:
                # Check for values with multiple consecutive spaces
                if sample_values.str.contains('  +').any():
                    issues.append(f"Multiple consecutive spaces in column '{col}' - may indicate formatting issues")
        
        info["tsv_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_parquet_specific(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parquet-specific processing."""
        info = {
            "format": "parquet",
            "compression": "unknown",  # Would need pyarrow metadata
            "schema_preserved": True,
            "metadata_available": False
        }
        
        issues = []
        
        # Parquet preserves data types well, so fewer issues expected
        # Check for any type inconsistencies that might indicate corruption
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # In Parquet, object columns should be consistent
                sample_values = df[col].dropna()
                if len(sample_values) > 0:
                    # Check for mixed types in object columns
                    types_found = set(type(v).__name__ for v in sample_values.head(100))
                    if len(types_found) > 1:
                        issues.append(f"Mixed data types in column '{col}': {', '.join(types_found)}")
        
        info["parquet_specific_issues"] = issues
        info["issues_found"] = len(issues)
        
        return info
    
    async def _process_generic(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic processing for unknown formats."""
        return {
            "format": "generic",
            "processing_applied": "basic_validation",
            "issues_found": 0,
            "generic_issues": []
        }


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    
    def __init__(self, message: str, stage: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class ProcessingJobManager:
    """Manages data processing jobs and their lifecycle."""
    
    def __init__(self, job_repo: DataProcessingJobRepository):
        self.job_repo = job_repo
    
    async def get_job_status(self, job_id: UUID) -> Optional[DataProcessingJob]:
        """Get job status."""
        return await self.job_repo.get_by_id(job_id)
    
    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a running job."""
        job = await self.job_repo.get_by_id(job_id)
        if job and job.status == "running":
            job.fail("Job cancelled by user")
            await self.job_repo.save(job)
            return True
        return False
    
    async def retry_failed_job(self, job_id: UUID) -> Optional[DataProcessingJob]:
        """Retry a failed job."""
        job = await self.job_repo.get_by_id(job_id)
        if job and job.status == "failed":
            # Create new job with same parameters
            new_job = DataProcessingJob(
                id=UUID(),
                dataset_id=job.dataset_id,
                job_type=job.job_type,
                parameters=job.parameters,
                created_at=datetime.utcnow()
            )
            return await self.job_repo.save(new_job)
        return None
    
    async def get_jobs_by_dataset(self, dataset_id: UUID) -> List[DataProcessingJob]:
        """Get all jobs for a dataset."""
        return await self.job_repo.get_by_dataset(dataset_id)
    
    async def get_pending_jobs(self, limit: int = 10) -> List[DataProcessingJob]:
        """Get pending jobs for processing."""
        return await self.job_repo.get_pending_jobs(limit)