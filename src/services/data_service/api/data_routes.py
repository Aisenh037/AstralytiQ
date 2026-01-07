"""
Data service API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import io

from .schemas import (
    DatasetCreateRequest, DatasetResponse, DatasetListResponse,
    DataSchemaResponse, DataQualityReportResponse, DataProfileResponse,
    DataTransformationRequest, DataProcessingJobResponse,
    FileUploadResponse, DataValidationRequest, DataSearchRequest,
    TenantStatsResponse, ErrorResponse, SuccessResponse
)
from ..domain.entities import (
    Dataset, DataFormat, DataDomainService, DataProcessingJob
)
from ..domain.repositories import DatasetRepository, DataProcessingJobRepository, FileStorageRepository
from ..infrastructure.processors import DataFormatProcessor, DataValidator, SchemaDetector, DataProfiler, DataFormatConverter
from ..infrastructure.pipeline import DataProcessingPipeline, ProcessingJobManager
from ..infrastructure.transformation_service import DataTransformationService
from ..infrastructure.lineage_service import DataLineageService
from src.shared.infrastructure.container import get_configured_container


router = APIRouter(prefix="/api/v1/data", tags=["data"])


def get_dataset_repository() -> DatasetRepository:
    """Get dataset repository dependency."""
    container = get_configured_container()
    return container.dataset_repository()


def get_job_repository() -> DataProcessingJobRepository:
    """Get job repository dependency."""
    container = get_configured_container()
    return container.data_processing_job_repository()


def get_file_storage() -> FileStorageRepository:
    """Get file storage dependency."""
    container = get_configured_container()
    return container.file_storage_repository()


def get_transformation_service() -> DataTransformationService:
    """Get transformation service dependency."""
    container = get_configured_container()
    
    # Create lineage service
    lineage_service = DataLineageService(
        container.dataset_repository(),
        container.data_processing_job_repository(),
        container.file_storage_repository()
    )
    
    # Create transformation service with lineage integration
    return DataTransformationService(
        container.dataset_repository(),
        container.data_processing_job_repository(),
        container.file_storage_repository(),
        lineage_service
    )


def get_lineage_service() -> DataLineageService:
    """Get lineage service dependency."""
    container = get_configured_container()
    return DataLineageService(
        container.dataset_repository(),
        container.data_processing_job_repository(),
        container.file_storage_repository()
    )


# Mock tenant and user context for now
async def get_current_tenant_id() -> UUID:
    """Get current tenant ID from context."""
    # In real implementation, this would extract from JWT token or request context
    return UUID("12345678-1234-5678-9012-123456789012")


async def get_current_user_id() -> UUID:
    """Get current user ID from context."""
    # In real implementation, this would extract from JWT token
    return UUID("87654321-4321-8765-2109-876543210987")


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user_id: UUID = Depends(get_current_user_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Upload a data file and create dataset."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file size
        is_valid_size, size_error = DataDomainService.validate_file_size(file_size)
        if not is_valid_size:
            raise HTTPException(status_code=400, detail=size_error)
        
        # Validate dataset name
        is_valid_name, name_error = DataDomainService.is_dataset_name_valid(name)
        if not is_valid_name:
            raise HTTPException(status_code=400, detail=name_error)
        
        # Check if dataset name already exists
        existing_dataset = await dataset_repo.get_by_name(tenant_id, name)
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"Dataset with name '{name}' already exists")
        
        # Detect file format
        file_format = DataDomainService.detect_file_format(file.filename, file.content_type)
        
        # Upload file to storage
        file_path = await file_storage.upload_file(
            file_content, file.filename, tenant_id, file.content_type
        )
        
        # Create dataset
        dataset = Dataset.create_new_dataset(
            name=name,
            description=description or "",
            tenant_id=tenant_id,
            created_by=user_id,
            file_path=file_path,
            file_format=file_format,
            file_size=file_size
        )
        
        # Save dataset
        saved_dataset = await dataset_repo.save(dataset)
        
        # Schedule background processing
        background_tasks.add_task(
            process_dataset_background,
            saved_dataset.id,
            file_content,
            file_format,
            user_id
        )
        
        return FileUploadResponse(
            dataset_id=saved_dataset.id,
            file_path=file_path,
            file_size=file_size,
            file_format=file_format,
            message="File uploaded successfully. Processing started in background."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_dataset_background(
    dataset_id: UUID,
    file_content: bytes,
    file_format: DataFormat,
    user_id: UUID
):
    """Background task to process uploaded dataset."""
    try:
        container = get_configured_container()
        dataset_repo = container.dataset_repository()
        
        # Get dataset
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            return
        
        # Mark as processing
        dataset.mark_processing()
        await dataset_repo.save(dataset)
        
        # Process file
        df = await DataFormatProcessor.process_file(file_content, file_format)
        
        # Detect schema
        schema = await SchemaDetector.detect_schema(df)
        dataset.update_schema(schema)
        
        # Validate data quality
        quality_report = await DataValidator.validate_data(df, schema)
        dataset.update_quality_report(quality_report)
        
        # Mark as processed
        dataset.mark_processed()
        await dataset_repo.save(dataset)
        
        # Record initial lineage (source dataset with no parents)
        lineage_service = DataLineageService(
            dataset_repo,
            container.data_processing_job_repository(),
            container.file_storage_repository()
        )
        
        await lineage_service.record_dataset_creation(
            dataset=dataset,
            source_datasets=[],  # No source datasets for uploaded files
            transformations=[],  # No transformations for initial upload
            created_by=user_id
        )
        
    except Exception as e:
        # Mark as failed
        if dataset:
            dataset.mark_failed(str(e))
            await dataset_repo.save(dataset)


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """List datasets for tenant."""
    try:
        if status:
            datasets = await dataset_repo.get_by_status(tenant_id, status)
            # Apply pagination manually for status filter
            paginated_datasets = datasets[offset:offset + limit]
            total = len(datasets)
        else:
            datasets = await dataset_repo.get_by_tenant(tenant_id, limit, offset)
            # For total count, we'd need a separate count query in real implementation
            total = len(datasets) + offset  # Approximation
        
        return DatasetListResponse(
            datasets=[DatasetResponse.from_orm(ds) for ds in datasets],
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """Get dataset by ID."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return DatasetResponse.from_orm(dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")


@router.delete("/datasets/{dataset_id}", response_model=SuccessResponse)
async def delete_dataset(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Delete dataset."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete file from storage
        await file_storage.delete_file(dataset.file_path)
        
        # Delete dataset record
        success = await dataset_repo.delete(dataset_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete dataset")
        
        return SuccessResponse(message="Dataset deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.get("/datasets/{dataset_id}/schema", response_model=DataSchemaResponse)
async def get_dataset_schema(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """Get dataset schema."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not dataset.schema:
            raise HTTPException(status_code=404, detail="Schema not available. Dataset may still be processing.")
        
        return DataSchemaResponse(**dataset.schema)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@router.get("/datasets/{dataset_id}/quality", response_model=DataQualityReportResponse)
async def get_data_quality_report(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """Get data quality report for dataset."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not dataset.metadata or "quality_report" not in dataset.metadata:
            raise HTTPException(status_code=404, detail="Quality report not available. Dataset may still be processing.")
        
        quality_data = dataset.metadata["quality_report"]
        return DataQualityReportResponse(**quality_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality report: {str(e)}")


@router.get("/datasets/{dataset_id}/profile", response_model=DataProfileResponse)
async def get_data_profile(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Get data profile for dataset."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load and process file
        file_content = await file_storage.download_file(dataset.file_path)
        file_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        df = await DataFormatProcessor.process_file(file_content, file_format)
        profile = await DataProfiler.profile_data(df)
        
        return DataProfileResponse(**profile)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate profile: {str(e)}")


@router.post("/datasets/search", response_model=DatasetListResponse)
async def search_datasets(
    search_request: DataSearchRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """Search datasets."""
    try:
        datasets = await dataset_repo.search_datasets(
            tenant_id=tenant_id,
            query=search_request.query,
            limit=search_request.limit,
            offset=search_request.offset
        )
        
        return DatasetListResponse(
            datasets=[DatasetResponse.from_orm(ds) for ds in datasets],
            total=len(datasets) + search_request.offset,  # Approximation
            limit=search_request.limit,
            offset=search_request.offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Download dataset file."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file content
        file_content = await file_storage.download_file(dataset.file_path)
        
        # Determine content type
        file_format = dataset.metadata.get("file_format", "csv")
        content_type_map = {
            "csv": "text/csv",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "json": "application/json",
            "xml": "application/xml",
            "tsv": "text/tab-separated-values",
            "parquet": "application/octet-stream"
        }
        content_type = content_type_map.get(file_format, "application/octet-stream")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={dataset.name}.{file_format}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/stats", response_model=TenantStatsResponse)
async def get_tenant_stats(
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository)
):
    """Get tenant statistics."""
    try:
        stats = await dataset_repo.get_tenant_stats(tenant_id)
        return TenantStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/datasets/{dataset_id}/validate", response_model=DataQualityReportResponse)
async def validate_dataset(
    dataset_id: UUID,
    validation_request: DataValidationRequest,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Validate dataset with custom rules."""
    try:
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Verify tenant access
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load and process file
        file_content = await file_storage.download_file(dataset.file_path)
        file_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        df = await DataFormatProcessor.process_file(file_content, file_format)
        
        # Use provided schema or existing schema
        schema = None
        if validation_request.schema:
            from ..domain.entities import DataSchema
            schema = DataSchema(**validation_request.schema.dict())
        elif dataset.schema:
            from ..domain.entities import DataSchema
            schema = DataSchema(**dataset.schema)
        
        # Validate data
        quality_report = await DataValidator.validate_data(df, schema)
        
        return DataQualityReportResponse(**quality_report.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/datasets/{dataset_id}/process", response_model=DataProcessingJobResponse)
async def process_dataset_with_pipeline(
    dataset_id: UUID,
    processing_options: Optional[Dict[str, Any]] = None,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    job_repo: DataProcessingJobRepository = Depends(get_job_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Process dataset through the unified processing pipeline."""
    try:
        # Verify dataset exists and belongs to tenant
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create processing pipeline
        pipeline = DataProcessingPipeline(dataset_repo, job_repo, file_storage)
        
        # Start processing
        job = await pipeline.process_dataset(dataset_id, processing_options)
        
        return DataProcessingJobResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/jobs/{job_id}", response_model=DataProcessingJobResponse)
async def get_processing_job(
    job_id: UUID,
    job_repo: DataProcessingJobRepository = Depends(get_job_repository)
):
    """Get processing job status."""
    try:
        job_manager = ProcessingJobManager(job_repo)
        job = await job_manager.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return DataProcessingJobResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.post("/jobs/{job_id}/cancel", response_model=SuccessResponse)
async def cancel_processing_job(
    job_id: UUID,
    job_repo: DataProcessingJobRepository = Depends(get_job_repository)
):
    """Cancel a running processing job."""
    try:
        job_manager = ProcessingJobManager(job_repo)
        success = await job_manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled (not running or not found)")
        
        return SuccessResponse(message="Job cancelled successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.post("/jobs/{job_id}/retry", response_model=DataProcessingJobResponse)
async def retry_processing_job(
    job_id: UUID,
    job_repo: DataProcessingJobRepository = Depends(get_job_repository)
):
    """Retry a failed processing job."""
    try:
        job_manager = ProcessingJobManager(job_repo)
        new_job = await job_manager.retry_failed_job(job_id)
        
        if not new_job:
            raise HTTPException(status_code=400, detail="Job cannot be retried (not failed or not found)")
        
        return DataProcessingJobResponse.from_orm(new_job)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {str(e)}")


@router.get("/datasets/{dataset_id}/jobs", response_model=List[DataProcessingJobResponse])
async def get_dataset_jobs(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    job_repo: DataProcessingJobRepository = Depends(get_job_repository)
):
    """Get all processing jobs for a dataset."""
    try:
        # Verify dataset exists and belongs to tenant
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get jobs
        job_manager = ProcessingJobManager(job_repo)
        jobs = await job_manager.get_jobs_by_dataset(dataset_id)
        
        return [DataProcessingJobResponse.from_orm(job) for job in jobs]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")


@router.post("/datasets/{dataset_id}/process-format", response_model=Dict[str, Any])
async def process_with_format_options(
    dataset_id: UUID,
    format_options: Dict[str, Any],
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Process dataset with specific format options."""
    try:
        # Verify dataset exists and belongs to tenant
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file content
        file_content = await file_storage.download_file(dataset.file_path)
        file_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        # Process with custom options
        df = await DataFormatProcessor.process_file(file_content, file_format, **format_options)
        
        # Generate quick analysis
        analysis = {
            "rows_processed": len(df),
            "columns_processed": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(5).to_dict('records') if len(df) > 0 else [],
            "format_options_used": format_options
        }
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Format processing failed: {str(e)}")


@router.get("/formats", response_model=Dict[str, Any])
async def get_supported_formats():
    """Get information about supported file formats."""
    formats_info = {
        "csv": {
            "name": "Comma Separated Values",
            "extensions": [".csv"],
            "options": {
                "delimiter": "Field delimiter (default: ',')",
                "encoding": "File encoding (auto-detected if not specified)",
                "header": "Row number to use as column names (default: 0)",
                "na_values": "Additional strings to recognize as NA/NaN",
                "skipinitialspace": "Skip spaces after delimiter (default: True)",
                "skip_blank_lines": "Skip blank lines (default: True)"
            }
        },
        "excel": {
            "name": "Microsoft Excel",
            "extensions": [".xlsx", ".xls"],
            "options": {
                "sheet_name": "Sheet name or index to read (default: first sheet)",
                "header": "Row number to use as column names (default: 0)",
                "skiprows": "Rows to skip at the beginning",
                "nrows": "Number of rows to read",
                "na_values": "Additional strings to recognize as NA/NaN"
            }
        },
        "json": {
            "name": "JavaScript Object Notation",
            "extensions": [".json"],
            "options": {
                "encoding": "File encoding (auto-detected if not specified)"
            }
        },
        "xml": {
            "name": "Extensible Markup Language",
            "extensions": [".xml"],
            "options": {
                "encoding": "File encoding (auto-detected if not specified)"
            }
        },
        "tsv": {
            "name": "Tab Separated Values",
            "extensions": [".tsv", ".txt"],
            "options": {
                "encoding": "File encoding (auto-detected if not specified)",
                "header": "Row number to use as column names (default: 0)",
                "na_values": "Additional strings to recognize as NA/NaN",
                "skipinitialspace": "Skip spaces after delimiter (default: True)"
            }
        },
        "parquet": {
            "name": "Apache Parquet",
            "extensions": [".parquet"],
            "options": {}
        }
    }
    
    return {
        "supported_formats": list(formats_info.keys()),
        "format_details": formats_info,
        "total_formats": len(formats_info)
    }


@router.post("/datasets/{dataset_id}/convert", response_model=Dict[str, Any])
async def convert_dataset_format(
    dataset_id: UUID,
    target_format: DataFormat,
    conversion_options: Optional[Dict[str, Any]] = None,
    tenant_id: UUID = Depends(get_current_tenant_id),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    file_storage: FileStorageRepository = Depends(get_file_storage)
):
    """Convert dataset to a different format."""
    try:
        # Verify dataset exists and belongs to tenant
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get current data
        file_content = await file_storage.download_file(dataset.file_path)
        current_format = DataFormat(dataset.metadata.get("file_format", "csv"))
        
        # Load data
        df = await DataFormatProcessor.process_file(file_content, current_format)
        
        # Convert to target format
        converted_content = await DataFormatConverter.convert_dataframe(
            df, target_format, conversion_options or {}
        )
        
        # Save converted file
        new_filename = f"{dataset.name}_converted.{target_format.value}"
        new_file_path = await file_storage.upload_file(
            converted_content, new_filename, tenant_id
        )
        
        return {
            "original_format": current_format.value,
            "target_format": target_format.value,
            "converted_file_path": new_file_path,
            "original_size": len(file_content),
            "converted_size": len(converted_content),
            "rows": len(df),
            "columns": len(df.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Format conversion failed: {str(e)}")


# ===== TRANSFORMATION ENDPOINTS =====

@router.post("/datasets/{dataset_id}/transform", response_model=Dict[str, Any])
async def transform_dataset(
    dataset_id: UUID,
    transformations: List[Dict[str, Any]],
    create_new_dataset: bool = False,
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Apply transformations to a dataset."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Apply transformations
        result, new_dataset = await transformation_service.apply_transformations(
            dataset_id=dataset_id,
            transformations=transformations,
            save_result=True,
            create_new_dataset=create_new_dataset
        )
        
        response = result.to_dict()
        if new_dataset:
            response["new_dataset_id"] = str(new_dataset.id)
            response["new_dataset_name"] = new_dataset.name
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")


@router.post("/datasets/{dataset_id}/transform/preview", response_model=Dict[str, Any])
async def preview_transformation(
    dataset_id: UUID,
    transformations: List[Dict[str, Any]],
    sample_size: int = 1000,
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Preview transformation results on a sample of data."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get preview
        preview = await transformation_service.preview_transformation(
            dataset_id=dataset_id,
            transformations=transformations,
            sample_size=sample_size
        )
        
        return preview
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.post("/datasets/{dataset_id}/transform/validate", response_model=Dict[str, Any])
async def validate_transformation_pipeline(
    dataset_id: UUID,
    transformations: List[Dict[str, Any]],
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Validate a transformation pipeline without executing it."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validate pipeline
        is_valid, errors = await transformation_service.validate_transformation_pipeline(
            dataset_id=dataset_id,
            transformations=transformations
        )
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "transformations_count": len(transformations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/datasets/{dataset_id}/transform/suggestions", response_model=List[Dict[str, Any]])
async def get_transformation_suggestions(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Get transformation suggestions based on data quality analysis."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get suggestions
        suggestions = await transformation_service.get_transformation_suggestions(dataset_id)
        
        return suggestions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/datasets/{dataset_id}/transform/history", response_model=List[Dict[str, Any]])
async def get_transformation_history(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Get transformation history for a dataset."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get history
        history = await transformation_service.get_transformation_history(dataset_id)
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/datasets/{dataset_id}/transform/job", response_model=DataProcessingJobResponse)
async def create_transformation_job(
    dataset_id: UUID,
    transformations: List[Dict[str, Any]],
    create_new_dataset: bool = False,
    tenant_id: UUID = Depends(get_current_tenant_id),
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Create a background transformation job."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create job
        job = await transformation_service.create_transformation_job(
            dataset_id=dataset_id,
            transformations=transformations,
            job_parameters={"create_new_dataset": create_new_dataset}
        )
        
        return DataProcessingJobResponse.from_orm(job)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/transformations", response_model=Dict[str, Any])
async def get_available_transformations(
    transformation_service: DataTransformationService = Depends(get_transformation_service)
):
    """Get information about available transformations."""
    try:
        transformations = await transformation_service.get_available_transformations()
        return transformations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transformations: {str(e)}")


# ===== DATA LINEAGE ENDPOINTS =====

@router.get("/datasets/{dataset_id}/lineage", response_model=Dict[str, Any])
async def get_dataset_lineage(
    dataset_id: UUID,
    direction: str = Query(default="both", pattern="^(upstream|downstream|both)$"),
    max_depth: int = Query(default=10, ge=1, le=20),
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Get lineage information for a specific dataset."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get lineage
        lineage = await lineage_service.get_dataset_lineage(
            dataset_id=dataset_id,
            direction=direction,
            max_depth=max_depth
        )
        
        return lineage
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lineage: {str(e)}")


@router.get("/datasets/{dataset_id}/lineage/impact", response_model=Dict[str, Any])
async def get_dataset_impact_analysis(
    dataset_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Get impact analysis for a dataset."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get impact analysis
        impact = await lineage_service.get_impact_analysis(dataset_id)
        
        return impact
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get impact analysis: {str(e)}")


@router.get("/lineage/path", response_model=Dict[str, Any])
async def get_lineage_path(
    source_dataset_id: UUID = Query(...),
    target_dataset_id: UUID = Query(...),
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Get the lineage path between two datasets."""
    try:
        # Validate both datasets exist and belong to tenant
        dataset_repo = get_dataset_repository()
        
        source_dataset = await dataset_repo.get_by_id(source_dataset_id)
        if not source_dataset:
            raise HTTPException(status_code=404, detail="Source dataset not found")
        
        target_dataset = await dataset_repo.get_by_id(target_dataset_id)
        if not target_dataset:
            raise HTTPException(status_code=404, detail="Target dataset not found")
        
        if source_dataset.tenant_id != tenant_id or target_dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get lineage path
        path = await lineage_service.get_lineage_path(source_dataset_id, target_dataset_id)
        
        if not path:
            return {
                "source_dataset_id": str(source_dataset_id),
                "target_dataset_id": str(target_dataset_id),
                "path_found": False,
                "message": "No lineage path found between the datasets"
            }
        
        path["path_found"] = True
        return path
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lineage path: {str(e)}")


@router.get("/lineage/statistics", response_model=Dict[str, Any])
async def get_lineage_statistics(
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Get lineage statistics for the tenant."""
    try:
        stats = await lineage_service.get_lineage_statistics(tenant_id)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lineage statistics: {str(e)}")


@router.post("/lineage/search", response_model=List[Dict[str, Any]])
async def search_lineage(
    query: str = Query(..., min_length=1),
    search_type: str = Query(default="dataset_name", pattern="^(dataset_name|transformation_type|metadata)$"),
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Search lineage graph by various criteria."""
    try:
        results = await lineage_service.search_lineage(
            tenant_id=tenant_id,
            query=query,
            search_type=search_type
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lineage search failed: {str(e)}")


@router.get("/lineage/visualization", response_model=Dict[str, Any])
async def get_lineage_visualization_data(
    dataset_id: Optional[UUID] = Query(default=None),
    max_nodes: int = Query(default=100, ge=10, le=500),
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Get data formatted for lineage visualization."""
    try:
        # If dataset_id provided, validate access
        if dataset_id:
            dataset_repo = get_dataset_repository()
            dataset = await dataset_repo.get_by_id(dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            if dataset.tenant_id != tenant_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get visualization data
        vis_data = await lineage_service.get_lineage_visualization_data(
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            max_nodes=max_nodes
        )
        
        return vis_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get visualization data: {str(e)}")


@router.post("/lineage/refresh", response_model=SuccessResponse)
async def refresh_lineage_cache(
    tenant_id: UUID = Depends(get_current_tenant_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Refresh lineage cache for the tenant."""
    try:
        await lineage_service.refresh_lineage_cache(tenant_id)
        
        return SuccessResponse(message="Lineage cache refreshed successfully")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh lineage cache: {str(e)}")


@router.post("/datasets/{dataset_id}/lineage/record", response_model=Dict[str, Any])
async def record_dataset_lineage(
    dataset_id: UUID,
    source_datasets: List[UUID] = [],
    transformations: List[Dict[str, Any]] = [],
    tenant_id: UUID = Depends(get_current_tenant_id),
    user_id: UUID = Depends(get_current_user_id),
    lineage_service: DataLineageService = Depends(get_lineage_service)
):
    """Manually record lineage for a dataset."""
    try:
        # Validate dataset access
        dataset_repo = get_dataset_repository()
        dataset = await dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validate source datasets belong to tenant
        for source_id in source_datasets:
            source_dataset = await dataset_repo.get_by_id(source_id)
            if not source_dataset or source_dataset.tenant_id != tenant_id:
                raise HTTPException(status_code=400, detail=f"Invalid source dataset: {source_id}")
        
        # Convert transformation data to DataTransformation objects
        from ..domain.entities import DataTransformation, TransformationType
        transformation_objects = []
        for trans_data in transformations:
            transformation = DataTransformation(
                transformation_type=TransformationType(trans_data["transformation_type"]),
                parameters=trans_data.get("parameters", {}),
                description=trans_data.get("description")
            )
            transformation_objects.append(transformation)
        
        # Record lineage
        lineage = await lineage_service.record_dataset_creation(
            dataset=dataset,
            source_datasets=source_datasets,
            transformations=transformation_objects,
            created_by=user_id
        )
        
        return {
            "dataset_id": str(dataset_id),
            "lineage_recorded": True,
            "source_datasets_count": len(source_datasets),
            "transformations_count": len(transformations),
            "created_at": lineage.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record lineage: {str(e)}")