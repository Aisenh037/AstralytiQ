"""
Data service API schemas.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, validator

from ..domain.entities import DataFormat, DataQualityIssueType, TransformationType


class DatasetCreateRequest(BaseModel):
    """Request schema for creating a dataset."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate dataset name."""
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Dataset name cannot contain: {', '.join(invalid_chars)}")
        
        return v.strip()


class DatasetResponse(BaseModel):
    """Response schema for dataset."""
    id: UUID
    name: str
    description: Optional[str]
    tenant_id: UUID
    created_by: UUID
    file_path: str
    file_size: int
    status: str
    schema: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Response schema for dataset list."""
    datasets: List[DatasetResponse]
    total: int
    limit: int
    offset: int


class DataSchemaColumn(BaseModel):
    """Schema for a data column."""
    name: str
    type: str
    nullable: bool = True
    description: Optional[str] = None


class DataSchemaResponse(BaseModel):
    """Response schema for data schema."""
    columns: List[DataSchemaColumn]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[Dict[str, Any]]] = None


class DataQualityIssueResponse(BaseModel):
    """Response schema for data quality issue."""
    issue_type: DataQualityIssueType
    description: str
    severity: str
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    suggested_fix: Optional[str] = None


class DataQualityReportResponse(BaseModel):
    """Response schema for data quality report."""
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows_count: int
    issues: List[DataQualityIssueResponse]
    quality_score: float


class DataProfileColumnResponse(BaseModel):
    """Response schema for column profile."""
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    quartiles: Optional[Dict[str, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    most_common: Optional[Dict[str, int]] = None
    min_date: Optional[datetime] = None
    max_date: Optional[datetime] = None
    date_range_days: Optional[int] = None


class DataProfileResponse(BaseModel):
    """Response schema for data profile."""
    basic_stats: Dict[str, Any]
    columns: Dict[str, DataProfileColumnResponse]


class DataTransformationRequest(BaseModel):
    """Request schema for data transformation."""
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    description: Optional[str] = None


class TransformationPipelineRequest(BaseModel):
    """Request schema for transformation pipeline."""
    transformations: List[Dict[str, Any]]
    create_new_dataset: bool = False
    description: Optional[str] = None


class TransformationPreviewRequest(BaseModel):
    """Request schema for transformation preview."""
    transformations: List[Dict[str, Any]]
    sample_size: int = 1000


class TransformationValidationRequest(BaseModel):
    """Request schema for transformation validation."""
    transformations: List[Dict[str, Any]]


class TransformationSuggestionResponse(BaseModel):
    """Response schema for transformation suggestions."""
    transformation: str
    reason: str
    suggested_config: Dict[str, Any]
    priority: str


class TransformationHistoryResponse(BaseModel):
    """Response schema for transformation history."""
    transformations: List[Dict[str, Any]]
    execution_summary: List[Dict[str, Any]]
    timestamp: str
    rows_before: Optional[int]
    rows_after: Optional[int]


class TransformationResultResponse(BaseModel):
    """Response schema for transformation results."""
    transformation_summary: Dict[str, Any]
    transformation_config: List[Dict[str, Any]]
    sample_data: Dict[str, Any]
    new_dataset_id: Optional[UUID] = None
    new_dataset_name: Optional[str] = None


class DataProcessingJobResponse(BaseModel):
    """Response schema for data processing job."""
    id: UUID
    dataset_id: UUID
    job_type: str
    status: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class FileUploadResponse(BaseModel):
    """Response schema for file upload."""
    dataset_id: UUID
    file_path: str
    file_size: int
    file_format: DataFormat
    message: str


class DataValidationRequest(BaseModel):
    """Request schema for data validation."""
    schema: Optional[DataSchemaResponse] = None
    validation_rules: Optional[Dict[str, Any]] = None


class DataSearchRequest(BaseModel):
    """Request schema for dataset search."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class TenantStatsResponse(BaseModel):
    """Response schema for tenant statistics."""
    total_datasets: int
    status_counts: Dict[str, int]
    total_file_size: int
    average_file_size: float


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SuccessResponse(BaseModel):
    """Success response schema."""
    message: str
    data: Optional[Dict[str, Any]] = None

class ProcessingOptionsRequest(BaseModel):
    """Request schema for processing options."""
    enable_profiling: bool = True
    schema_sample_size: int = 1000
    parse_options: Dict[str, Any] = Field(default_factory=dict)
    validation_options: Dict[str, Any] = Field(default_factory=dict)


class FormatOptionsRequest(BaseModel):
    """Request schema for format-specific options."""
    format: DataFormat
    options: Dict[str, Any] = Field(default_factory=dict)


class ConversionRequest(BaseModel):
    """Request schema for format conversion."""
    target_format: DataFormat
    conversion_options: Dict[str, Any] = Field(default_factory=dict)


class ProcessingStageResponse(BaseModel):
    """Response schema for processing stage information."""
    stage_name: str
    status: str
    processing_time: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineResultResponse(BaseModel):
    """Response schema for pipeline processing result."""
    dataset_id: UUID
    stages_completed: List[str]
    processing_time: Dict[str, float]
    total_processing_time: float
    status: str
    errors: List[str] = Field(default_factory=list)
    parse_info: Optional[Dict[str, Any]] = None
    schema_info: Optional[Dict[str, Any]] = None
    quality_info: Optional[Dict[str, Any]] = None
    profile_info: Optional[Dict[str, Any]] = None
    format_info: Optional[Dict[str, Any]] = None


class FormatInfoResponse(BaseModel):
    """Response schema for format information."""
    supported_formats: List[str]
    format_details: Dict[str, Any]
    total_formats: int


class ConversionResultResponse(BaseModel):
    """Response schema for format conversion result."""
    original_format: str
    target_format: str
    converted_file_path: str
    original_size: int
    converted_size: int
    rows: int
    columns: int


# ===== DATA LINEAGE SCHEMAS =====

class LineageNodeResponse(BaseModel):
    """Response schema for lineage node."""
    dataset_id: str
    dataset_name: str
    node_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_ids: List[str] = Field(default_factory=list)
    child_ids: List[str] = Field(default_factory=list)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetLineageResponse(BaseModel):
    """Response schema for dataset lineage."""
    dataset_id: str
    dataset_name: str
    upstream: List[LineageNodeResponse] = Field(default_factory=list)
    downstream: List[LineageNodeResponse] = Field(default_factory=list)
    error: Optional[str] = None


class LineagePathResponse(BaseModel):
    """Response schema for lineage path."""
    source_dataset_id: str
    target_dataset_id: str
    path_found: bool
    path_length: Optional[int] = None
    path: Optional[List[LineageNodeResponse]] = None
    message: Optional[str] = None


class ImpactAnalysisResponse(BaseModel):
    """Response schema for impact analysis."""
    source_dataset: Dict[str, str]
    directly_affected_datasets: int
    total_affected_datasets: int
    transformation_types_used: Dict[str, int]
    max_depth: int
    affected_dataset_details: List[Dict[str, Any]]


class LineageStatisticsResponse(BaseModel):
    """Response schema for lineage statistics."""
    total_datasets: int
    total_relationships: int
    source_datasets: int
    derived_datasets: int
    sink_datasets: int
    transformation_counts: Dict[str, int]
    average_transformations_per_dataset: float


class LineageSearchRequest(BaseModel):
    """Request schema for lineage search."""
    query: str = Field(..., min_length=1)
    search_type: str = Field(default="dataset_name", pattern="^(dataset_name|transformation_type|metadata)$")


class LineageVisualizationNode(BaseModel):
    """Schema for lineage visualization node."""
    id: str
    label: str
    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    transformations_count: int
    parents_count: int
    children_count: int
    category: str  # source, intermediate, sink


class LineageVisualizationEdge(BaseModel):
    """Schema for lineage visualization edge."""
    source: str
    target: str
    type: str


class LineageVisualizationResponse(BaseModel):
    """Response schema for lineage visualization data."""
    nodes: List[LineageVisualizationNode]
    edges: List[LineageVisualizationEdge]
    total_nodes: int
    total_edges: int
    focus_dataset_id: Optional[str] = None


class RecordLineageRequest(BaseModel):
    """Request schema for recording lineage."""
    source_datasets: List[UUID] = Field(default_factory=list)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)


class LineageRecordResponse(BaseModel):
    """Response schema for lineage recording."""
    dataset_id: str
    lineage_recorded: bool
    source_datasets_count: int
    transformations_count: int
    created_at: str