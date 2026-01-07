"""
Data service domain entities and business logic.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from enum import Enum
import json

from src.shared.domain.models import Dataset as BaseDataset, DatasetStatus
from src.shared.domain.base import DomainService, ValueObject


class DataFormat(str, Enum):
    """Supported data formats."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    XML = "xml"
    TSV = "tsv"


class DataQualityIssueType(str, Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    INVALID_FORMAT = "invalid_format"
    OUTLIERS = "outliers"
    INCONSISTENT_TYPES = "inconsistent_types"
    CONSTRAINT_VIOLATION = "constraint_violation"


class TransformationType(str, Enum):
    """Types of data transformations."""
    CLEAN = "clean"
    NORMALIZE = "normalize"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    JOIN = "join"
    PIVOT = "pivot"
    UNPIVOT = "unpivot"
    DERIVE = "derive"


class DataSchema(ValueObject):
    """Data schema definition."""
    columns: List[Dict[str, Any]]
    primary_key: Optional[List[str]] = None
    foreign_keys: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[Dict[str, Any]]] = None
    
    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        return [col["name"] for col in self.columns]
    
    def get_column_types(self) -> Dict[str, str]:
        """Get mapping of column names to types."""
        return {col["name"]: col["type"] for col in self.columns}
    
    def validate_data_types(self, data: Dict[str, Any]) -> List[str]:
        """Validate data against schema types."""
        errors = []
        column_types = self.get_column_types()
        
        for column, expected_type in column_types.items():
            if column in data:
                value = data[column]
                if not self._is_valid_type(value, expected_type):
                    errors.append(f"Column '{column}' expected {expected_type}, got {type(value).__name__}")
        
        return errors
    
    def _is_valid_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if value is None:
            return True  # Allow nulls for now
        
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "datetime": (str, datetime),  # Accept string or datetime
            "date": (str, datetime)
        }
        
        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it


class DataQualityIssue(ValueObject):
    """Data quality issue."""
    issue_type: DataQualityIssueType
    description: str
    severity: str = "medium"  # low, medium, high, critical
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type.value,
            "description": self.description,
            "severity": self.severity,
            "affected_rows": self.affected_rows,
            "affected_columns": self.affected_columns,
            "suggested_fix": self.suggested_fix
        }


class DataQualityReport(ValueObject):
    """Data quality assessment report."""
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows_count: int
    issues: List[DataQualityIssue]
    quality_score: float  # 0-100
    
    def get_issues_by_severity(self, severity: str) -> List[DataQualityIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == "critical" for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values_count": self.missing_values_count,
            "duplicate_rows_count": self.duplicate_rows_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "quality_score": self.quality_score
        }


class DataTransformation(ValueObject):
    """Data transformation definition."""
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    description: Optional[str] = None
    
    def apply_to_data(self, data: Any) -> Any:
        """Apply transformation to data (placeholder)."""
        # This would contain the actual transformation logic
        # For now, return data unchanged
        return data


class DataLineage(ValueObject):
    """Data lineage tracking."""
    source_datasets: List[UUID]
    transformations: List[DataTransformation]
    created_at: datetime
    created_by: UUID
    
    def add_transformation(self, transformation: DataTransformation) -> None:
        """Add a transformation to the lineage."""
        self.transformations.append(transformation)
    
    def get_transformation_chain(self) -> List[str]:
        """Get list of transformation descriptions."""
        return [
            f"{t.transformation_type.value}: {t.description or 'No description'}"
            for t in self.transformations
        ]


class Dataset(BaseDataset):
    """Extended Dataset entity with business logic."""
    
    def get_file_format(self) -> DataFormat:
        """Get the file format of the dataset."""
        if self.metadata and "file_format" in self.metadata:
            return DataFormat(self.metadata["file_format"])
        # Default to CSV if not specified
        return DataFormat.CSV
    
    def get_quality_report(self) -> Optional[DataQualityReport]:
        """Get the quality report if available."""
        if self.metadata and "quality_report" in self.metadata:
            quality_data = self.metadata["quality_report"]
            return DataQualityReport(**quality_data)
        return None
    
    def update_schema(self, schema: DataSchema) -> None:
        """Update dataset schema."""
        self.schema = schema.dict()
        self.updated_at = datetime.utcnow()
    
    def update_quality_report(self, report: DataQualityReport) -> None:
        """Update data quality report."""
        self.metadata = self.metadata or {}
        self.metadata["quality_report"] = report.to_dict()
        self.updated_at = datetime.utcnow()
    
    def add_lineage(self, lineage: DataLineage) -> None:
        """Add data lineage information."""
        self.metadata = self.metadata or {}
        self.metadata["lineage"] = {
            "source_datasets": [str(ds_id) for ds_id in lineage.source_datasets],
            "transformations": [t.dict() for t in lineage.transformations],
            "created_at": lineage.created_at.isoformat(),
            "created_by": str(lineage.created_by)
        }
        self.updated_at = datetime.utcnow()
    
    def mark_processing(self) -> None:
        """Mark dataset as processing."""
        self.status = DatasetStatus.PROCESSING
        self.updated_at = datetime.utcnow()
    
    def mark_processed(self) -> None:
        """Mark dataset as processed successfully."""
        self.status = DatasetStatus.PROCESSED
        self.updated_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str) -> None:
        """Mark dataset processing as failed."""
        self.status = DatasetStatus.FAILED
        self.metadata = self.metadata or {}
        self.metadata["error"] = error_message
        self.updated_at = datetime.utcnow()
    
    def get_quality_score(self) -> Optional[float]:
        """Get data quality score."""
        if self.metadata and "quality_report" in self.metadata:
            return self.metadata["quality_report"].get("quality_score")
        return None
    
    def get_row_count(self) -> Optional[int]:
        """Get number of rows in dataset."""
        if self.metadata and "quality_report" in self.metadata:
            return self.metadata["quality_report"].get("total_rows")
        return None
    
    def get_column_count(self) -> Optional[int]:
        """Get number of columns in dataset."""
        if self.metadata and "quality_report" in self.metadata:
            return self.metadata["quality_report"].get("total_columns")
        return None
    
    @classmethod
    def create_new_dataset(
        cls,
        name: str,
        description: str,
        tenant_id: UUID,
        created_by: UUID,
        file_path: str,
        file_format: DataFormat,
        file_size: int
    ) -> "Dataset":
        """Create a new dataset."""
        return cls(
            name=name,
            description=description,
            tenant_id=tenant_id,
            created_by=created_by,
            file_path=file_path,
            file_size=file_size,
            status=DatasetStatus.UPLOADED,
            metadata={
                "file_format": file_format.value,
                "upload_timestamp": datetime.utcnow().isoformat()
            }
        )


class DataProcessingJob(ValueObject):
    """Data processing job."""
    id: UUID
    dataset_id: UUID
    job_type: str
    status: str = "pending"  # pending, running, completed, failed
    parameters: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start(self) -> None:
        """Mark job as started."""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark job as completed."""
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, error_message: str) -> None:
        """Mark job as failed."""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.utcnow()


class DataDomainService(DomainService):
    """Domain service for data-related business logic."""
    
    @staticmethod
    def detect_file_format(filename: str, content_type: Optional[str] = None) -> DataFormat:
        """Detect file format from filename and content type."""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.csv'):
            return DataFormat.CSV
        elif filename_lower.endswith(('.xlsx', '.xls')):
            return DataFormat.EXCEL
        elif filename_lower.endswith('.json'):
            return DataFormat.JSON
        elif filename_lower.endswith('.parquet'):
            return DataFormat.PARQUET
        elif filename_lower.endswith('.xml'):
            return DataFormat.XML
        elif filename_lower.endswith('.tsv'):
            return DataFormat.TSV
        
        # Fallback to content type
        if content_type:
            if 'csv' in content_type:
                return DataFormat.CSV
            elif 'json' in content_type:
                return DataFormat.JSON
            elif 'xml' in content_type:
                return DataFormat.XML
        
        # Default to CSV
        return DataFormat.CSV
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 100) -> tuple[bool, Optional[str]]:
        """Validate file size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        
        return True, None
    
    @staticmethod
    def generate_schema_from_sample(sample_data: List[Dict[str, Any]]) -> DataSchema:
        """Generate schema from sample data."""
        if not sample_data:
            return DataSchema(columns=[])
        
        # Get all unique column names
        all_columns = set()
        for row in sample_data:
            all_columns.update(row.keys())
        
        columns = []
        for col_name in sorted(all_columns):
            # Infer type from sample values
            sample_values = [row.get(col_name) for row in sample_data if col_name in row]
            inferred_type = DataDomainService._infer_column_type(sample_values)
            
            columns.append({
                "name": col_name,
                "type": inferred_type,
                "nullable": any(v is None or v == "" for v in sample_values)
            })
        
        return DataSchema(columns=columns)
    
    @staticmethod
    def _infer_column_type(values: List[Any]) -> str:
        """Infer column type from sample values."""
        non_null_values = [v for v in values if v is not None and v != ""]
        
        if not non_null_values:
            return "string"
        
        # Check if all values are integers
        if all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in non_null_values):
            return "integer"
        
        # Check if all values are floats
        try:
            [float(v) for v in non_null_values]
            return "float"
        except (ValueError, TypeError):
            pass
        
        # Check if all values are booleans
        if all(isinstance(v, bool) or str(v).lower() in ['true', 'false', '1', '0'] for v in non_null_values):
            return "boolean"
        
        # Default to string
        return "string"
    
    @staticmethod
    def calculate_quality_score(report: DataQualityReport) -> float:
        """Calculate overall data quality score."""
        if report.total_rows == 0:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Deduct for missing values
        missing_percentage = (report.missing_values_count / (report.total_rows * report.total_columns)) * 100
        score -= missing_percentage * 0.5
        
        # Deduct for duplicates
        duplicate_percentage = (report.duplicate_rows_count / report.total_rows) * 100
        score -= duplicate_percentage * 0.3
        
        # Deduct for issues by severity
        for issue in report.issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
            elif issue.severity == "low":
                score -= 2
        
        return max(0.0, min(100.0, score))
    
    @staticmethod
    def is_dataset_name_valid(name: str) -> tuple[bool, Optional[str]]:
        """Validate dataset name."""
        if not name or len(name.strip()) == 0:
            return False, "Dataset name cannot be empty"
        
        if len(name) > 100:
            return False, "Dataset name cannot exceed 100 characters"
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in name for char in invalid_chars):
            return False, f"Dataset name cannot contain: {', '.join(invalid_chars)}"
        
        return True, None