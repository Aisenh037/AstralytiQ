"""
Data service repository interfaces.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from .entities import Dataset, DataProcessingJob


class DatasetRepository(ABC):
    """Repository interface for Dataset entities."""
    
    @abstractmethod
    async def save(self, dataset: Dataset) -> Dataset:
        """Save dataset."""
        pass
    
    @abstractmethod
    async def get_by_id(self, dataset_id: UUID) -> Optional[Dataset]:
        """Get dataset by ID."""
        pass
    
    @abstractmethod
    async def get_by_tenant(self, tenant_id: UUID, limit: int = 100, offset: int = 0) -> List[Dataset]:
        """Get datasets by tenant."""
        pass
    
    @abstractmethod
    async def get_by_name(self, tenant_id: UUID, name: str) -> Optional[Dataset]:
        """Get dataset by name within tenant."""
        pass
    
    @abstractmethod
    async def search_datasets(
        self,
        tenant_id: UUID,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        """Search datasets by name or description."""
        pass
    
    @abstractmethod
    async def get_by_status(self, tenant_id: UUID, status: str) -> List[Dataset]:
        """Get datasets by status."""
        pass
    
    @abstractmethod
    async def delete(self, dataset_id: UUID) -> bool:
        """Delete dataset."""
        pass
    
    @abstractmethod
    async def get_tenant_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get dataset statistics for tenant."""
        pass


class DataProcessingJobRepository(ABC):
    """Repository interface for DataProcessingJob entities."""
    
    @abstractmethod
    async def save(self, job: DataProcessingJob) -> DataProcessingJob:
        """Save processing job."""
        pass
    
    @abstractmethod
    async def get_by_id(self, job_id: UUID) -> Optional[DataProcessingJob]:
        """Get job by ID."""
        pass
    
    @abstractmethod
    async def get_by_dataset(self, dataset_id: UUID) -> List[DataProcessingJob]:
        """Get jobs for dataset."""
        pass
    
    @abstractmethod
    async def get_pending_jobs(self, limit: int = 10) -> List[DataProcessingJob]:
        """Get pending jobs for processing."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str, limit: int = 100) -> List[DataProcessingJob]:
        """Get jobs by status."""
        pass
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
        """Delete job."""
        pass


class FileStorageRepository(ABC):
    """Repository interface for file storage operations."""
    
    @abstractmethod
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: UUID,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file and return file path."""
        pass
    
    @abstractmethod
    async def download_file(self, file_path: str) -> bytes:
        """Download file content."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file."""
        pass
    
    @abstractmethod
    async def update_file(self, file_path: str, file_content: bytes) -> bool:
        """Update existing file content."""
        pass
    
    @abstractmethod
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        pass
    
    @abstractmethod
    async def list_files(self, tenant_id: UUID, prefix: str = "") -> List[Dict[str, Any]]:
        """List files for tenant."""
        pass