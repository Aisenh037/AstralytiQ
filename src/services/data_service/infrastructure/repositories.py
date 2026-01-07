"""
Data service repository implementations.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import os
import aiofiles
from pathlib import Path
from datetime import datetime

from ..domain.repositories import DatasetRepository, DataProcessingJobRepository, FileStorageRepository
from ..domain.entities import Dataset, DataProcessingJob
from src.shared.infrastructure.repositories import SQLAlchemyRepository


class SQLDatasetRepository(SQLAlchemyRepository[Dataset], DatasetRepository):
    """SQLAlchemy implementation of DatasetRepository."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Dataset)
    
    async def get_by_tenant(self, tenant_id: UUID, limit: int = 100, offset: int = 0) -> List[Dataset]:
        """Get datasets by tenant."""
        query = (
            select(Dataset)
            .where(Dataset.tenant_id == tenant_id)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_name(self, tenant_id: UUID, name: str) -> Optional[Dataset]:
        """Get dataset by name within tenant."""
        query = select(Dataset).where(
            and_(Dataset.tenant_id == tenant_id, Dataset.name == name)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def search_datasets(
        self,
        tenant_id: UUID,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        """Search datasets by name or description."""
        search_query = (
            select(Dataset)
            .where(
                and_(
                    Dataset.tenant_id == tenant_id,
                    or_(
                        Dataset.name.ilike(f"%{query}%"),
                        Dataset.description.ilike(f"%{query}%")
                    )
                )
            )
            .order_by(Dataset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(search_query)
        return list(result.scalars().all())
    
    async def get_by_status(self, tenant_id: UUID, status: str) -> List[Dataset]:
        """Get datasets by status."""
        query = select(Dataset).where(
            and_(Dataset.tenant_id == tenant_id, Dataset.status == status)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_tenant_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get dataset statistics for tenant."""
        # Total datasets
        total_query = select(func.count(Dataset.id)).where(Dataset.tenant_id == tenant_id)
        total_result = await self.session.execute(total_query)
        total_datasets = total_result.scalar() or 0
        
        # Datasets by status
        status_query = (
            select(Dataset.status, func.count(Dataset.id))
            .where(Dataset.tenant_id == tenant_id)
            .group_by(Dataset.status)
        )
        status_result = await self.session.execute(status_query)
        status_counts = dict(status_result.all())
        
        # Total file size
        size_query = select(func.sum(Dataset.file_size)).where(Dataset.tenant_id == tenant_id)
        size_result = await self.session.execute(size_query)
        total_size = size_result.scalar() or 0
        
        return {
            "total_datasets": total_datasets,
            "status_counts": status_counts,
            "total_file_size": total_size,
            "average_file_size": total_size / total_datasets if total_datasets > 0 else 0
        }


class InMemoryDataProcessingJobRepository(DataProcessingJobRepository):
    """In-memory implementation of DataProcessingJobRepository."""
    
    def __init__(self):
        self._jobs: Dict[UUID, DataProcessingJob] = {}
    
    async def save(self, job: DataProcessingJob) -> DataProcessingJob:
        """Save processing job."""
        self._jobs[job.id] = job
        return job
    
    async def get_by_id(self, job_id: UUID) -> Optional[DataProcessingJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    async def get_by_dataset(self, dataset_id: UUID) -> List[DataProcessingJob]:
        """Get jobs for dataset."""
        return [job for job in self._jobs.values() if job.dataset_id == dataset_id]
    
    async def get_pending_jobs(self, limit: int = 10) -> List[DataProcessingJob]:
        """Get pending jobs for processing."""
        pending_jobs = [job for job in self._jobs.values() if job.status == "pending"]
        return sorted(pending_jobs, key=lambda x: x.created_at)[:limit]
    
    async def get_by_status(self, status: str, limit: int = 100) -> List[DataProcessingJob]:
        """Get jobs by status."""
        jobs = [job for job in self._jobs.values() if job.status == status]
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)[:limit]
    
    async def delete(self, job_id: UUID) -> bool:
        """Delete job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


class LocalFileStorageRepository(FileStorageRepository):
    """Local file system implementation of FileStorageRepository."""
    
    def __init__(self, base_path: str = "data/uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: UUID,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file and return file path."""
        # Create tenant directory
        tenant_dir = self.base_path / str(tenant_id)
        tenant_dir.mkdir(exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        file_path = tenant_dir / unique_filename
        
        # Write file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Return relative path
        return str(file_path.relative_to(self.base_path))
    
    async def download_file(self, file_path: str) -> bytes:
        """Download file content."""
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file."""
        full_path = self.base_path / file_path
        
        if full_path.exists():
            full_path.unlink()
            return True
        return False
    
    async def update_file(self, file_path: str, file_content: bytes) -> bool:
        """Update existing file content."""
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return False
        
        # Write new content
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(file_content)
        
        return True
    
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return None
        
        stat = full_path.stat()
        return {
            "path": file_path,
            "size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "is_file": full_path.is_file()
        }
    
    async def list_files(self, tenant_id: UUID, prefix: str = "") -> List[Dict[str, Any]]:
        """List files for tenant."""
        tenant_dir = self.base_path / str(tenant_id)
        
        if not tenant_dir.exists():
            return []
        
        files = []
        for file_path in tenant_dir.glob(f"{prefix}*"):
            if file_path.is_file():
                stat = file_path.stat()
                relative_path = str(file_path.relative_to(self.base_path))
                files.append({
                    "path": relative_path,
                    "name": file_path.name,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime)
                })
        
        return sorted(files, key=lambda x: x["created_at"], reverse=True)