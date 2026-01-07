"""
Data lineage tracking service for comprehensive data provenance and impact analysis.
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from uuid import UUID, uuid4
from datetime import datetime
from collections import defaultdict, deque

from ..domain.entities import (
    Dataset, DataLineage, DataTransformation, TransformationType,
    DataProcessingJob, DataDomainService
)
from ..domain.repositories import (
    DatasetRepository, DataProcessingJobRepository, FileStorageRepository
)


class LineageNode:
    """Represents a node in the data lineage graph."""
    
    def __init__(
        self,
        dataset_id: UUID,
        dataset_name: str,
        node_type: str = "dataset",  # dataset, transformation, job
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.node_type = node_type
        self.metadata = metadata or {}
        self.parents: List['LineageNode'] = []
        self.children: List['LineageNode'] = []
        self.transformations: List[DataTransformation] = []
    
    def add_parent(self, parent: 'LineageNode') -> None:
        """Add a parent node."""
        if parent not in self.parents:
            self.parents.append(parent)
            parent.children.append(self)
    
    def add_transformation(self, transformation: DataTransformation) -> None:
        """Add a transformation to this node."""
        self.transformations.append(transformation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "dataset_id": str(self.dataset_id),
            "dataset_name": self.dataset_name,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "parent_ids": [str(p.dataset_id) for p in self.parents],
            "child_ids": [str(c.dataset_id) for c in self.children],
            "transformations": [
                {
                    "type": t.transformation_type.value,
                    "parameters": t.parameters,
                    "description": t.description
                }
                for t in self.transformations
            ]
        }


class LineageGraph:
    """Represents the complete data lineage graph."""
    
    def __init__(self):
        self.nodes: Dict[UUID, LineageNode] = {}
        self.edges: List[Tuple[UUID, UUID]] = []
    
    def add_node(self, node: LineageNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.dataset_id] = node
    
    def add_edge(self, source_id: UUID, target_id: UUID) -> None:
        """Add an edge between two nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            target_node.add_parent(source_node)
            
            edge = (source_id, target_id)
            if edge not in self.edges:
                self.edges.append(edge)
    
    def get_ancestors(self, dataset_id: UUID, max_depth: int = 10) -> List[LineageNode]:
        """Get all ancestor nodes of a dataset."""
        if dataset_id not in self.nodes:
            return []
        
        ancestors = []
        visited = set()
        queue = deque([(self.nodes[dataset_id], 0)])
        
        while queue:
            node, depth = queue.popleft()
            
            if node.dataset_id in visited or depth >= max_depth:
                continue
            
            visited.add(node.dataset_id)
            
            for parent in node.parents:
                if parent.dataset_id not in visited:
                    ancestors.append(parent)
                    queue.append((parent, depth + 1))
        
        return ancestors
    
    def get_descendants(self, dataset_id: UUID, max_depth: int = 10) -> List[LineageNode]:
        """Get all descendant nodes of a dataset."""
        if dataset_id not in self.nodes:
            return []
        
        descendants = []
        visited = set()
        queue = deque([(self.nodes[dataset_id], 0)])
        
        while queue:
            node, depth = queue.popleft()
            
            if node.dataset_id in visited or depth >= max_depth:
                continue
            
            visited.add(node.dataset_id)
            
            for child in node.children:
                if child.dataset_id not in visited:
                    descendants.append(child)
                    queue.append((child, depth + 1))
        
        return descendants
    
    def get_lineage_path(self, source_id: UUID, target_id: UUID) -> Optional[List[LineageNode]]:
        """Get the lineage path between two datasets."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        # BFS to find path
        queue = deque([(self.nodes[source_id], [self.nodes[source_id]])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            
            if node.dataset_id == target_id:
                return path
            
            if node.dataset_id in visited:
                continue
            
            visited.add(node.dataset_id)
            
            for child in node.children:
                if child.dataset_id not in visited:
                    queue.append((child, path + [child]))
        
        return None
    
    def get_impact_analysis(self, dataset_id: UUID) -> Dict[str, Any]:
        """Get impact analysis for a dataset."""
        if dataset_id not in self.nodes:
            return {"error": "Dataset not found in lineage"}
        
        node = self.nodes[dataset_id]
        descendants = self.get_descendants(dataset_id)
        
        # Analyze impact by transformation types
        transformation_impact = defaultdict(int)
        affected_datasets = set()
        
        for descendant in descendants:
            affected_datasets.add(descendant.dataset_id)
            for transformation in descendant.transformations:
                transformation_impact[transformation.transformation_type.value] += 1
        
        return {
            "source_dataset": {
                "id": str(dataset_id),
                "name": node.dataset_name
            },
            "directly_affected_datasets": len([c for c in node.children]),
            "total_affected_datasets": len(affected_datasets),
            "transformation_types_used": dict(transformation_impact),
            "max_depth": self._calculate_max_depth(dataset_id),
            "affected_dataset_details": [
                {
                    "id": str(d.dataset_id),
                    "name": d.dataset_name,
                    "transformations": len(d.transformations)
                }
                for d in descendants
            ]
        }
    
    def _calculate_max_depth(self, dataset_id: UUID) -> int:
        """Calculate maximum depth from a dataset."""
        if dataset_id not in self.nodes:
            return 0
        
        max_depth = 0
        queue = deque([(self.nodes[dataset_id], 0)])
        visited = set()
        
        while queue:
            node, depth = queue.popleft()
            
            if node.dataset_id in visited:
                continue
            
            visited.add(node.dataset_id)
            max_depth = max(max_depth, depth)
            
            for child in node.children:
                if child.dataset_id not in visited:
                    queue.append((child, depth + 1))
        
        return max_depth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": {str(node_id): node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [(str(source), str(target)) for source, target in self.edges],
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }


class DataLineageService:
    """Service for managing data lineage tracking and analysis."""
    
    def __init__(
        self,
        dataset_repo: DatasetRepository,
        job_repo: DataProcessingJobRepository,
        file_storage: FileStorageRepository
    ):
        self.dataset_repo = dataset_repo
        self.job_repo = job_repo
        self.file_storage = file_storage
        self._lineage_cache: Dict[UUID, LineageGraph] = {}
    
    async def record_dataset_creation(
        self,
        dataset: Dataset,
        source_datasets: Optional[List[UUID]] = None,
        transformations: Optional[List[DataTransformation]] = None,
        created_by: UUID = None
    ) -> DataLineage:
        """Record lineage for dataset creation."""
        
        lineage = DataLineage(
            source_datasets=source_datasets or [],
            transformations=transformations or [],
            created_at=datetime.utcnow(),
            created_by=created_by or dataset.created_by
        )
        
        # Add lineage to dataset metadata
        dataset.add_lineage(lineage)
        await self.dataset_repo.save(dataset)
        
        # Clear cache to force refresh
        self._clear_lineage_cache(dataset.tenant_id)
        
        return lineage
    
    async def record_transformation_lineage(
        self,
        source_dataset_id: UUID,
        target_dataset_id: UUID,
        transformations: List[DataTransformation],
        created_by: UUID,
        job_id: Optional[UUID] = None
    ) -> DataLineage:
        """Record lineage for data transformation."""
        
        # Get target dataset
        target_dataset = await self.dataset_repo.get_by_id(target_dataset_id)
        if not target_dataset:
            raise ValueError(f"Target dataset {target_dataset_id} not found")
        
        # Create lineage record
        lineage = DataLineage(
            source_datasets=[source_dataset_id],
            transformations=transformations,
            created_at=datetime.utcnow(),
            created_by=created_by
        )
        
        # Add job reference if provided
        if job_id:
            lineage_metadata = {
                "job_id": str(job_id),
                "transformation_type": "pipeline"
            }
        else:
            lineage_metadata = {
                "transformation_type": "direct"
            }
        
        # Update target dataset with lineage
        target_dataset.add_lineage(lineage)
        
        # Add transformation metadata
        if not target_dataset.metadata:
            target_dataset.metadata = {}
        
        target_dataset.metadata["lineage_metadata"] = lineage_metadata
        await self.dataset_repo.save(target_dataset)
        
        # Clear cache
        self._clear_lineage_cache(target_dataset.tenant_id)
        
        return lineage
    
    async def record_processing_job_lineage(
        self,
        job: DataProcessingJob,
        source_datasets: List[UUID],
        target_dataset_id: Optional[UUID] = None
    ) -> None:
        """Record lineage for processing jobs."""
        
        # Get job dataset
        job_dataset = await self.dataset_repo.get_by_id(job.dataset_id)
        if not job_dataset:
            return
        
        # Create transformation record for the job
        job_transformation = DataTransformation(
            transformation_type=TransformationType.CLEAN,  # Default type for jobs
            parameters=job.parameters,
            description=f"Processing job: {job.job_type}"
        )
        
        # Record lineage
        await self.record_transformation_lineage(
            source_dataset_id=job.dataset_id,
            target_dataset_id=target_dataset_id or job.dataset_id,
            transformations=[job_transformation],
            created_by=job_dataset.created_by,
            job_id=job.id
        )
    
    async def build_lineage_graph(self, tenant_id: UUID) -> LineageGraph:
        """Build complete lineage graph for a tenant."""
        
        # Check cache first
        if tenant_id in self._lineage_cache:
            return self._lineage_cache[tenant_id]
        
        graph = LineageGraph()
        
        # Get all datasets for tenant
        datasets = await self.dataset_repo.get_by_tenant(tenant_id, limit=1000)
        
        # Create nodes for all datasets
        for dataset in datasets:
            node = LineageNode(
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                node_type="dataset",
                metadata={
                    "status": dataset.status.value,
                    "created_at": dataset.created_at.isoformat(),
                    "file_size": dataset.file_size,
                    "quality_score": dataset.get_quality_score()
                }
            )
            graph.add_node(node)
        
        # Build relationships from lineage metadata
        for dataset in datasets:
            if dataset.metadata and "lineage" in dataset.metadata:
                lineage_data = dataset.metadata["lineage"]
                source_dataset_ids = [
                    UUID(ds_id) for ds_id in lineage_data.get("source_datasets", [])
                ]
                
                # Add transformations to target node
                target_node = graph.nodes.get(dataset.id)
                if target_node:
                    transformations_data = lineage_data.get("transformations", [])
                    for trans_data in transformations_data:
                        transformation = DataTransformation(
                            transformation_type=TransformationType(trans_data["transformation_type"]),
                            parameters=trans_data["parameters"],
                            description=trans_data.get("description")
                        )
                        target_node.add_transformation(transformation)
                
                # Add edges from source datasets
                for source_id in source_dataset_ids:
                    if source_id in graph.nodes:
                        graph.add_edge(source_id, dataset.id)
        
        # Cache the graph
        self._lineage_cache[tenant_id] = graph
        
        return graph
    
    async def get_dataset_lineage(
        self,
        dataset_id: UUID,
        direction: str = "both",  # upstream, downstream, both
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """Get lineage information for a specific dataset."""
        
        # Get dataset to determine tenant
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Build lineage graph
        graph = await self.build_lineage_graph(dataset.tenant_id)
        
        if dataset_id not in graph.nodes:
            return {
                "dataset_id": str(dataset_id),
                "dataset_name": dataset.name,
                "upstream": [],
                "downstream": [],
                "error": "Dataset not found in lineage graph"
            }
        
        result = {
            "dataset_id": str(dataset_id),
            "dataset_name": dataset.name,
            "upstream": [],
            "downstream": []
        }
        
        # Get upstream lineage
        if direction in ["upstream", "both"]:
            ancestors = graph.get_ancestors(dataset_id, max_depth)
            result["upstream"] = [node.to_dict() for node in ancestors]
        
        # Get downstream lineage
        if direction in ["downstream", "both"]:
            descendants = graph.get_descendants(dataset_id, max_depth)
            result["downstream"] = [node.to_dict() for node in descendants]
        
        return result
    
    async def get_lineage_path(
        self,
        source_dataset_id: UUID,
        target_dataset_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get the lineage path between two datasets."""
        
        # Get source dataset to determine tenant
        source_dataset = await self.dataset_repo.get_by_id(source_dataset_id)
        if not source_dataset:
            raise ValueError(f"Source dataset {source_dataset_id} not found")
        
        # Build lineage graph
        graph = await self.build_lineage_graph(source_dataset.tenant_id)
        
        # Find path
        path = graph.get_lineage_path(source_dataset_id, target_dataset_id)
        
        if not path:
            return None
        
        return {
            "source_dataset_id": str(source_dataset_id),
            "target_dataset_id": str(target_dataset_id),
            "path_length": len(path),
            "path": [node.to_dict() for node in path]
        }
    
    async def get_impact_analysis(self, dataset_id: UUID) -> Dict[str, Any]:
        """Get impact analysis for a dataset."""
        
        # Get dataset to determine tenant
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Build lineage graph
        graph = await self.build_lineage_graph(dataset.tenant_id)
        
        # Get impact analysis
        impact = graph.get_impact_analysis(dataset_id)
        
        return impact
    
    async def get_lineage_statistics(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get lineage statistics for a tenant."""
        
        # Build lineage graph
        graph = await self.build_lineage_graph(tenant_id)
        
        # Calculate statistics
        total_datasets = len(graph.nodes)
        total_relationships = len(graph.edges)
        
        # Count datasets by type
        source_datasets = 0  # No parents
        derived_datasets = 0  # Has parents
        sink_datasets = 0    # No children
        
        transformation_counts = defaultdict(int)
        
        for node in graph.nodes.values():
            if not node.parents:
                source_datasets += 1
            if node.parents:
                derived_datasets += 1
            if not node.children:
                sink_datasets += 1
            
            for transformation in node.transformations:
                transformation_counts[transformation.transformation_type.value] += 1
        
        return {
            "total_datasets": total_datasets,
            "total_relationships": total_relationships,
            "source_datasets": source_datasets,
            "derived_datasets": derived_datasets,
            "sink_datasets": sink_datasets,
            "transformation_counts": dict(transformation_counts),
            "average_transformations_per_dataset": (
                sum(transformation_counts.values()) / total_datasets
                if total_datasets > 0 else 0
            )
        }
    
    async def search_lineage(
        self,
        tenant_id: UUID,
        query: str,
        search_type: str = "dataset_name"  # dataset_name, transformation_type, metadata
    ) -> List[Dict[str, Any]]:
        """Search lineage graph by various criteria."""
        
        # Build lineage graph
        graph = await self.build_lineage_graph(tenant_id)
        
        results = []
        query_lower = query.lower()
        
        for node in graph.nodes.values():
            match_found = False
            
            if search_type == "dataset_name":
                if query_lower in node.dataset_name.lower():
                    match_found = True
            
            elif search_type == "transformation_type":
                for transformation in node.transformations:
                    if query_lower in transformation.transformation_type.value.lower():
                        match_found = True
                        break
            
            elif search_type == "metadata":
                # Search in metadata values
                for key, value in node.metadata.items():
                    if isinstance(value, str) and query_lower in value.lower():
                        match_found = True
                        break
            
            if match_found:
                results.append(node.to_dict())
        
        return results
    
    async def get_lineage_visualization_data(
        self,
        tenant_id: UUID,
        dataset_id: Optional[UUID] = None,
        max_nodes: int = 100
    ) -> Dict[str, Any]:
        """Get data formatted for lineage visualization."""
        
        # Build lineage graph
        graph = await self.build_lineage_graph(tenant_id)
        
        # If specific dataset requested, focus on its subgraph
        if dataset_id:
            if dataset_id not in graph.nodes:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Get subgraph around the dataset
            focus_node = graph.nodes[dataset_id]
            ancestors = graph.get_ancestors(dataset_id, max_depth=5)
            descendants = graph.get_descendants(dataset_id, max_depth=5)
            
            # Combine nodes
            relevant_nodes = [focus_node] + ancestors + descendants
            relevant_node_ids = {node.dataset_id for node in relevant_nodes}
            
            # Filter edges to only include relevant ones
            relevant_edges = [
                (source, target) for source, target in graph.edges
                if source in relevant_node_ids and target in relevant_node_ids
            ]
        else:
            # Use full graph but limit nodes
            relevant_nodes = list(graph.nodes.values())[:max_nodes]
            relevant_node_ids = {node.dataset_id for node in relevant_nodes}
            relevant_edges = [
                (source, target) for source, target in graph.edges
                if source in relevant_node_ids and target in relevant_node_ids
            ]
        
        # Format for visualization
        vis_nodes = []
        for node in relevant_nodes:
            vis_node = {
                "id": str(node.dataset_id),
                "label": node.dataset_name,
                "type": node.node_type,
                "metadata": node.metadata,
                "transformations_count": len(node.transformations),
                "parents_count": len(node.parents),
                "children_count": len(node.children)
            }
            
            # Add styling based on node characteristics
            if not node.parents:
                vis_node["category"] = "source"
            elif not node.children:
                vis_node["category"] = "sink"
            else:
                vis_node["category"] = "intermediate"
            
            vis_nodes.append(vis_node)
        
        vis_edges = [
            {
                "source": str(source),
                "target": str(target),
                "type": "lineage"
            }
            for source, target in relevant_edges
        ]
        
        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "total_nodes": len(vis_nodes),
            "total_edges": len(vis_edges),
            "focus_dataset_id": str(dataset_id) if dataset_id else None
        }
    
    def _clear_lineage_cache(self, tenant_id: UUID) -> None:
        """Clear lineage cache for a tenant."""
        if tenant_id in self._lineage_cache:
            del self._lineage_cache[tenant_id]
    
    async def refresh_lineage_cache(self, tenant_id: UUID) -> None:
        """Refresh lineage cache for a tenant."""
        self._clear_lineage_cache(tenant_id)
        await self.build_lineage_graph(tenant_id)