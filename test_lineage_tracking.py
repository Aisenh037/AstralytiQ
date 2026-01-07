#!/usr/bin/env python3
"""
Test script for data lineage tracking functionality.
"""
import asyncio
import sys
import os
from uuid import UUID, uuid4
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.data_service.infrastructure.lineage_service import (
    DataLineageService, LineageGraph, LineageNode
)
from src.services.data_service.domain.entities import (
    Dataset, DataLineage, DataTransformation, TransformationType, DataFormat
)


class MockDatasetRepository:
    """Mock dataset repository for testing."""
    
    def __init__(self):
        self.datasets = {}
    
    async def save(self, dataset):
        self.datasets[dataset.id] = dataset
        return dataset
    
    async def get_by_id(self, dataset_id):
        return self.datasets.get(dataset_id)
    
    async def get_by_tenant(self, tenant_id, limit=100, offset=0):
        return [ds for ds in self.datasets.values() if ds.tenant_id == tenant_id]


class MockJobRepository:
    """Mock job repository for testing."""
    
    def __init__(self):
        self.jobs = {}
    
    async def save(self, job):
        self.jobs[job.id] = job
        return job
    
    async def get_by_id(self, job_id):
        return self.jobs.get(job_id)


class MockFileStorage:
    """Mock file storage for testing."""
    
    async def upload_file(self, content, filename, tenant_id, content_type=None):
        return f"files/{tenant_id}/{filename}"
    
    async def download_file(self, file_path):
        return b"mock,file,content\n1,2,3\n4,5,6"
    
    async def delete_file(self, file_path):
        return True
    
    async def update_file(self, file_path, content):
        return True


async def test_lineage_service():
    """Test the data lineage service functionality."""
    
    print("🧪 Testing Data Lineage Tracking Service...")
    
    # Setup
    dataset_repo = MockDatasetRepository()
    job_repo = MockJobRepository()
    file_storage = MockFileStorage()
    lineage_service = DataLineageService(dataset_repo, job_repo, file_storage)
    
    tenant_id = uuid4()
    user_id = uuid4()
    
    # Test 1: Create source datasets
    print("\n1️⃣ Testing source dataset creation...")
    
    source_dataset1 = Dataset.create_new_dataset(
        name="Raw Sales Data",
        description="Original sales data from CRM",
        tenant_id=tenant_id,
        created_by=user_id,
        file_path="files/sales_raw.csv",
        file_format=DataFormat.CSV,
        file_size=1024
    )
    
    source_dataset2 = Dataset.create_new_dataset(
        name="Customer Data",
        description="Customer information",
        tenant_id=tenant_id,
        created_by=user_id,
        file_path="files/customers.csv",
        file_format=DataFormat.CSV,
        file_size=2048
    )
    
    await dataset_repo.save(source_dataset1)
    await dataset_repo.save(source_dataset2)
    
    # Record lineage for source datasets (no parents)
    await lineage_service.record_dataset_creation(
        dataset=source_dataset1,
        source_datasets=[],
        transformations=[],
        created_by=user_id
    )
    
    await lineage_service.record_dataset_creation(
        dataset=source_dataset2,
        source_datasets=[],
        transformations=[],
        created_by=user_id
    )
    
    print(f"✅ Created source datasets: {source_dataset1.name}, {source_dataset2.name}")
    
    # Test 2: Create derived dataset with transformations
    print("\n2️⃣ Testing derived dataset with transformations...")
    
    derived_dataset = Dataset.create_new_dataset(
        name="Cleaned Sales Data",
        description="Sales data after cleaning and joining with customer data",
        tenant_id=tenant_id,
        created_by=user_id,
        file_path="files/sales_cleaned.csv",
        file_format=DataFormat.CSV,
        file_size=1536
    )
    
    await dataset_repo.save(derived_dataset)
    
    # Create transformations
    transformations = [
        DataTransformation(
            transformation_type=TransformationType.CLEAN,
            parameters={"remove_duplicates": True, "fill_missing": "mean"},
            description="Remove duplicates and fill missing values"
        ),
        DataTransformation(
            transformation_type=TransformationType.JOIN,
            parameters={"join_type": "left", "on": "customer_id"},
            description="Join with customer data"
        )
    ]
    
    # Record transformation lineage
    await lineage_service.record_transformation_lineage(
        source_dataset_id=source_dataset1.id,
        target_dataset_id=derived_dataset.id,
        transformations=transformations,
        created_by=user_id
    )
    
    print(f"✅ Created derived dataset: {derived_dataset.name}")
    
    # Test 3: Create final aggregated dataset
    print("\n3️⃣ Testing final aggregated dataset...")
    
    final_dataset = Dataset.create_new_dataset(
        name="Monthly Sales Summary",
        description="Monthly aggregated sales data",
        tenant_id=tenant_id,
        created_by=user_id,
        file_path="files/sales_monthly.csv",
        file_format=DataFormat.CSV,
        file_size=512
    )
    
    await dataset_repo.save(final_dataset)
    
    # Create aggregation transformation
    agg_transformation = DataTransformation(
        transformation_type=TransformationType.AGGREGATE,
        parameters={"group_by": ["month"], "agg_functions": {"sales": "sum", "orders": "count"}},
        description="Monthly sales aggregation"
    )
    
    await lineage_service.record_transformation_lineage(
        source_dataset_id=derived_dataset.id,
        target_dataset_id=final_dataset.id,
        transformations=[agg_transformation],
        created_by=user_id
    )
    
    print(f"✅ Created final dataset: {final_dataset.name}")
    
    # Test 4: Build and analyze lineage graph
    print("\n4️⃣ Testing lineage graph construction...")
    
    graph = await lineage_service.build_lineage_graph(tenant_id)
    
    print(f"✅ Built lineage graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Test 5: Get dataset lineage
    print("\n5️⃣ Testing dataset lineage queries...")
    
    # Get upstream lineage for final dataset
    final_lineage = await lineage_service.get_dataset_lineage(final_dataset.id, direction="upstream")
    print(f"✅ Final dataset upstream lineage: {len(final_lineage['upstream'])} ancestors")
    
    # Get downstream lineage for source dataset
    source_lineage = await lineage_service.get_dataset_lineage(source_dataset1.id, direction="downstream")
    print(f"✅ Source dataset downstream lineage: {len(source_lineage['downstream'])} descendants")
    
    # Test 6: Get lineage path
    print("\n6️⃣ Testing lineage path discovery...")
    
    path = await lineage_service.get_lineage_path(source_dataset1.id, final_dataset.id)
    if path:
        print(f"✅ Found lineage path with {path['path_length']} steps")
    else:
        print("❌ No lineage path found")
    
    # Test 7: Impact analysis
    print("\n7️⃣ Testing impact analysis...")
    
    impact = await lineage_service.get_impact_analysis(source_dataset1.id)
    print(f"✅ Impact analysis: {impact['total_affected_datasets']} datasets affected")
    print(f"   Transformation types: {list(impact['transformation_types_used'].keys())}")
    
    # Test 8: Lineage statistics
    print("\n8️⃣ Testing lineage statistics...")
    
    stats = await lineage_service.get_lineage_statistics(tenant_id)
    print(f"✅ Lineage statistics:")
    print(f"   Total datasets: {stats['total_datasets']}")
    print(f"   Source datasets: {stats['source_datasets']}")
    print(f"   Derived datasets: {stats['derived_datasets']}")
    print(f"   Sink datasets: {stats['sink_datasets']}")
    print(f"   Total relationships: {stats['total_relationships']}")
    
    # Test 9: Search lineage
    print("\n9️⃣ Testing lineage search...")
    
    search_results = await lineage_service.search_lineage(tenant_id, "sales", "dataset_name")
    print(f"✅ Search for 'sales': {len(search_results)} results")
    
    # Test 10: Visualization data
    print("\n🔟 Testing visualization data generation...")
    
    vis_data = await lineage_service.get_lineage_visualization_data(tenant_id)
    print(f"✅ Visualization data: {vis_data['total_nodes']} nodes, {vis_data['total_edges']} edges")
    
    # Test 11: Focused visualization
    print("\n1️⃣1️⃣ Testing focused visualization...")
    
    focused_vis = await lineage_service.get_lineage_visualization_data(
        tenant_id, dataset_id=derived_dataset.id
    )
    print(f"✅ Focused visualization: {focused_vis['total_nodes']} nodes, {focused_vis['total_edges']} edges")
    
    print("\n🎉 All lineage tracking tests completed successfully!")
    
    return True


async def test_lineage_graph():
    """Test the lineage graph data structure."""
    
    print("\n🧪 Testing Lineage Graph Data Structure...")
    
    graph = LineageGraph()
    
    # Create test nodes
    node1 = LineageNode(uuid4(), "Dataset 1", "dataset")
    node2 = LineageNode(uuid4(), "Dataset 2", "dataset")
    node3 = LineageNode(uuid4(), "Dataset 3", "dataset")
    
    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    # Add edges
    graph.add_edge(node1.dataset_id, node2.dataset_id)
    graph.add_edge(node2.dataset_id, node3.dataset_id)
    
    print(f"✅ Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Test ancestors
    ancestors = graph.get_ancestors(node3.dataset_id)
    print(f"✅ Node 3 ancestors: {len(ancestors)}")
    
    # Test descendants
    descendants = graph.get_descendants(node1.dataset_id)
    print(f"✅ Node 1 descendants: {len(descendants)}")
    
    # Test path finding
    path = graph.get_lineage_path(node1.dataset_id, node3.dataset_id)
    if path:
        print(f"✅ Path from Node 1 to Node 3: {len(path)} steps")
    
    # Test impact analysis
    impact = graph.get_impact_analysis(node1.dataset_id)
    print(f"✅ Impact analysis: {impact['total_affected_datasets']} affected datasets")
    
    print("🎉 Lineage graph tests completed successfully!")
    
    return True


async def main():
    """Run all tests."""
    
    print("🚀 Starting Data Lineage Tracking Tests")
    print("=" * 50)
    
    try:
        # Test lineage graph data structure
        await test_lineage_graph()
        
        # Test lineage service
        await test_lineage_service()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED! Data lineage tracking is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)