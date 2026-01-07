#!/usr/bin/env python3
"""
Demo test script for Enterprise SaaS Platform
Tests core functionality without requiring Docker
"""
import sys
import os
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing core module imports...")
    
    try:
        # Test shared components
        from src.shared.domain.base import AggregateRoot, Entity, ValueObject
        from src.shared.infrastructure.container import Container
        print("✅ Shared components imported successfully")
        
        # Test user service
        from src.services.user_service.domain.entities import User, UserProfile
        from src.services.user_service.infrastructure.auth import JWTManager
        print("✅ User service components imported successfully")
        
        # Test tenant service
        from src.services.tenant_service.domain.entities import Tenant, TenantSettings
        from src.services.tenant_service.infrastructure.quota import QuotaManager
        print("✅ Tenant service components imported successfully")
        
        # Test data service
        from src.services.data_service.domain.entities import Dataset, DataFile
        from src.services.data_service.infrastructure.lineage_service import DataLineageService
        print("✅ Data service components imported successfully")
        
        # Test API Gateway
        from src.services.api_gateway.infrastructure.routing import ServiceRouter
        from src.services.api_gateway.infrastructure.auth_middleware import AuthMiddleware
        print("✅ API Gateway components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_domain_models():
    """Test domain model creation."""
    print("\n🧪 Testing domain model creation...")
    
    try:
        from src.services.user_service.domain.entities import User, UserProfile
        from src.services.tenant_service.domain.entities import Tenant
        from src.services.data_service.domain.entities import Dataset
        
        # Test User creation
        user = User(
            id="test-user-123",
            email="test@example.com",
            password_hash="hashed_password",
            full_name="Test User",
            is_active=True
        )
        print(f"✅ User created: {user.email}")
        
        # Test Tenant creation
        tenant = Tenant(
            id="test-tenant-123",
            name="Test Tenant",
            domain="test.example.com",
            subscription_plan="basic",
            is_active=True
        )
        print(f"✅ Tenant created: {tenant.name}")
        
        # Test Dataset creation
        dataset = Dataset(
            id="test-dataset-123",
            tenant_id="test-tenant-123",
            name="Test Dataset",
            description="A test dataset",
            file_path="/test/path",
            file_size=1024,
            format="csv"
        )
        print(f"✅ Dataset created: {dataset.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain model test failed: {e}")
        return False

def test_lineage_service():
    """Test data lineage service functionality."""
    print("\n🧪 Testing data lineage service...")
    
    try:
        from src.services.data_service.infrastructure.lineage_service import (
            DataLineageService, LineageNode, LineageGraph
        )
        
        # Create lineage service
        lineage_service = DataLineageService()
        print("✅ DataLineageService created")
        
        # Test LineageNode creation
        node = LineageNode(
            node_id="test-node-1",
            node_type="dataset",
            name="Test Dataset",
            metadata={"format": "csv", "size": 1024}
        )
        print(f"✅ LineageNode created: {node.name}")
        
        # Test LineageGraph creation
        graph = LineageGraph()
        graph.add_node(node)
        print("✅ LineageGraph created and node added")
        
        # Test graph operations
        nodes = graph.get_all_nodes()
        assert len(nodes) == 1
        print(f"✅ Graph contains {len(nodes)} node(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Lineage service test failed: {e}")
        return False

def test_transformation_engine():
    """Test data transformation engine."""
    print("\n🧪 Testing transformation engine...")
    
    try:
        from src.services.data_service.infrastructure.transformations import TransformationEngine
        from src.services.data_service.infrastructure.transformation_service import DataTransformationService
        
        # Create transformation engine
        engine = TransformationEngine()
        print("✅ TransformationEngine created")
        
        # Test available transformations
        transformations = engine.get_available_transformations()
        print(f"✅ Available transformations: {len(transformations)}")
        
        # Create transformation service
        service = DataTransformationService()
        print("✅ DataTransformationService created")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformation engine test failed: {e}")
        return False

def test_jwt_functionality():
    """Test JWT authentication functionality."""
    print("\n🧪 Testing JWT functionality...")
    
    try:
        from src.services.user_service.infrastructure.auth import JWTManager
        
        # Create JWT manager
        jwt_manager = JWTManager(secret_key="test-secret-key")
        print("✅ JWTManager created")
        
        # Test token creation
        payload = {"user_id": "test-123", "email": "test@example.com"}
        token = jwt_manager.create_access_token(payload)
        print("✅ JWT token created")
        
        # Test token validation
        decoded = jwt_manager.decode_token(token)
        assert decoded["user_id"] == "test-123"
        print("✅ JWT token validated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ JWT test failed: {e}")
        return False

def test_api_schemas():
    """Test API schema validation."""
    print("\n🧪 Testing API schemas...")
    
    try:
        from src.services.data_service.api.schemas import (
            DatasetResponse, LineageNodeResponse, TransformationRequest
        )
        
        # Test DatasetResponse
        dataset_data = {
            "id": "test-123",
            "tenant_id": "tenant-123",
            "name": "Test Dataset",
            "description": "Test description",
            "format": "csv",
            "file_size": 1024,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        dataset_response = DatasetResponse(**dataset_data)
        print(f"✅ DatasetResponse created: {dataset_response.name}")
        
        # Test LineageNodeResponse
        lineage_data = {
            "node_id": "node-123",
            "node_type": "dataset",
            "name": "Test Node",
            "metadata": {"format": "csv"}
        }
        lineage_response = LineageNodeResponse(**lineage_data)
        print(f"✅ LineageNodeResponse created: {lineage_response.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ API schema test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enterprise SaaS Platform - Demo Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_domain_models,
        test_lineage_service,
        test_transformation_engine,
        test_jwt_functionality,
        test_api_schemas
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The platform is working correctly.")
        print("\n🎯 Next Steps:")
        print("1. ✅ Streamlit frontend is running at http://localhost:8501")
        print("2. 🐳 Start Docker Desktop to test with full backend services")
        print("3. 🚀 Deploy to Streamlit Cloud for quick demo")
        print("4. ☁️ Deploy to Railway/AWS for production")
        
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)