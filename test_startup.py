#!/usr/bin/env python3
"""
Simple startup test for the Enterprise SaaS Platform.
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.shared.domain.models import User, Tenant, Dataset, MLModel
        print("✅ Domain models imported successfully")
        
        from src.shared.infrastructure.database import DatabaseSettings, DatabaseManager
        print("✅ Database infrastructure imported successfully")
        
        from src.shared.infrastructure.container import Container
        print("✅ Dependency injection container imported successfully")
        
        from src.services.api_gateway.main import create_app as create_gateway_app
        from src.services.user_service.main import create_app as create_user_app
        from src.services.tenant_service.main import create_app as create_tenant_app
        from src.services.data_service.main import create_app as create_data_app
        from src.services.ml_service.main import create_app as create_ml_app
        print("✅ All service applications imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_app_creation():
    """Test that FastAPI apps can be created."""
    print("\n🧪 Testing app creation...")
    
    try:
        from src.services.api_gateway.main import create_app as create_gateway_app
        from src.services.user_service.main import create_app as create_user_app
        from src.services.tenant_service.main import create_app as create_tenant_app
        from src.services.data_service.main import create_app as create_data_app
        from src.services.ml_service.main import create_app as create_ml_app
        
        # Create all apps
        gateway_app = create_gateway_app()
        user_app = create_user_app()
        tenant_app = create_tenant_app()
        data_app = create_data_app()
        ml_app = create_ml_app()
        
        print("✅ API Gateway created successfully")
        print("✅ User Service created successfully")
        print("✅ Tenant Service created successfully")
        print("✅ Data Service created successfully")
        print("✅ ML Service created successfully")
        
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False


def test_domain_models():
    """Test domain model creation."""
    print("\n🧪 Testing domain models...")
    
    try:
        from src.shared.domain.models import (
            User, Tenant, Dataset, MLModel, 
            UserRole, SubscriptionPlan, DataSourceType, ModelType
        )
        from uuid import uuid4
        
        # Test User creation
        user = User(
            email="test@example.com",
            password_hash="hashed_password",
            role=UserRole.ANALYST,
            tenant_id=uuid4()
        )
        print("✅ User model created successfully")
        
        # Test Tenant creation
        tenant = Tenant(
            name="Test Company",
            subscription_plan=SubscriptionPlan.PROFESSIONAL
        )
        print("✅ Tenant model created successfully")
        
        # Test Dataset creation
        dataset = Dataset(
            name="Test Dataset",
            tenant_id=uuid4(),
            source_type=DataSourceType.CSV
        )
        print("✅ Dataset model created successfully")
        
        # Test MLModel creation
        ml_model = MLModel(
            name="Test Model",
            tenant_id=uuid4(),
            ml_model_type=ModelType.LINEAR_REGRESSION,
            version="1.0.0"
        )
        print("✅ MLModel created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Domain model test failed: {e}")
        return False


def main():
    """Run all startup tests."""
    print("🚀 Enterprise SaaS Platform - Startup Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_creation,
        test_domain_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The platform is ready for development.")
        print("\n📋 Next steps:")
        print("  1. Start database services (PostgreSQL, MongoDB, Redis)")
        print("  2. Run: python scripts/start_services.py")
        print("  3. Visit: http://localhost:8000/docs")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)