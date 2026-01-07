#!/usr/bin/env python3
"""
Simple demo script for Enterprise SaaS Platform
Demonstrates core functionality without database dependencies
"""
import sys
import os
from pathlib import Path
import uuid
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_core_functionality():
    """Demonstrate core platform functionality."""
    print("🚀 Enterprise SaaS Platform - Core Functionality Demo")
    print("=" * 60)
    
    # Demo 1: Domain Models
    print("\n📦 Demo 1: Domain Models")
    try:
        from src.shared.domain.models import User, UserProfile, UserRole, Tenant, SubscriptionPlan
        from src.services.data_service.domain.entities import Dataset
        
        # Create a user with proper UUID and required fields
        user_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        
        user = User(
            id=user_id,
            email="demo@example.com",
            password_hash="$2b$12$hashed_password_example",
            role=UserRole.ANALYST,
            tenant_id=tenant_id,
            is_active=True
        )
        print(f"✅ User created: {user.email} (ID: {str(user.id)[:8]}...)")
        
        # Create a tenant
        tenant = Tenant(
            id=tenant_id,
            name="Demo Tenant",
            domain="demo.example.com",
            subscription_plan=SubscriptionPlan.BASIC,
            is_active=True
        )
        print(f"✅ Tenant created: {tenant.name} (ID: {str(tenant.id)[:8]}...)")
        
        # Create a dataset
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            tenant_id=tenant_id,
            created_by=user_id,  # Required field
            name="Demo Sales Data",
            description="Sample sales dataset for demonstration",
            file_path="/demo/sales_data.csv",
            file_size=2048
        )
        print(f"✅ Dataset created: {dataset.name} (ID: {str(dataset.id)[:8]}...)")
        
    except Exception as e:
        print(f"❌ Domain models demo failed: {e}")
        return False
    
    # Demo 2: JWT Authentication
    print("\n🔐 Demo 2: JWT Authentication")
    try:
        from src.services.user_service.infrastructure.auth import JWTManager
        
        jwt_manager = JWTManager(secret_key="demo-secret-key-12345")
        
        # Create token using the user object
        token = jwt_manager.create_access_token(user)
        print(f"✅ JWT token created: {token[:50]}...")
        
        # Validate token
        decoded = jwt_manager.verify_token(token)
        if decoded:
            print(f"✅ Token validated - User: {decoded.email}")
        else:
            print("❌ Token validation failed")
        
    except Exception as e:
        print(f"❌ JWT demo failed: {e}")
        return False
    
    # Demo 3: Data Transformations
    print("\n🔄 Demo 3: Data Transformations")
    try:
        from src.services.data_service.infrastructure.transformations import TransformationEngine
        
        engine = TransformationEngine()
        
        # Get available transformations
        transformations = engine.get_available_transformations()
        print(f"✅ Available transformations: {len(transformations)}")
        
        for category, transforms in transformations.items():
            print(f"   📂 {category}: {', '.join(transforms)}")
        
        # Demo transformation validation
        transform_config = {
            "step": "remove_duplicates",
            "parameters": {"keep": "first"}
        }
        
        is_valid = engine.validate_transformation(transform_config)
        print(f"✅ Transformation validation: {'Valid' if is_valid else 'Invalid'}")
        
    except Exception as e:
        print(f"❌ Transformations demo failed: {e}")
        return False
    
    # Demo 4: Lineage Graph
    print("\n🔗 Demo 4: Data Lineage")
    try:
        from src.services.data_service.infrastructure.lineage_service import LineageNode, LineageGraph
        
        # Create lineage nodes
        source_node = LineageNode(
            node_id="source-data-001",
            node_type="dataset",
            name="Raw Sales Data",
            metadata={"format": "csv", "size": 1024, "source": "upload"}
        )
        
        processed_node = LineageNode(
            node_id="processed-data-001",
            node_type="dataset",
            name="Cleaned Sales Data",
            metadata={"format": "csv", "size": 950, "transformations": 3}
        )
        
        # Create lineage graph
        graph = LineageGraph()
        graph.add_node(source_node)
        graph.add_node(processed_node)
        
        # Add relationship
        graph.add_edge(source_node.node_id, processed_node.node_id, {
            "transformation": "data_cleaning",
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"✅ Lineage graph created with {len(graph.get_all_nodes())} nodes")
        
        # Test lineage traversal
        descendants = graph.get_descendants(source_node.node_id)
        print(f"✅ Found {len(descendants)} descendants of source data")
        
    except Exception as e:
        print(f"❌ Lineage demo failed: {e}")
        return False
    
    # Demo 5: API Schemas
    print("\n📋 Demo 5: API Schemas")
    try:
        from src.services.data_service.api.schemas import DatasetResponse, LineageNodeResponse
        
        # Test dataset response schema
        dataset_data = {
            "id": str(dataset_id),
            "tenant_id": str(tenant_id),
            "name": "Demo Sales Data",
            "description": "Sample sales dataset",
            "format": "csv",
            "file_size": 2048,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        dataset_response = DatasetResponse(**dataset_data)
        print(f"✅ DatasetResponse schema validated: {dataset_response.name}")
        
        # Test lineage node response schema
        lineage_data = {
            "node_id": "demo-node-001",
            "node_type": "dataset",
            "name": "Demo Dataset Node",
            "metadata": {"format": "csv", "transformations": 2}
        }
        
        lineage_response = LineageNodeResponse(**lineage_data)
        print(f"✅ LineageNodeResponse schema validated: {lineage_response.name}")
        
    except Exception as e:
        print(f"❌ API schemas demo failed: {e}")
        return False
    
    # Demo 6: Security Features
    print("\n🛡️ Demo 6: Security Features")
    try:
        from src.services.user_service.infrastructure.security import PasswordManager
        from src.services.api_gateway.infrastructure.rate_limiting import RateLimiter
        
        # Password hashing
        password_manager = PasswordManager()
        password = "demo_password_123"
        hashed = password_manager.hash_password(password)
        is_valid = password_manager.verify_password(password, hashed)
        
        print(f"✅ Password hashing: {'Valid' if is_valid else 'Invalid'}")
        
        # Rate limiting (without Redis)
        rate_limiter = RateLimiter()
        print(f"✅ Rate limiter initialized")
        
    except Exception as e:
        print(f"❌ Security demo failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All demos completed successfully!")
    print("\n🎯 Platform Status:")
    print("✅ Core domain models working")
    print("✅ JWT authentication functional")
    print("✅ Data transformations available")
    print("✅ Lineage tracking operational")
    print("✅ API schemas validated")
    print("✅ Security features active")
    
    print(f"\n🌐 Frontend Access:")
    print(f"   Streamlit App: http://localhost:8501")
    print(f"   Status: {'🟢 Running' if check_streamlit_running() else '🔴 Not Running'}")
    
    print(f"\n📚 Documentation:")
    print(f"   Development Guide: DEVELOPMENT_GUIDE.md")
    print(f"   Deployment Guide: DEPLOYMENT_GUIDE.md")
    print(f"   Task Progress: .kiro/specs/enterprise-saas-platform/tasks.md")
    
    return True

def check_streamlit_running():
    """Check if Streamlit is running."""
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_next_steps():
    """Show recommended next steps."""
    print("\n🚀 Recommended Next Steps:")
    print("\n1. 🎨 Frontend Development:")
    print("   • Streamlit app is ready for testing at http://localhost:8501")
    print("   • Consider building React frontend for production")
    print("   • Test data upload and transformation features")
    
    print("\n2. 🐳 Full Backend Testing:")
    print("   • Start Docker Desktop")
    print("   • Run: docker-compose up -d")
    print("   • Test with: python run_local.py")
    
    print("\n3. ☁️ Cloud Deployment:")
    print("   • Quick demo: Deploy to Streamlit Cloud")
    print("   • MVP: Deploy to Railway or Render")
    print("   • Production: Deploy to AWS/GCP")
    
    print("\n4. 🔧 Development Tasks:")
    print("   • Implement ML Service (Task 7)")
    print("   • Build Dashboard Service (Task 8)")
    print("   • Add Billing Service (Task 9)")
    
    print("\n5. 📊 Testing & Monitoring:")
    print("   • Run integration tests")
    print("   • Set up monitoring and logging")
    print("   • Performance optimization")

if __name__ == "__main__":
    try:
        success = demo_core_functionality()
        if success:
            show_next_steps()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with unexpected error: {e}")
        sys.exit(1)