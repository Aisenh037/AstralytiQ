"""
Basic structure tests for the Enterprise SaaS Platform.
"""
import pytest
from uuid import uuid4

def test_domain_models_import():
    """Test that domain models can be imported successfully."""
    from src.shared.domain.models import User, Tenant, Dataset, MLModel, Prediction
    assert User is not None
    assert Tenant is not None
    assert Dataset is not None
    assert MLModel is not None
    assert Prediction is not None


def test_user_model_creation():
    """Test User model creation."""
    from src.shared.domain.models import User, UserRole, UserProfile
    
    user = User(
        email="test@example.com",
        password_hash="hashed_password",
        role=UserRole.ANALYST,
        tenant_id=uuid4()
    )
    
    assert user.email == "test@example.com"
    assert user.role == UserRole.ANALYST
    assert user.is_active is True
    assert user.email_verified is False


def test_tenant_model_creation():
    """Test Tenant model creation."""
    from src.shared.domain.models import Tenant, SubscriptionPlan
    
    tenant = Tenant(
        name="Test Company",
        domain="test.example.com",
        subscription_plan=SubscriptionPlan.PROFESSIONAL
    )
    
    assert tenant.name == "Test Company"
    assert tenant.domain == "test.example.com"
    assert tenant.subscription_plan == SubscriptionPlan.PROFESSIONAL
    assert tenant.is_active is True


def test_database_settings():
    """Test database settings configuration."""
    from src.shared.infrastructure.database import DatabaseSettings
    
    settings = DatabaseSettings()
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert "postgresql+asyncpg://" in settings.postgres_url


def test_dependency_injection_container():
    """Test dependency injection container."""
    from src.shared.infrastructure.container import Container
    
    container = Container()
    
    # Test service registration
    class TestService:
        def __init__(self):
            self.value = "test"
    
    container.register_singleton(TestService, TestService)
    
    # Test service resolution
    service1 = container.resolve(TestService)
    service2 = container.resolve(TestService)
    
    assert service1 is service2  # Should be the same instance (singleton)
    assert service1.value == "test"


def test_api_gateway_creation():
    """Test API Gateway FastAPI app creation."""
    from src.services.api_gateway.main import create_app
    
    app = create_app()
    assert app is not None
    assert app.title == "Enterprise SaaS Platform - API Gateway"


def test_service_creation():
    """Test all service FastAPI apps can be created."""
    from src.services.user_service.main import create_app as create_user_app
    from src.services.tenant_service.main import create_app as create_tenant_app
    from src.services.data_service.main import create_app as create_data_app
    from src.services.ml_service.main import create_app as create_ml_app
    
    user_app = create_user_app()
    tenant_app = create_tenant_app()
    data_app = create_data_app()
    ml_app = create_ml_app()
    
    assert user_app.title == "User Management Service"
    assert tenant_app.title == "Tenant Management Service"
    assert data_app.title == "Data Processing Service"
    assert ml_app.title == "ML/Analytics Service"


if __name__ == "__main__":
    pytest.main([__file__])