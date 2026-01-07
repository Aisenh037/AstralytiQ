"""
Tenant service domain entities and business logic.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
import re

from src.shared.domain.models import Tenant as BaseTenant, TenantSettings, SubscriptionPlan
from src.shared.domain.base import DomainService, ValueObject


class TenantQuota(ValueObject):
    """Tenant resource quota configuration."""
    max_users: int = 10
    max_datasets: int = 50
    max_models: int = 20
    max_storage_gb: int = 10
    max_api_calls_per_month: int = 10000
    max_concurrent_jobs: int = 5
    
    def is_within_limits(
        self,
        current_users: int = 0,
        current_datasets: int = 0,
        current_models: int = 0,
        current_storage_gb: float = 0,
        current_api_calls: int = 0,
        current_jobs: int = 0
    ) -> tuple[bool, List[str]]:
        """Check if current usage is within quota limits."""
        violations = []
        
        if current_users >= self.max_users:
            violations.append(f"User limit exceeded: {current_users}/{self.max_users}")
        
        if current_datasets >= self.max_datasets:
            violations.append(f"Dataset limit exceeded: {current_datasets}/{self.max_datasets}")
        
        if current_models >= self.max_models:
            violations.append(f"Model limit exceeded: {current_models}/{self.max_models}")
        
        if current_storage_gb >= self.max_storage_gb:
            violations.append(f"Storage limit exceeded: {current_storage_gb:.1f}GB/{self.max_storage_gb}GB")
        
        if current_api_calls >= self.max_api_calls_per_month:
            violations.append(f"API call limit exceeded: {current_api_calls}/{self.max_api_calls_per_month}")
        
        if current_jobs >= self.max_concurrent_jobs:
            violations.append(f"Concurrent job limit exceeded: {current_jobs}/{self.max_concurrent_jobs}")
        
        return len(violations) == 0, violations


class TenantBranding(ValueObject):
    """Tenant branding configuration."""
    logo_url: Optional[str] = None
    primary_color: str = "#007bff"
    secondary_color: str = "#6c757d"
    company_name: Optional[str] = None
    custom_domain: Optional[str] = None
    favicon_url: Optional[str] = None
    
    def validate_colors(self) -> bool:
        """Validate color format (hex colors)."""
        hex_pattern = r'^#[0-9A-Fa-f]{6}$'
        return (
            re.match(hex_pattern, self.primary_color) is not None and
            re.match(hex_pattern, self.secondary_color) is not None
        )


class TenantFeatures(ValueObject):
    """Tenant feature flags."""
    advanced_analytics: bool = False
    custom_models: bool = False
    api_access: bool = True
    data_export: bool = True
    white_labeling: bool = False
    sso_integration: bool = False
    audit_logs: bool = False
    priority_support: bool = False
    
    @classmethod
    def for_subscription_plan(cls, plan: SubscriptionPlan) -> "TenantFeatures":
        """Get features based on subscription plan."""
        if plan == SubscriptionPlan.FREE:
            return cls(
                advanced_analytics=False,
                custom_models=False,
                api_access=False,
                data_export=False,
                white_labeling=False,
                sso_integration=False,
                audit_logs=False,
                priority_support=False
            )
        elif plan == SubscriptionPlan.BASIC:
            return cls(
                advanced_analytics=False,
                custom_models=False,
                api_access=True,
                data_export=True,
                white_labeling=False,
                sso_integration=False,
                audit_logs=False,
                priority_support=False
            )
        elif plan == SubscriptionPlan.PROFESSIONAL:
            return cls(
                advanced_analytics=True,
                custom_models=True,
                api_access=True,
                data_export=True,
                white_labeling=False,
                sso_integration=True,
                audit_logs=True,
                priority_support=False
            )
        elif plan == SubscriptionPlan.ENTERPRISE:
            return cls(
                advanced_analytics=True,
                custom_models=True,
                api_access=True,
                data_export=True,
                white_labeling=True,
                sso_integration=True,
                audit_logs=True,
                priority_support=True
            )
        else:
            return cls()


class Tenant(BaseTenant):
    """Extended Tenant entity with business logic."""
    
    def get_quota(self) -> TenantQuota:
        """Get tenant quota based on subscription plan."""
        quota_configs = {
            SubscriptionPlan.FREE: TenantQuota(
                max_users=3,
                max_datasets=10,
                max_models=5,
                max_storage_gb=1,
                max_api_calls_per_month=1000,
                max_concurrent_jobs=1
            ),
            SubscriptionPlan.BASIC: TenantQuota(
                max_users=10,
                max_datasets=50,
                max_models=20,
                max_storage_gb=10,
                max_api_calls_per_month=10000,
                max_concurrent_jobs=3
            ),
            SubscriptionPlan.PROFESSIONAL: TenantQuota(
                max_users=50,
                max_datasets=200,
                max_models=100,
                max_storage_gb=100,
                max_api_calls_per_month=100000,
                max_concurrent_jobs=10
            ),
            SubscriptionPlan.ENTERPRISE: TenantQuota(
                max_users=1000,
                max_datasets=1000,
                max_models=500,
                max_storage_gb=1000,
                max_api_calls_per_month=1000000,
                max_concurrent_jobs=50
            )
        }
        
        return quota_configs.get(self.subscription_plan, TenantQuota())
    
    def get_features(self) -> TenantFeatures:
        """Get tenant features based on subscription plan."""
        return TenantFeatures.for_subscription_plan(self.subscription_plan)
    
    def get_branding(self) -> TenantBranding:
        """Get tenant branding configuration."""
        branding_data = self.settings.branding if self.settings else {}
        return TenantBranding(**branding_data)
    
    def update_branding(self, branding: TenantBranding) -> None:
        """Update tenant branding."""
        if not branding.validate_colors():
            raise ValueError("Invalid color format. Use hex colors (e.g., #007bff)")
        
        if not self.settings:
            self.settings = TenantSettings()
        
        self.settings.branding = branding.dict()
        self.updated_at = datetime.utcnow()
    
    def upgrade_subscription(self, new_plan: SubscriptionPlan) -> None:
        """Upgrade tenant subscription plan."""
        if new_plan == self.subscription_plan:
            return
        
        # Validate upgrade path (can't downgrade without explicit method)
        plan_hierarchy = {
            SubscriptionPlan.FREE: 0,
            SubscriptionPlan.BASIC: 1,
            SubscriptionPlan.PROFESSIONAL: 2,
            SubscriptionPlan.ENTERPRISE: 3
        }
        
        current_level = plan_hierarchy.get(self.subscription_plan, 0)
        new_level = plan_hierarchy.get(new_plan, 0)
        
        if new_level < current_level:
            raise ValueError("Use downgrade_subscription() method for downgrades")
        
        self.subscription_plan = new_plan
        self.updated_at = datetime.utcnow()
    
    def downgrade_subscription(self, new_plan: SubscriptionPlan, force: bool = False) -> List[str]:
        """Downgrade tenant subscription plan with validation."""
        warnings = []
        
        if new_plan == self.subscription_plan:
            return warnings
        
        # Check if downgrade will violate quotas
        new_quota = TenantQuota()
        if new_plan == SubscriptionPlan.FREE:
            new_quota = TenantQuota(max_users=3, max_datasets=10, max_models=5)
        elif new_plan == SubscriptionPlan.BASIC:
            new_quota = TenantQuota(max_users=10, max_datasets=50, max_models=20)
        elif new_plan == SubscriptionPlan.PROFESSIONAL:
            new_quota = TenantQuota(max_users=50, max_datasets=200, max_models=100)
        
        # In a real implementation, you would check current usage
        # For now, just add warnings
        if new_plan == SubscriptionPlan.FREE:
            warnings.append("Free plan limits: 3 users, 10 datasets, 5 models")
        
        if not force and warnings:
            raise ValueError(f"Cannot downgrade due to: {'; '.join(warnings)}")
        
        self.subscription_plan = new_plan
        self.updated_at = datetime.utcnow()
        
        return warnings
    
    def activate(self) -> None:
        """Activate tenant."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate tenant."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update tenant settings."""
        if not self.settings:
            self.settings = TenantSettings()
        
        # Merge settings
        current_settings = self.settings.dict()
        current_settings.update(new_settings)
        
        self.settings = TenantSettings(**current_settings)
        self.updated_at = datetime.utcnow()
    
    @classmethod
    def create_new_tenant(
        cls,
        name: str,
        domain: Optional[str] = None,
        subscription_plan: SubscriptionPlan = SubscriptionPlan.FREE,
        owner_id: Optional[UUID] = None
    ) -> "Tenant":
        """Create a new tenant with default settings."""
        # Validate domain format
        if domain and not cls._is_valid_domain(domain):
            raise ValueError("Invalid domain format")
        
        # Create default settings
        features = TenantFeatures.for_subscription_plan(subscription_plan)
        settings = TenantSettings(
            features=features.dict(),
            branding=TenantBranding().dict(),
            limits=TenantQuota().dict() if subscription_plan == SubscriptionPlan.FREE else {}
        )
        
        return cls(
            name=name,
            domain=domain,
            subscription_plan=subscription_plan,
            owner_id=owner_id,
            settings=settings
        )
    
    @staticmethod
    def _is_valid_domain(domain: str) -> bool:
        """Validate domain format."""
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return re.match(domain_pattern, domain) is not None


class TenantDomainService(DomainService):
    """Domain service for tenant-related business logic."""
    
    @staticmethod
    def can_create_resource(
        tenant: Tenant,
        resource_type: str,
        current_count: int
    ) -> tuple[bool, Optional[str]]:
        """Check if tenant can create a new resource."""
        quota = tenant.get_quota()
        
        limits = {
            "user": quota.max_users,
            "dataset": quota.max_datasets,
            "model": quota.max_models
        }
        
        limit = limits.get(resource_type)
        if limit is None:
            return True, None
        
        if current_count >= limit:
            return False, f"{resource_type.title()} limit reached ({current_count}/{limit})"
        
        return True, None
    
    @staticmethod
    def can_use_feature(tenant: Tenant, feature_name: str) -> bool:
        """Check if tenant can use a specific feature."""
        features = tenant.get_features()
        return getattr(features, feature_name, False)
    
    @staticmethod
    def get_storage_usage_percentage(tenant: Tenant, current_storage_gb: float) -> float:
        """Get storage usage as percentage of quota."""
        quota = tenant.get_quota()
        if quota.max_storage_gb == 0:
            return 0.0
        return min((current_storage_gb / quota.max_storage_gb) * 100, 100.0)
    
    @staticmethod
    def is_domain_available(domain: str, existing_domains: List[str]) -> bool:
        """Check if domain is available for use."""
        return domain.lower() not in [d.lower() for d in existing_domains]
    
    @staticmethod
    def generate_subdomain(tenant_name: str) -> str:
        """Generate a subdomain from tenant name."""
        # Convert to lowercase, replace spaces and special chars with hyphens
        subdomain = re.sub(r'[^a-zA-Z0-9\-]', '-', tenant_name.lower())
        # Remove multiple consecutive hyphens
        subdomain = re.sub(r'-+', '-', subdomain)
        # Remove leading/trailing hyphens
        subdomain = subdomain.strip('-')
        # Limit length
        return subdomain[:63] if len(subdomain) > 63 else subdomain