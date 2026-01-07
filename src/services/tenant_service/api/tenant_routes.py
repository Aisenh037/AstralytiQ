"""
Tenant management API routes.
"""
from typing import List, Optional, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from .schemas import (
    CreateTenantRequest, UpdateTenantRequest, UpdateSubscriptionRequest,
    TenantBrandingRequest, TenantSettingsRequest,
    TenantResponse, TenantSummaryResponse, TenantStatsResponse,
    TenantUsageResponse, MessageResponse, ValidationErrorResponse
)
from ..domain.entities import Tenant, TenantDomainService, TenantBranding
from ..infrastructure.repositories import TenantRepository, TenantUsageRepository
from ..infrastructure.quota import get_quota_enforcer, get_quota_monitor
from src.shared.infrastructure.database import get_postgres_session, get_redis_client
from src.shared.domain.models import SubscriptionPlan


router = APIRouter(prefix="/tenants", tags=["tenant-management"])


async def get_tenant_repository(
    session: AsyncSession = Depends(get_postgres_session)
) -> TenantRepository:
    """Get tenant repository dependency."""
    return TenantRepository(session)


async def get_tenant_usage_repository(
    session: AsyncSession = Depends(get_postgres_session)
) -> TenantUsageRepository:
    """Get tenant usage repository dependency."""
    redis_client = await get_redis_client()
    return TenantUsageRepository(session, redis_client)


@router.post("/", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: CreateTenantRequest,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Create a new tenant."""
    
    # Validate domain availability
    if request.domain:
        if await tenant_repository.domain_exists(request.domain):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain already exists"
            )
    
    # Create tenant
    try:
        tenant = Tenant.create_new_tenant(
            name=request.name,
            domain=request.domain,
            subscription_plan=request.subscription_plan,
            owner_id=request.owner_id
        )
        
        saved_tenant = await tenant_repository.save(tenant)
        return TenantResponse.model_validate(saved_tenant)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[TenantSummaryResponse])
async def list_tenants(
    subscription_plan: Optional[SubscriptionPlan] = Query(None, description="Filter by subscription plan"),
    search: Optional[str] = Query(None, description="Search by name or domain"),
    limit: int = Query(100, ge=1, le=1000, description="Number of tenants to return"),
    offset: int = Query(0, ge=0, description="Number of tenants to skip"),
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """List tenants with optional filtering."""
    
    if search:
        tenants = await tenant_repository.search_tenants(search, limit=limit, offset=offset)
    elif subscription_plan:
        tenants = await tenant_repository.get_by_subscription_plan(subscription_plan)
        tenants = tenants[offset:offset + limit]  # Apply pagination
    else:
        tenants = await tenant_repository.get_active_tenants(limit=limit, offset=offset)
    
    return [TenantSummaryResponse.model_validate(tenant) for tenant in tenants]


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Get tenant by ID."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    return TenantResponse.model_validate(tenant)


@router.put("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: UUID,
    request: UpdateTenantRequest,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Update tenant information."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Update fields
    if request.name:
        tenant.name = request.name
    
    if request.domain:
        # Check domain availability
        if request.domain != tenant.domain and await tenant_repository.domain_exists(request.domain):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain already exists"
            )
        tenant.domain = request.domain
    
    updated_tenant = await tenant_repository.save(tenant)
    return TenantResponse.model_validate(updated_tenant)


@router.put("/{tenant_id}/subscription", response_model=TenantResponse)
async def update_subscription(
    tenant_id: UUID,
    request: UpdateSubscriptionRequest,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Update tenant subscription plan."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    try:
        # Check if it's an upgrade or downgrade
        plan_hierarchy = {
            SubscriptionPlan.FREE: 0,
            SubscriptionPlan.BASIC: 1,
            SubscriptionPlan.PROFESSIONAL: 2,
            SubscriptionPlan.ENTERPRISE: 3
        }
        
        current_level = plan_hierarchy.get(tenant.subscription_plan, 0)
        new_level = plan_hierarchy.get(request.subscription_plan, 0)
        
        if new_level > current_level:
            # Upgrade
            tenant.upgrade_subscription(request.subscription_plan)
        elif new_level < current_level:
            # Downgrade
            warnings = tenant.downgrade_subscription(
                request.subscription_plan,
                force=request.force_downgrade
            )
            if warnings and not request.force_downgrade:
                return ValidationErrorResponse(
                    error="Downgrade would violate current usage",
                    details=warnings
                )
        
        updated_tenant = await tenant_repository.save(tenant)
        return TenantResponse.model_validate(updated_tenant)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{tenant_id}/branding", response_model=MessageResponse)
async def update_branding(
    tenant_id: UUID,
    request: TenantBrandingRequest,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Update tenant branding."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    try:
        branding = TenantBranding(
            logo_url=request.logo_url,
            primary_color=request.primary_color,
            secondary_color=request.secondary_color,
            company_name=request.company_name,
            custom_domain=request.custom_domain,
            favicon_url=request.favicon_url
        )
        
        tenant.update_branding(branding)
        await tenant_repository.save(tenant)
        
        return MessageResponse(
            message="Branding updated successfully",
            success=True
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{tenant_id}/settings", response_model=MessageResponse)
async def update_settings(
    tenant_id: UUID,
    request: TenantSettingsRequest,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Update tenant settings."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    tenant.update_settings(request.settings)
    await tenant_repository.save(tenant)
    
    return MessageResponse(
        message="Settings updated successfully",
        success=True
    )


@router.put("/{tenant_id}/activate", response_model=MessageResponse)
async def activate_tenant(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Activate tenant."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    tenant.activate()
    await tenant_repository.save(tenant)
    
    return MessageResponse(
        message=f"Tenant '{tenant.name}' activated successfully",
        success=True
    )


@router.put("/{tenant_id}/deactivate", response_model=MessageResponse)
async def deactivate_tenant(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Deactivate tenant."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    tenant.deactivate()
    await tenant_repository.save(tenant)
    
    return MessageResponse(
        message=f"Tenant '{tenant.name}' deactivated successfully",
        success=True
    )


@router.get("/{tenant_id}/stats", response_model=TenantStatsResponse)
async def get_tenant_stats(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Get tenant usage statistics."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    stats = await tenant_repository.get_tenant_stats(tenant_id)
    return TenantStatsResponse(**stats)


@router.get("/{tenant_id}/usage", response_model=TenantUsageResponse)
async def get_tenant_usage(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository),
    quota_monitor = Depends(get_quota_monitor)
):
    """Get tenant quota usage and warnings."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    usage_report = await quota_monitor.get_tenant_quota_report(tenant)
    
    return TenantUsageResponse(
        tenant_id=tenant.id,
        current_usage=usage_report.get("current_usage", {}),
        quota_limits=usage_report["quota_limits"],
        usage_percentages=usage_report["usage_percentages"],
        warnings=usage_report["warnings"],
        status=usage_report["status"]
    )


@router.get("/{tenant_id}/features", response_model=Dict[str, bool])
async def get_tenant_features(
    tenant_id: UUID,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Get tenant features based on subscription plan."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    features = tenant.get_features()
    return features.dict()


@router.get("/domain/{domain}", response_model=TenantResponse)
async def get_tenant_by_domain(
    domain: str,
    tenant_repository: TenantRepository = Depends(get_tenant_repository)
):
    """Get tenant by domain."""
    
    tenant = await tenant_repository.get_by_domain(domain)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found for domain"
        )
    
    return TenantResponse.model_validate(tenant)


@router.post("/{tenant_id}/check-quota/{resource_type}")
async def check_quota(
    tenant_id: UUID,
    resource_type: str,
    quantity: int = Query(1, ge=1, description="Requested quantity"),
    tenant_repository: TenantRepository = Depends(get_tenant_repository),
    quota_enforcer = Depends(get_quota_enforcer)
):
    """Check if tenant can create resources without violating quota."""
    
    tenant = await tenant_repository.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    can_proceed, violation = await quota_enforcer.check_quota(
        tenant, resource_type, quantity
    )
    
    if can_proceed:
        return MessageResponse(
            message=f"Can create {quantity} {resource_type}(s)",
            success=True
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Quota would be exceeded",
                "violation": violation.to_dict() if violation else None
            }
        )