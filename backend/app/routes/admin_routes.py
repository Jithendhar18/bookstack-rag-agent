"""Admin API routes - HTTP handlers with dependency injection."""

import logging
from typing import List, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.auth.dependencies import require_roles, CurrentUser
from app.services.admin_service import AdminService
from app.schemas.schemas import SystemMetrics, UserResponse, UserUpdateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


async def get_admin_service(db: AsyncSession = Depends(get_db)) -> AdminService:
    """Dependency injection for AdminService."""
    return AdminService(db)


@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics(
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin"]))],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
):
    """
    Get system metrics and analytics.
    
    Returns aggregated metrics for the tenant including document counts,
    chunk counts, user counts, query statistics, and average latency.
    Admin access required.
    
    Args:
        current_user: Current authenticated user
        admin_service: Injected admin service
        
    Returns:
        SystemMetrics with all system statistics
    """
    metrics = await admin_service.get_system_metrics(current_user.tenant_id)
    return SystemMetrics(**metrics)


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin"]))],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
    page: int = 1,
    page_size: int = 20,
):
    """
    List users in the tenant.
    
    Retrieve paginated list of users with their roles.
    Admin access required.
    
    Args:
        page: Page number (1-indexed)
        page_size: Users per page
        current_user: Current authenticated user
        admin_service: Injected admin service
        
    Returns:
        List of UserResponse objects
    """
    skip = (page - 1) * page_size
    users = await admin_service.list_users(
        current_user.tenant_id, skip=skip, limit=page_size
    )
    
    responses = []
    for user in users:
        role = await admin_service.get_user_role(user.id)
        responses.append(
            UserResponse(
                id=user.id,
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                is_active=user.is_active,
                role=role.name if role else "user",
                tenant_id=user.tenant_id,
                created_at=user.created_at,
            )
        )
    
    return responses


@router.patch("/users/{user_id}", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: Annotated[CurrentUser, Depends(require_roles(["admin"]))],
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
):
    """
    Update user details.
    
    Update a user's profile information (full_name, is_active, role).
    Logs audit trail of the update. Admin access required.
    
    Args:
        user_id: UUID of user to update
        request: UserUpdateRequest with fields to update
        current_user: Current authenticated user
        admin_service: Injected admin service
        
    Returns:
        Updated UserResponse
    """
    # Convert string UUID to UUID object
    user_uuid = UUID(user_id)
    
    user = await admin_service.update_user(
        user_id=user_uuid,
        full_name=request.full_name,
        is_active=request.is_active,
        role=request.role,
        tenant_id=current_user.tenant_id,
        updated_by=current_user.user_id,
    )
    
    role = await admin_service.get_user_role(user.id)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        role=role.name if role else "user",
        tenant_id=user.tenant_id,
        created_at=user.created_at,
    )
