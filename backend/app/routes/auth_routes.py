"""Authentication API routes - HTTP layer with dependency injection."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.auth.dependencies import get_current_user, CurrentUser
from app.schemas.schemas import (
    LoginRequest,
    RegisterRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    """
    Dependency injection for AuthService.
    
    Args:
        db: Database session
        
    Returns:
        AuthService instance
    """
    return AuthService(db)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
    """
    Authenticate user and return JWT tokens.
    
    Args:
        request: Login credentials (username, password)
        http_request: HTTP request object for IP logging
        auth_service: Injected auth service
        
    Returns:
        TokenResponse with access_token, refresh_token, and expiry
    """
    # Authenticate user
    user, role = await auth_service.authenticate_user(
        request.username, request.password
    )

    # Generate tokens
    tokens = await auth_service.generate_tokens(user.id, role.name, user.tenant_id)

    # Log login event
    await auth_service.log_login(
        user.id,
        user.tenant_id,
        ip_address=http_request.client.host if http_request.client else None,
    )

    return TokenResponse(**tokens)


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: RegisterRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
    """
    Register a new user with default 'user' role.
    
    Args:
        request: Registration data (email, username, password, full_name, tenant_id)
        auth_service: Injected auth service
        
    Returns:
        UserResponse with created user details
    """
    user = await auth_service.register_user(
        email=request.email,
        username=request.username,
        password=request.password,
        full_name=request.full_name,
        tenant_id=request.tenant_id,
    )

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        role="user",
        tenant_id=user.tenant_id,
        created_at=user.created_at,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
    """
    Get new access token using refresh token.
    
    Args:
        request: RefreshTokenRequest with refresh_token
        auth_service: Injected auth service
        
    Returns:
        TokenResponse with new access_token, refresh_token, and expiry
    """
    tokens = await auth_service.refresh_access_token(request.refresh_token)
    return TokenResponse(**tokens)


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
):
    """
    Get current user profile from JWT token.
    
    Args:
        current_user: Current user from JWT token
        auth_service: Injected auth service
        
    Returns:
        UserResponse with current user details
    """
    user = await auth_service.get_user_by_id(current_user.user_id)
    role = await auth_service.get_user_role(current_user.user_id)

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        role=role.name,
        tenant_id=user.tenant_id,
        created_at=user.created_at,
    )
