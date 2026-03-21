"""FastAPI dependencies for authentication and RBAC."""

from typing import List
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.jwt_handler import decode_token
from app.db.session import get_db
from app.db.models import User, Role, Permission

security = HTTPBearer()


class CurrentUser:
    """Resolved user context from JWT."""
    def __init__(self, user_id: UUID, role: str, tenant_id: str):
        self.user_id = user_id
        self.role = role
        self.tenant_id = tenant_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """Validate JWT and return current user context."""
    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return CurrentUser(
        user_id=UUID(user_id),
        role=payload.get("role"),
        tenant_id=payload.get("tenant_id"),
    )


def require_roles(allowed_roles: List[str]):
    """Dependency factory that checks if the current user has one of the allowed roles."""
    async def _check_role(current_user: CurrentUser = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not authorized. Required: {allowed_roles}",
            )
        return current_user
    return _check_role


def require_permission(resource: str, action: str):
    """Dependency factory that checks if the user's role has a specific permission."""
    async def _check_permission(
        current_user: CurrentUser = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ):
        result = await db.execute(
            select(Permission)
            .join(Role)
            .join(User)
            .where(
                User.id == current_user.user_id,
                Permission.resource == resource,
                Permission.action == action,
            )
        )
        perm = result.scalar_one_or_none()
        if perm is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {resource}:{action}",
            )
        return current_user
    return _check_permission
