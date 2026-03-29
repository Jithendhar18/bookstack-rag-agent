"""Authentication service - business logic for auth operations."""

import uuid
import logging
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from app.auth.password import hash_password, verify_password
from app.auth.jwt_handler import create_access_token, create_refresh_token, decode_token
from app.db.models import User, Role, AuditLog
from app.repositories.user_repository import UserRepository
from app.repositories.role_repository import RoleRepository
from app.services.base import BaseService
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AuthService(BaseService):
    """
    Authentication service handling login, registration, token management.
    
    Contains all auth business logic, independent of FastAPI.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize auth service with repositories.
        
        Args:
            db: SQLAlchemy AsyncSession instance
        """
        super().__init__(db)
        self.user_repo = UserRepository(db)
        self.role_repo = RoleRepository(db)

    async def authenticate_user(
        self, username: str, password: str
    ) -> tuple[User, Role]:
        """
        Authenticate user by username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            Tuple of (User, Role) if valid
            
        Raises:
            HTTPException: If credentials invalid
        """
        user = await self.user_repo.get_active_by_username(username)

        if user is None or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        # Get user's role
        role = await self.role_repo.get_by_id(user.role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User role not found",
            )

        return user, role

    async def generate_tokens(
        self, user_id: uuid.UUID, role_name: str, tenant_id: str
    ) -> dict[str, str]:
        """
        Generate access and refresh tokens for user.
        
        Args:
            user_id: User's UUID
            role_name: User's role name
            tenant_id: User's tenant ID
            
        Returns:
            Dictionary with access_token, refresh_token, expires_in, and token_type
        """
        access_token = create_access_token(user_id, role_name, tenant_id)
        refresh_token = create_refresh_token(user_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    async def log_login(
        self,
        user_id: uuid.UUID,
        tenant_id: str,
        ip_address: Optional[str] = None,
    ) -> AuditLog:
        """
        Create audit log entry for login.
        
        Args:
            user_id: User's UUID
            tenant_id: User's tenant ID
            ip_address: IP address of login request
            
        Returns:
            Created AuditLog instance
        """
        audit_log = AuditLog(
            id=uuid.uuid4(),
            user_id=user_id,
            action="login",
            resource="auth",
            ip_address=ip_address,
            tenant_id=tenant_id,
        )
        self.db.add(audit_log)
        await self.db.commit()
        return audit_log

    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str],
        tenant_id: str,
    ) -> User:
        """
        Register a new user.
        
        Args:
            email: User's email
            username: User's username
            password: User's password
            full_name: User's full name
            tenant_id: User's tenant ID
            
        Returns:
            Created User instance
            
        Raises:
            HTTPException: If user already exists or default role not found
        """
        # Check if user already exists
        existing_user = await self.user_repo.get_by_username_or_email(username, email)
        if existing_user is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists",
            )

        # Get default 'user' role
        user_role = await self.role_repo.get_by_name("user")
        if user_role is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Default role not found. Run seed first.",
            )

        # Create new user
        new_user = User(
            id=uuid.uuid4(),
            email=email,
            username=username,
            hashed_password=hash_password(password),
            full_name=full_name,
            tenant_id=tenant_id,
            role_id=user_role.id,
        )

        created_user = await self.user_repo.create(new_user)
        return created_user

    async def refresh_access_token(
        self, refresh_token: str
    ) -> dict[str, str]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Refresh token string
            
        Returns:
            Dictionary with new access_token, refresh_token, expires_in, and token_type
            
        Raises:
            HTTPException: If refresh token invalid or user not found
        """
        payload = decode_token(refresh_token)
        if payload is None or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        user_id = payload.get("sub")
        user = await self.user_repo.get_by_id(user_id)

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Get user's role
        role = await self.role_repo.get_by_id(user.role_id)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User role not found",
            )

        tokens = await self.generate_tokens(user.id, role.name, user.tenant_id)
        return tokens

    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """
        Retrieve user by ID.
        
        Args:
            user_id: User's UUID
            
        Returns:
            User instance or None if not found
        """
        return await self.user_repo.get_by_id(user_id)

    async def get_user_role(self, user_id: uuid.UUID) -> Optional[Role]:
        """
        Get role for a user.
        
        Args:
            user_id: User's UUID
            
        Returns:
            Role instance or None if not found
        """
        user = await self.user_repo.get_by_id(user_id)
        if user is None:
            return None
        return await self.role_repo.get_by_id(user.role_id)
