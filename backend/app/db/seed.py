"""Database seeding: default roles and permissions, admin user."""

import asyncio
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal, init_db
from app.db.models import Role, Permission, User, RoleName
from app.auth.password import hash_password


ROLE_PERMISSIONS = {
    RoleName.ADMIN: [
        ("ingestion", "read"), ("ingestion", "write"), ("ingestion", "delete"),
        ("query", "read"), ("query", "write"),
        ("admin", "read"), ("admin", "write"), ("admin", "delete"),
        ("users", "read"), ("users", "write"), ("users", "delete"),
    ],
    RoleName.DEVELOPER: [
        ("ingestion", "read"), ("ingestion", "write"),
        ("query", "read"), ("query", "write"),
        ("admin", "read"),
        ("users", "read"),
    ],
    RoleName.USER: [
        ("query", "read"), ("query", "write"),
    ],
}


async def seed_roles_and_permissions(db: AsyncSession):
    """Create default roles and their permissions if they don't exist."""
    for role_name, perms in ROLE_PERMISSIONS.items():
        result = await db.execute(select(Role).where(Role.name == role_name))
        role = result.scalar_one_or_none()
        if role is None:
            role = Role(id=uuid.uuid4(), name=role_name, description=f"{role_name.value} role")
            db.add(role)
            await db.flush()

        for resource, action in perms:
            result = await db.execute(
                select(Permission).where(
                    Permission.role_id == role.id,
                    Permission.resource == resource,
                    Permission.action == action,
                )
            )
            if result.scalar_one_or_none() is None:
                db.add(Permission(
                    id=uuid.uuid4(),
                    role_id=role.id,
                    resource=resource,
                    action=action,
                ))

    await db.commit()


async def seed_admin_user(db: AsyncSession):
    """Create default admin user if it doesn't exist."""
    result = await db.execute(select(User).where(User.username == "admin"))
    if result.scalar_one_or_none() is not None:
        return

    result = await db.execute(select(Role).where(Role.name == RoleName.ADMIN))
    admin_role = result.scalar_one_or_none()
    if admin_role is None:
        return

    admin = User(
        id=uuid.uuid4(),
        email="admin@bookstack-rag.local",
        username="admin",
        hashed_password=hash_password("admin1234"),
        full_name="System Admin",
        is_active=True,
        tenant_id="default",
        role_id=admin_role.id,
    )
    db.add(admin)
    await db.commit()


async def run_seeds():
    """Initialize DB and run all seeders."""
    await init_db()
    async with AsyncSessionLocal() as db:
        await seed_roles_and_permissions(db)
        await seed_admin_user(db)


if __name__ == "__main__":
    asyncio.run(run_seeds())
