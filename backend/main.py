"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from app.core.logging_config import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.core.exceptions import generic_exception_handler
from app.db.session import init_db
from app.db.seed import run_seeds

from app.routes.health_routes import router as health_router
from app.routes.auth_routes import router as auth_router
from app.routes.ingestion_routes import router as ingestion_router
from app.routes.query_routes import router as query_router
from app.routes.admin_routes import router as admin_router

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging("DEBUG" if settings.DEBUG else "INFO")
    logger.info("Starting BookStack RAG Agent")

    # Run Alembic migrations
    import os
    if os.getenv("AUTO_MIGRATE", "true").lower() == "true":
        await init_db()
        setup_logging("DEBUG" if settings.DEBUG else "INFO")
        logger.info("Database migrations applied")

    # Seed default roles & admin user
    await run_seeds()
    logger.info("Database seeded")

    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="BookStack RAG Agent",
        version="1.0.0",
        description="BookStack RAG Agent with LangGraph",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestContextMiddleware)
    allowed_origins = (
        ["*"]
        if settings.DEBUG
        else [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    app.add_exception_handler(Exception, generic_exception_handler)

    # Routes
    app.include_router(health_router)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(ingestion_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
