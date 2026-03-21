"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import get_settings
from app.core.logging_config import setup_logging
from app.core.observability import setup_langsmith
from app.core.middleware import RequestContextMiddleware
from app.core.exceptions import generic_exception_handler
from app.db.session import init_db
from app.db.seed import run_seeds

from app.api.health_routes import router as health_router
from app.api.auth_routes import router as auth_router
from app.api.ingestion_routes import router as ingestion_router
from app.api.query_routes import router as query_router
from app.api.admin_routes import router as admin_router

settings = get_settings()
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging("DEBUG" if settings.DEBUG else "INFO")
    setup_langsmith()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize database and seed
    await init_db()
    await run_seeds()
    logger.info("Database initialized and seeded")

    # Initialize Redis cache
    from app.core.cache import get_cache
    try:
        cache = await get_cache()
        logger.info("Redis cache connected")
    except Exception as e:
        logger.warning(f"Redis cache unavailable: {e}")

    yield

    # Shutdown: close Redis
    try:
        from app.core.cache import _cache
        if _cache:
            await _cache.close()
    except Exception:
        pass

    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="BookStack RAG Agent with LangGraph, LangSmith, and RBAC",
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

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
