"""LangSmith observability integration."""

import os
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def setup_langsmith():
    """Configure LangSmith tracing via environment variables."""
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        logger.info(f"LangSmith tracing enabled for project: {settings.LANGCHAIN_PROJECT}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LangSmith API key not set — tracing disabled")
