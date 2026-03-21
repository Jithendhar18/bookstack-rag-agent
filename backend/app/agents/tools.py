"""Tool registry for the RAG agent — extensible tool system."""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from langsmith import traceable
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ToolSchema:
    """Schema describing a tool available to the agent."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_role: str = "user"


class ToolRegistry:
    """Central registry for agent tools."""

    def __init__(self):
        self._tools: Dict[str, "BaseTool"] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in tools."""
        self.register(DocumentSearchTool())
        self.register(MetadataFetchTool())
        self.register(DocumentListTool())

    def register(self, tool: "BaseTool"):
        self._tools[tool.schema.name] = tool
        logger.debug(f"Tool registered: {tool.schema.name}")

    def get(self, name: str) -> Optional["BaseTool"]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    def list_tool_descriptions(self) -> str:
        """Format tool descriptions for LLM prompt."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.schema.name}: {t.schema.description}")
        return "\n".join(lines)

    @traceable(name="execute_tool")
    async def execute(self, tool_name: str, params: Dict[str, Any], db: Optional[AsyncSession] = None) -> Dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            result = await tool.run(params, db)
            return {"tool": tool_name, "result": result}
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return {"error": str(e)}


class BaseTool:
    """Base class for agent tools."""
    schema: ToolSchema

    async def run(self, params: Dict[str, Any], db: Optional[AsyncSession] = None) -> Any:
        raise NotImplementedError


class DocumentSearchTool(BaseTool):
    """Search documents by title or content keyword."""

    schema = ToolSchema(
        name="document_search",
        description="Search for documents by title keyword. Returns matching document titles and IDs.",
        parameters={"keyword": "string - search keyword"},
    )

    async def run(self, params: Dict[str, Any], db: Optional[AsyncSession] = None) -> Any:
        if db is None:
            return {"error": "Database session required"}

        from app.db.models import Document
        keyword = params.get("keyword", "")
        result = await db.execute(
            select(Document.id, Document.title, Document.bookstack_type)
            .where(Document.title.ilike(f"%{keyword}%"))
            .limit(10)
        )
        rows = result.all()
        return [{"id": str(r[0]), "title": r[1], "type": r[2]} for r in rows]


class MetadataFetchTool(BaseTool):
    """Fetch metadata for a specific document."""

    schema = ToolSchema(
        name="metadata_fetch",
        description="Get detailed metadata for a document by its ID.",
        parameters={"document_id": "string - UUID of the document"},
    )

    async def run(self, params: Dict[str, Any], db: Optional[AsyncSession] = None) -> Any:
        if db is None:
            return {"error": "Database session required"}

        from app.db.models import Document, Chunk
        doc_id = params.get("document_id")
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            return {"error": "Document not found"}

        chunk_count = (await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
        )).scalar() or 0

        return {
            "id": str(doc.id),
            "title": doc.title,
            "bookstack_id": doc.bookstack_id,
            "type": doc.bookstack_type,
            "status": doc.status.value if doc.status else "unknown",
            "chunk_count": chunk_count,
            "ingested_at": str(doc.ingested_at) if doc.ingested_at else None,
        }


class DocumentListTool(BaseTool):
    """List recent documents."""

    schema = ToolSchema(
        name="document_list",
        description="List recently ingested documents. Returns titles and statuses.",
        parameters={"limit": "integer - number of documents to return (default 10)"},
    )

    async def run(self, params: Dict[str, Any], db: Optional[AsyncSession] = None) -> Any:
        if db is None:
            return {"error": "Database session required"}

        from app.db.models import Document
        limit = min(params.get("limit", 10), 50)
        result = await db.execute(
            select(Document.id, Document.title, Document.status)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        rows = result.all()
        return [
            {"id": str(r[0]), "title": r[1], "status": r[2].value if r[2] else "unknown"}
            for r in rows
        ]


# Global registry singleton
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
