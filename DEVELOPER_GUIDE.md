# Developer Guide

## Project Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) Ollama for local LLM

### Local Development

```bash
# 1. Start infrastructure
docker compose up -d postgres redis qdrant

# 2. Create virtualenv
cd backend
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env   # (or copy an example from examples/)
# Edit .env with your values

# 5. Run migrations
alembic upgrade head

# 6. Seed database
python scripts/seed_db.py

# 7. Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Full Docker Setup

```bash
docker compose up -d
```

---

## How the Pipeline Works

### Pipeline Execution Flow

Each query goes through a LangGraph state machine with 9 nodes:

```
┌─────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────┐
│  Input   │──▶│ QueryRewrite │──▶│ Retriever │──▶│ Reranker │
│  Node    │   │   (toggle)   │   │ (factory) │   │ (toggle) │
└─────────┘   └──────────────┘   └───────────┘   └──────────┘
                                                       │
┌──────────┐   ┌───────────────┐   ┌───────────┐      │
│ Response │◀──│   Response    │◀──│    LLM     │◀─────┘
│  (END)   │   │  Validator    │   │ Reasoning  │
└──────────┘   │   (toggle)    │   │ (fallback) │
               └───────────────┘   └────────┬──┘
                                            │
                                   ┌────────▼────────┐
                                   │    Context      │
                                   │  Compressor     │
                                   │    (toggle)     │
                                   └─────────────────┘
```

### Node Details

#### 1. Input Node
- Validates state has a query
- Runs guardrails check (prompt injection detection) if `GUARDRAILS_ENABLED=true`
- On guardrails failure: returns blocked response, skips rest of pipeline

#### 2. Query Rewrite Node
- **Toggle**: `QUERY_REWRITER_ENABLED`
- Uses primary LLM to rewrite the user query for better retrieval
- Failsafe: on error, passes original query through

#### 3. Retriever Node
- Mode selected by `RETRIEVAL_MODE`: `dense`, `keyword`, or `hybrid`
- `DenseRetriever`: Embeds query → vector similarity in Qdrant
- `KeywordRetriever`: Full-text search in Qdrant
- `HybridRetriever`: Both dense + keyword, merged via Reciprocal Rank Fusion (RRF)
- Number of results: `RETRIEVAL_TOP_K`

#### 4. Reranker Node
- **Toggle**: `RERANKER_ENABLED`
- Uses CrossEncoder model to rescore retrieved documents
- Keeps top `RERANKER_TOP_N` results
- When disabled: factory returns `NoOpReranker` (pass-through)

#### 5. Context Compressor Node
- **Toggle**: `CONTEXT_COMPRESSION_ENABLED`
- Deduplicates chunks by content hash
- Applies MMR (Maximal Marginal Relevance) for diversity
- Truncates to token limit (4096 tokens default)

#### 6. LLM Reasoning Node
- Constructs system prompt + context + query
- Calls primary LLM via factory
- **Fallback**: if primary fails and `LLM_FALLBACK_PROVIDER` is set, retries with fallback LLM
- Failsafe: on total failure, returns "unable to generate" message

#### 7. Response Validator Node
- **Toggle**: `GUARDRAILS_ENABLED`
- Checks if LLM response is grounded in retrieved documents
- Flags ungrounded responses

#### 8. Response Node
- Attaches metadata: latency, active modules, source count
- Logs timing information

---

## Adding a New LLM Provider

### Step 1: Create the Provider Class

Create `backend/app/providers/llm/my_provider.py`:

```python
from langchain_core.language_models import BaseChatModel
from app.providers.base import BaseLLM


class MyProviderLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, **kwargs):
        self._model = model
        # Initialize your LangChain chat model here
        self._client = ...  # your ChatModel

    async def generate(self, messages: list[dict]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        response = await self._client.ainvoke(lc_messages)
        return response.content

    async def stream(self, messages: list[dict]):
        # Similar to generate but yield chunks
        ...

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def langchain_client(self) -> BaseChatModel:
        return self._client
```

### Step 2: Register in Factory

Edit `backend/app/providers/factory.py`:

```python
def get_llm(force_new: bool = False) -> BaseLLM:
    ...
    if provider == "my_provider":
        from app.providers.llm.my_provider import MyProviderLLM
        _llm_instance = MyProviderLLM(
            model=settings.LLM_MODEL,
            api_key=settings.MY_PROVIDER_API_KEY,
        )
    ...
```

### Step 3: Add Config

Edit `backend/config.py`, add any new env vars:

```python
MY_PROVIDER_API_KEY: str = ""
```

### Step 4: Update Config Guide

Add the new provider to `CONFIG_GUIDE.md` under "LLM Provider".

That's it. No other code changes are needed — the pipeline, caching, fallback, and logging will work automatically.

---

## Adding a New Embedding Provider

Same pattern as LLM:

1. Create class implementing `BaseEmbedding` (with `embed()`, `embed_batch()`, `dimension`)
2. Register in `factory.py` under `get_embedding()`
3. Add env vars to config

---

## Adding a New Retrieval Strategy

1. Create class implementing `BaseRetriever` (with `retrieve()`)
2. Register in `factory.py` under `get_retriever()`
3. Add the mode name to `RETRIEVAL_MODE` options in `CONFIG_GUIDE.md`

---

## How Toggles Work

Each toggleable node follows this pattern:

```python
async def some_node(state: AgentState) -> dict:
    settings = get_settings()
    if not settings.SOME_FEATURE_ENABLED:
        logger.info("Feature disabled, passing through")
        return {}  # Return empty → state unchanged → next node
    
    try:
        # Do the actual work
        result = await do_work(state)
        return {"field": result}
    except Exception as e:
        logger.error(f"Feature failed: {e}")
        return {}  # Failsafe: pass through on error
```

**Conventions:**
- Returning `{}` from a node means "don't modify state" (LangGraph merge semantics)
- Errors in optional nodes are caught and logged, never crash the pipeline
- Only the input node can halt the pipeline (guardrails block)

---

## How the Factory Works

```python
# Singleton pattern with lazy initialization
_llm_instance: Optional[BaseLLM] = None

def get_llm(force_new: bool = False) -> BaseLLM:
    global _llm_instance
    if _llm_instance is not None and not force_new:
        return _llm_instance
    
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    
    if provider in ("openai", "openrouter", "groq"):
        from app.providers.llm.openai_compatible import OpenAICompatibleLLM
        _llm_instance = OpenAICompatibleLLM(...)
    elif provider == "ollama":
        from app.providers.llm.ollama import OllamaLLM
        _llm_instance = OllamaLLM(...)
    
    return _llm_instance
```

- **Singleton**: Heavy models (embedding, reranker) are loaded once
- **Lazy imports**: Provider modules only imported when selected
- **`force_new=True`**: Recreate instance (useful for testing or hot-reload)

---

## Database Migrations

```bash
cd backend

# Create a migration
alembic revision --autogenerate -m "add user preferences table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Testing

```bash
cd backend

# Run all tests
pytest

# Run specific test file
pytest tests/test_providers.py

# Run with coverage
pytest --cov=app --cov-report=html
```

---

## Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── config.py            # All environment variables + Settings class
├── alembic.ini          # Database migration config
│
├── app/
│   ├── providers/       # ★ Pluggable component system
│   │   ├── base.py      # Abstract interfaces
│   │   ├── factory.py   # Factory functions
│   │   ├── llm/         # LLM implementations
│   │   ├── embeddings/  # Embedding implementations
│   │   ├── rerankers/   # Reranker implementations
│   │   └── retrievers/  # Retriever implementations
│   │
│   ├── agents/          # LangGraph pipeline
│   │   ├── graph.py     # Graph structure
│   │   ├── nodes.py     # Node functions
│   │   ├── state.py     # State schema
│   │   └── tools.py     # Agent tools
│   │
│   ├── api/             # FastAPI routes
│   ├── auth/            # JWT + RBAC
│   ├── core/            # Middleware, logging, guardrails
│   ├── db/              # SQLAlchemy models + session
│   ├── embeddings/      # Backward-compat embedding service
│   ├── ingestion/       # BookStack → Qdrant pipeline
│   ├── retrieval/       # Retrieval service + vector store
│   └── schemas/         # Pydantic request/response schemas
│
├── scripts/             # CLI utilities
└── docker/              # Dockerfile
```
