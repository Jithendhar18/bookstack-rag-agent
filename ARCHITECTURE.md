# Architecture Guide

## System Overview

BookStack RAG Agent v3.0 is a modular AI platform built on a **plugin architecture** where every pipeline component is swappable, toggleable, and configurable via environment variables.

---

## High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  BookStack   │────▶│  Ingestion   │────▶│  Qdrant    │
│  (docs)      │     │  Pipeline    │     │ (vectors)  │
└─────────────┘     └──────────────┘     └─────┬──────┘
                                               │
┌─────────────┐     ┌──────────────┐           │
│  Frontend    │────▶│  FastAPI     │◀──────────┘
│  (React)     │     │  + Auth      │
└─────────────┘     └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LangGraph   │
                    │  Pipeline    │
                    │  (9 nodes)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ LLM      │ │ Cache    │ │LangSmith │
        │ Provider  │ │ (Redis)  │ │ Tracing  │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Component Responsibilities

### 1. Provider System (`app/providers/`)

The core of the modular architecture. Every AI component is defined by an abstract interface and instantiated via factory functions.

```
providers/
├── base.py           # Abstract interfaces
│   ├── BaseLLM       # generate(), stream(), model_name
│   ├── BaseEmbedding # embed(), embed_batch(), dimension
│   ├── BaseReranker  # rerank()
│   ├── NoOpReranker  # Pass-through (disabled reranker)
│   └── BaseRetriever # retrieve()
│
├── factory.py        # Factory functions
│   ├── get_llm()     # → BaseLLM
│   ├── get_embedding()  # → BaseEmbedding
│   ├── get_reranker()   # → BaseReranker
│   ├── get_retriever()  # → BaseRetriever
│   ├── get_fallback_llm()  # → Optional[BaseLLM]
│   └── log_active_configuration()
│
├── llm/
│   ├── openai_compatible.py  # OpenAI, OpenRouter, Groq
│   └── ollama.py             # Local Ollama
│
├── embeddings/
│   ├── local.py    # SentenceTransformer (local)
│   └── openai.py   # OpenAI Embeddings API
│
├── rerankers/
│   └── cross_encoder.py  # CrossEncoder model
│
└── retrievers/
    └── strategies.py
        ├── DenseRetriever    # Vector similarity search
        ├── KeywordRetriever  # Full-text search
        └── HybridRetriever   # RRF merge of dense + keyword
```

### 2. LangGraph Pipeline (`app/agents/`)

A 9-node directed graph that processes queries through configurable stages:

```
Input → QueryRewrite → Retriever → Reranker → ContextCompressor
    → LLMReasoning → ResponseValidator → Response → END
```

**Key design decisions:**

- **Nodes don't instantiate their own components** — they call factory functions
- **Toggle logic is inside each node** — disabled nodes pass data through unchanged
- **Failsafe behavior** — errors in optional nodes don't crash the pipeline
- **LLM fallback** — if primary LLM fails, the reasoning node automatically tries the fallback provider

### 3. API Layer (`app/api/`)

FastAPI routes with JWT authentication and RBAC:

- **Query routes**: RAG query (sync + streaming via SSE)
- **Auth routes**: Login, register, token refresh
- **Ingestion routes**: Trigger BookStack content import
- **Admin routes**: Metrics, user management, evaluation
- **Health routes**: Liveness and component health checks

### 4. Ingestion Pipeline (`app/ingestion/`)

Async content ingestion from BookStack → PostgreSQL → Qdrant:

1. Fetch pages/books/chapters via BookStack API
2. Parse HTML to plain text
3. Deduplicate by content hash
4. Chunk text semantically
5. Embed chunks (via factory-provided embedding service)
6. Store vectors in Qdrant
7. Record metadata in PostgreSQL

Runs as Celery tasks via Redis broker.

### 5. Data Stores

| Store | Purpose | Technology |
|---|---|---|
| PostgreSQL | Users, documents, sessions, audit logs | SQLAlchemy + asyncpg |
| Qdrant | Vector embeddings + full-text index | qdrant-client |
| Redis | Query cache, Celery broker | redis.asyncio |

---

## Data Flow

### Query Processing

```
1. User sends POST /api/v1/query
2. JWT validated → CurrentUser extracted
3. Chat session created/loaded from PostgreSQL
4. Query dispatched to LangGraph pipeline:
   a. Input: validate + guardrails check
   b. QueryRewrite: LLM rewrites query (if enabled)
   c. Retriever: factory-selected strategy searches Qdrant
   d. Reranker: cross-encoder rescores (if enabled)
   e. ContextCompressor: dedup + MMR + token trim (if enabled)
   f. LLMReasoning: generate answer (with fallback)
   g. ResponseValidator: check grounding (if guardrails enabled)
   h. Response: attach latency + metadata
5. Cache result in Redis
6. Store messages in PostgreSQL
7. Return answer + sources + metadata
```

### Ingestion

```
1. Admin sends POST /api/v1/ingestion/ingest
2. Celery task created
3. Worker fetches pages from BookStack API
4. For each page:
   a. Parse HTML → text
   b. Compute content hash (skip unchanged)
   c. Chunk semantically
   d. Embed batch via factory
   e. Upsert to Qdrant
   f. Record in PostgreSQL
5. Invalidate Redis cache for tenant
```

---

## Design Decisions

### Why Factory Pattern?

- **Single point of change**: switch providers by changing one env var
- **Lazy initialization**: models load only when first needed
- **Singleton management**: heavy resources (LLM clients, embedding models) are shared
- **Fallback support**: factory can build alternative instances for failover

### Why Toggle Logic Inside Nodes?

Instead of conditionally adding/removing nodes from the graph:
- Graph structure stays deterministic and debuggable
- Disabled nodes simply pass data through (identity operation)
- No complex graph rewriting logic needed
- LangSmith traces show all nodes regardless (clear visibility)

### Why AI Profiles?

Profiles provide tested, coherent configurations:
- Users don't need to understand every setting to get started
- `cheap` is entry-level, `best` is maximum quality
- Individual overrides still work alongside profiles
- Reduces misconfiguration risk (e.g., wrong embedding dimension)

### Why NoOpReranker Instead of Conditional Edges?

A `NoOpReranker` that passes documents through:
- Keeps the pipeline structure consistent
- Makes A/B testing reranker impact trivial
- Simplifies error handling (always expect the same number of pipeline stages)
- Factory handles the decision, nodes don't need to know

---

## Security Architecture

- **JWT tokens** with asymmetric key + RBAC roles (admin, developer, user)
- **Prompt injection detection** via regex pattern matching (11+ patterns)
- **Output grounding validation** ensures answers are sourced from documents
- **Multi-tenancy** isolation via `tenant_id` filtering at every data layer
- **Audit logging** for login, query, ingestion, user management events
- **Rate limiting** via SlowAPI on all endpoints
