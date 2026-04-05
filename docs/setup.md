# Setup Guide

## Project Structure

```
bookstack-rag-agent/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── routes/            # ⭐ HTTP handlers (thin layer)
│   │   │   ├── __init__.py
│   │   │   ├── health_routes.py      # GET /health, /health/detailed
│   │   │   ├── auth_routes.py        # POST /login, /register, /refresh + GET /me
│   │   │   ├── query_routes.py       # POST /query, /query/stream
│   │   │   ├── ingestion_routes.py   # POST /ingest, GET /status, /documents, /books/*
│   │   │   └── admin_routes.py       # GET /metrics, /users; PATCH /users/{id}
│   │   │
│   │   ├── services/          # ⭐ Business logic layer
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # BaseService (async session injection)
│   │   │   ├── auth_service.py       # Auth ops (register, login, token refresh)
│   │   │   ├── query_service.py      # Chat sessions & message management
│   │   │   ├── ingestion_service.py  # Ingestion validation & doc listing
│   │   │   └── admin_service.py      # Metrics, user management, reports
│   │   │
│   │   ├── repositories/      # ⭐ Data access layer (SQL queries only)
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # BaseRepository[T] generic CRUD
│   │   │   ├── user_repository.py    # User-specific queries
│   │   │   ├── role_repository.py    # Role queries
│   │   │   ├── document_repository.py # Document & Chunk queries
│   │   │   ├── chat_repository.py    # ChatSession & ChatMessage queries
│   │   │   └── audit_log_repository.py # Audit queries
│   │   │
│   │   ├── agents/            # LangGraph RAG pipeline
│   │   │   ├── graph.py
│   │   │   ├── nodes.py       # 8 pipeline nodes
│   │   │   └── state.py       # AgentState TypedDict
│   │   │
│   │   ├── auth/              # JWT, password hashing, auth guards
│   │   │   ├── dependencies.py # CurrentUser, require_roles()
│   │   │   ├── jwt_handler.py
│   │   │   └── password.py
│   │   │
│   │   ├── core/              # Utilities & cross-cutting concerns
│   │   │   ├── cache.py       # In-memory cache (TTL)
│   │   │   ├── exceptions.py  # Custom exceptions
│   │   │   ├── guardrails.py  # Prompt injection, grounding
│   │   │   ├── logging_config.py
│   │   │   ├── middleware.py  # RequestContext, CORS
│   │   │
│   │   ├── db/                # Database layer
│   │   │   ├── models.py      # SQLAlchemy ORM (User, Document, Chunk, etc.)
│   │   │   ├── session.py     # Async session factory
│   │   │   └── seed.py        # Database seeding (default roles, admin user)
│   │   │
│   │   ├── ingestion/         # BookStack connector & chunking
│   │   │   ├── bookstack_client.py # BookStack API client
│   │   │   ├── content_parser.py   # HTML → text normalization
│   │   │   ├── chunker.py         # Semantic chunking
│   │   │   └── pipeline.py        # Orchestration
│   │   │
│   │   ├── providers/         # LLM, embeddings, rerankers (pluggable)
│   │   │   ├── factory.py     # Provider singletons
│   │   │   ├── base.py        # Abstract base classes
│   │   │   ├── llm/           # OpenAI, Ollama, OpenRouter
│   │   │   ├── embeddings/    # SentenceTransformers
│   │   │   ├── rerankers/     # CrossEncoder
│   │   │   └── retrievers/    # Dense, keyword, hybrid search
│   │   │
│   │   ├── retrieval/         # Vector store management
│   │   │   └── vector_store.py # Qdrant client wrapper
│   │   │
│   │   ├── schemas/           # Pydantic v2 models
│   │   │   └── schemas.py     # All request/response types
│   │   │
│   │   └── __init__.py
│   │
│   ├── main.py                # FastAPI app factory & startups
│   ├── config.py              # Settings loader (environment variables)
│   ├── requirements.txt
│   ├── alembic/               # Database migrations
│   └── .env.example
│
├── docs/                      # Documentation
│   ├── setup.md              # This file
│   ├── api.md                # Endpoint reference
│   ├── architecture.md       # System design & data flow
│   ├── frontend-integration.md # React/TypeScript integration guide
│   ├── technical-review.md
│   └── usage.md
│
├── docker/
│   └── Dockerfile
│
├── scripts/
│   ├── reset_db.sh           # Clear database + vector store
│   └── seed_db.py            # Populate default data
│
├── docker-compose.yml        # Postgres + Qdrant
├── README.md
└── venv/                      # Python virtual environment
```

## Architecture Layers

The backend follows **3-layer clean architecture**:

### 1. Routes Layer (`app/routes/`)
- **Responsibility**: HTTP handling only
- **What they do**: Parse requests, call services, serialize responses
- **No database access**: All DB calls go through services
- **Examples**: `auth_routes.py`, `ingestion_routes.py`

### 2. Services Layer (`app/services/`)
- **Responsibility**: Business logic & orchestration
- **What they do**: Validate, transform data, call repositories, manage transactions
- **Examples**: `AuthService` (login/register), `IngestionService` (document listing)

### 3. Repositories Layer (`app/repositories/`)
- **Responsibility**: Data access (SQL queries only)
- **What they do**: `SELECT`, `INSERT`, `UPDATE`, `DELETE` operations
- **One class per model**: `UserRepository`, `DocumentRepository`, `ChunkRepository`
- **Examples**: `get_by_id()`, `get_documents_paginated()`, `count_by_tenant()`

---

## Key Changes from Old Code

| Aspect | Old | New |
|--------|-----|-----|
| **Routes location** | `app/api/` (deleted) | `app/routes/` ✅ |
| **Services** | Mixed in routes | `app/services/` (dedicated) ✅ |
| **Repositories** | Didn't exist | `app/repositories/` (new) ✅ |
| **Database queries** | In services/routes | In repositories only ✅ |
| **Testing** | Hard to unit test | Easy - each layer is isolated ✅ |

---

## Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Docker & Docker Compose (recommended)

## Quick Start (Docker)

```bash
# Clone and configure
cp backend/.env.example backend/.env
# Edit backend/.env with your BookStack and LLM credentials

# Start all services
docker compose up -d

# Seed the database (first time only)
python scripts/seed_db.py
```

The services will be available at:
- **API**: http://localhost:8000
- **API docs**: http://localhost:8000/docs
- **Qdrant dashboard**: http://localhost:6333/dashboard

## Local Development (without Docker)

### 1. Start PostgreSQL and Qdrant

```bash
# Start only infrastructure services
docker compose up -d db qdrant
```

### 2. Set up Python environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run migrations and seed

```bash
cd backend
PYTHONPATH=. alembic upgrade head
python -c "import asyncio; from app.db.seed import run_seeds; asyncio.run(run_seeds())"
```

### 5. Start the server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

See [backend/.env.example](../backend/.env.example) for all available settings.

Key variables:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL async connection string |
| `BOOKSTACK_BASE_URL` | For ingestion | Your BookStack instance URL |
| `BOOKSTACK_TOKEN_ID` | For ingestion | BookStack API token ID |
| `BOOKSTACK_TOKEN_SECRET` | For ingestion | BookStack API token secret |
| `LLM_PROVIDER` | Yes | LLM provider: `openai`, `openrouter`, `groq`, `ollama` |
| `LLM_MODEL` | Yes | Model name (e.g. `llama-3.3-70b-versatile` for Groq) |
| `LLM_API_KEY` | Yes (not ollama) | API key for the LLM provider |
| `JWT_SECRET_KEY` | Yes | Secret for JWT token signing |
| `JWT_ALGORITHM` | No | JWT signing algorithm (default: `HS256`) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | Access token lifetime in minutes (default: `30`) |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | Refresh token lifetime in days (default: `7`) |
| `ADMIN_DEFAULT_PASSWORD` | No | Initial admin password (default: `admin1234`) |
| `RETRIEVAL_MODE` | No | `dense`, `keyword`, or `hybrid` (default: `hybrid`) |
| `RERANKER_ENABLED` | No | Enable cross-encoder reranking (default: `true`) |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins (default: `http://localhost:5173,http://localhost:3000`) |

## Reset Database & Vector Store

To clear all data and start fresh:

```bash
bash scripts/reset_db.sh
```

This script will:
1. Drop and recreate the PostgreSQL database
2. Delete the Qdrant collection (`bookstack_documents`)
3. Run all Alembic migrations
4. Seed default roles and admin user

The Qdrant collection will be automatically recreated with proper indexes on the next ingestion.

**Environment variables used:**
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DB_HOST`, `DB_PORT` — PostgreSQL
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME` — Qdrant

## Troubleshooting

### Port conflicts

If ports are already in use, update in `docker-compose.yml`:

```yaml
services:
  db:
    ports:
      - "5435:5432"  # Change 5435 to your desired port
  qdrant:
    ports:
      - "6333:6333"  # Change 6333 to your desired port
```

Then update `backend/.env`:
```
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:YOUR_DB_PORT/bookstack_rag
QDRANT_HOST=localhost
QDRANT_PORT=YOUR_QDRANT_PORT
```

### Health check

Verify all services are running:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
curl -s http://localhost:6333/health | jq .
```

### Docker networking

Docker Compose overrides `DATABASE_URL` and `QDRANT_HOST` to use internal service names (`db`, `qdrant`). The values in `.env` are only used for local (non-Docker) development.

If your BookStack instance runs on the host machine (not in Docker), use `host.docker.internal` instead of `localhost` in Docker mode:

```
BOOKSTACK_BASE_URL=http://host.docker.internal:6875
```

### Ingestion rate limits

BookStack enforces API rate limits. The ingestion pipeline includes built-in throttling (0.25s between requests) and retries with exponential backoff on 429 responses. A full ingestion of ~1,500 pages takes approximately 6-8 minutes.

### Logs not appearing

If background task logs (e.g., ingestion pipeline) don't appear in `docker compose logs`, ensure the Alembic `env.py` uses `disable_existing_loggers=False` in the `fileConfig()` call. This prevents Alembic's logging setup from silencing application loggers.

## Frontend Setup

The React frontend lives in the `ui-vector/` directory.

```bash
cd ui-vector
pnpm install
pnpm dev           # Starts on http://localhost:8080
```

The frontend expects the backend API at `http://localhost:8000/api/v1`. See [ui-vector/README.md](../ui-vector/README.md) for full details.
