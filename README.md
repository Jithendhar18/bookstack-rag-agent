# BookStack RAG Agent v3.0

A **fully configurable, modular AI platform** that answers questions from your BookStack documentation using Retrieval-Augmented Generation (RAG).

Every pipeline component — LLM, embeddings, reranker, retriever, guardrails, cache — is **pluggable, toggleable, and switchable via `.env`** with zero code changes.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└──────────────┬───────────────────────────────────────────────────┘
               ▼
┌──────────────────────┐     ┌─────────────────────┐
│   Input + Guardrails │────▶│  Query Rewriter     │ (optional)
│   (prompt injection) │     │  (LLM-based)        │
└──────────────────────┘     └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │    Retriever         │
                             │ dense|hybrid|keyword │
                             └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │    Reranker          │ (optional)
                             │ cross-encoder        │
                             └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │ Context Compressor   │ (optional)
                             │ dedup + MMR + trim   │
                             └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │      LLM            │
                             │ OpenAI/OpenRouter/   │
                             │ Groq/Ollama          │
                             │ + automatic fallback │
                             └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │ Response Validator   │ (optional)
                             │ grounding check      │
                             └─────────┬───────────┘
                                       ▼
                             ┌─────────────────────┐
                             │     Response         │
                             │ + latency + metadata │
                             └──────────────────────┘
```

Every `(optional)` node is controlled by an env toggle and gracefully passes data through when disabled.

---

## Key Features

- **Multi-LLM support**: OpenAI, OpenRouter, Groq, Ollama (local) — switch with one env var
- **Automatic LLM fallback**: If primary fails, falls back to secondary provider
- **Pluggable embeddings**: Local SentenceTransformer or OpenAI API
- **Toggleable reranker**: Cross-encoder reranking on/off
- **Retrieval strategies**: Dense, hybrid (RRF), or keyword — switchable via env
- **AI Profiles**: `cheap`, `balanced`, `best` — one setting configures everything
- **Guardrails**: Prompt injection detection + output grounding validation
- **LangSmith tracing**: Full observability across all pipeline stages
- **Multi-tenancy**: Tenant-scoped data isolation across all layers
- **Redis caching**: Query + retrieval result caching
- **Async ingestion**: Celery workers for BookStack content ingestion

---

## Quick Start

### 1. Docker (Recommended)

```bash
# Clone and configure
cp backend/.env.example backend/.env
# Edit .env with your settings

# Start all services
docker-compose up -d

# The API is available at http://localhost:8001
```

### 2. Local Development

```bash
# Start infrastructure
docker-compose up -d db redis qdrant

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env

# Run the server
python main.py
```

### 3. Using AI Profiles (Fastest Setup)

Set a single variable to configure the entire pipeline:

```env
# 100% free local setup
AI_PROFILE=cheap
OLLAMA_BASE_URL=http://localhost:11434

# Balanced cost/quality
AI_PROFILE=balanced
OPENROUTER_API_KEY=sk-or-...

# Maximum quality
AI_PROFILE=best
OPENROUTER_API_KEY=sk-or-...
```

---

## Configuration Overview

All configuration is via environment variables in `.env`. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for full details.

### Core Toggles

| Variable | Default | Description |
|---|---|---|
| `AI_PROFILE` | *(empty)* | `cheap` / `balanced` / `best` — sets smart defaults |
| `LLM_PROVIDER` | `openai` | `openai` / `openrouter` / `groq` / `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name for the provider |
| `EMBEDDING_PROVIDER` | `local` | `local` / `openai` |
| `RETRIEVAL_MODE` | `hybrid` | `dense` / `hybrid` / `keyword` |
| `RERANKER_ENABLED` | `true` | Enable/disable cross-encoder reranking |
| `QUERY_REWRITER_ENABLED` | `true` | Enable/disable LLM query rewriting |
| `CONTEXT_COMPRESSION_ENABLED` | `true` | Enable/disable context compression |
| `GUARDRAILS_ENABLED` | `true` | Enable/disable safety guardrails |
| `CACHE_ENABLED` | `true` | Enable/disable Redis caching |

### Switching Providers (No Code Changes)

```env
# Switch LLM to Groq
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

# Switch LLM to local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3

# Switch to dense-only retrieval
RETRIEVAL_MODE=dense

# Disable reranker for speed
RERANKER_ENABLED=false
```

---

## Project Structure

```
backend/
├── config.py                    # Central config with AI profiles
├── main.py                      # FastAPI entrypoint
├── app/
│   ├── providers/               # Pluggable provider system
│   │   ├── base.py              # Abstract interfaces (BaseLLM, BaseEmbedding, etc.)
│   │   ├── factory.py           # Factory functions (get_llm, get_embedding, etc.)
│   │   ├── llm/                 # LLM providers
│   │   ├── embeddings/          # Embedding providers
│   │   ├── rerankers/           # Reranker providers
│   │   └── retrievers/          # Retrieval strategies
│   ├── agents/                  # LangGraph pipeline
│   ├── api/                     # FastAPI routes
│   ├── auth/                    # JWT + RBAC
│   ├── core/                    # Cache, guardrails, observability
│   ├── db/                      # PostgreSQL models
│   ├── embeddings/              # Embedding service (delegates to provider)
│   ├── ingestion/               # BookStack content pipeline
│   ├── retrieval/               # Vector store + retrieval service
│   └── schemas/                 # Pydantic models
```

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/login` | Login → JWT tokens |
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| GET | `/api/v1/auth/me` | Current user profile |

### Query

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/query` | Submit RAG query |
| POST | `/api/v1/query/stream` | Stream query via SSE |

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/ingestion/ingest` | Start ingestion task |
| GET | `/api/v1/ingestion/status/{id}` | Check task status |
| GET | `/api/v1/ingestion/documents` | List documents |

### Admin

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/admin/metrics` | System metrics |
| GET | `/api/v1/admin/users` | List users |
| POST | `/api/v1/admin/evaluate` | Run evaluation |

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/health/detailed` | Component health |

---

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, component responsibilities |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | All env variables, profiles, example scenarios |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | How to extend, add providers, modify pipeline |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

---

## License

MIT
