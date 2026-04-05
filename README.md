# BookStack RAG Agent

AI-powered Q&A over your BookStack documentation using Retrieval-Augmented Generation.

Built with **FastAPI**, **LangGraph**, **Qdrant**, and **PostgreSQL**.  
React + TypeScript frontend with real-time streaming, chat sessions, and admin dashboard.

## Features

- **Multi-LLM support** — OpenAI, Groq, OpenRouter, Ollama (switch with one env var)
- **Local embeddings** — SentenceTransformers (BAAI/bge-base-en-v1.5), no API cost
- **Hybrid retrieval** — Dense + keyword search with Reciprocal Rank Fusion
- **Cross-encoder reranking** — toggleable for precision vs speed
- **Guardrails** — prompt injection detection + output grounding validation
- **JWT auth with RBAC** — admin, developer, user roles
- **SSE streaming** — real-time token streaming with pipeline stage indicators
- **In-memory caching** — TTL-based query and retrieval caching
- **Rate-limited ingestion** — handles large BookStack instances without API throttling

## Quick Start

```bash
# 1. Start everything with Docker
docker compose up -d

# 2. Configure (first time)
cd backend && cp .env.example .env
# Edit .env — set BOOKSTACK_BASE_URL, BOOKSTACK_TOKEN_ID/SECRET, LLM_API_KEY

# 3. Rebuild after config changes
docker compose up -d --force-recreate backend
```

Or run locally:

```bash
# Backend
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd ui-vector && pnpm install && pnpm dev
```

**Default admin login**: username `admin` / password `admin1234`

## Query Pipeline (LangGraph)

```
Input → Guardrails → Query Rewrite → Retriever → Reranker
    → Context Compressor → LLM → Response Validator → Response
```

Each optional node is controlled by an env toggle and passes data through when disabled.

## Configuration

All settings via environment variables. See [backend/.env.example](backend/.env.example).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `openai` / `openrouter` / `groq` / `ollama` |
| `LLM_API_KEY` | — | API key for the LLM provider |
| `RETRIEVAL_MODE` | `hybrid` | `dense` / `hybrid` / `keyword` |
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking |
| `GUARDRAILS_ENABLED` | `true` | Input/output safety checks |
| `CACHE_ENABLED` | `true` | In-memory result caching |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/login` | Login → JWT token |
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/query` | RAG query (sync) |
| POST | `/api/v1/query/stream` | Streaming RAG query (SSE) |
| POST | `/api/v1/ingestion/ingest` | Ingest from BookStack |
| GET | `/api/v1/ingestion/status/{task_id}` | Ingestion task status |
| GET | `/api/v1/admin/stats` | System stats (admin) |
| GET | `/health` | Health check |

Interactive API docs at http://localhost:8000/docs

## Documentation

- [docs/setup.md](docs/setup.md) — Installation and setup
- [docs/architecture.md](docs/architecture.md) — System design
- [docs/api.md](docs/api.md) — API reference
- [docs/usage.md](docs/usage.md) — Usage guide
- [docs/technical-review.md](docs/technical-review.md) — Design decisions and trade-offs
- [docs/frontend-integration.md](docs/frontend-integration.md) — Frontend integration guide

## License

MIT
