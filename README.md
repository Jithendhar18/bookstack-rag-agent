# BookStack RAG Agent

AI-powered Q&A over your BookStack documentation using Retrieval-Augmented Generation.

Built with **FastAPI**, **LangGraph**, **Qdrant**, and **PostgreSQL**.

## Features

- **Multi-LLM support** — OpenAI, Groq, OpenRouter, Ollama (switch with one env var)
- **Local embeddings** — SentenceTransformers (BAAI/bge-base-en-v1.5), no API cost
- **Hybrid retrieval** — Dense + keyword search with Reciprocal Rank Fusion
- **Cross-encoder reranking** — toggleable for precision vs speed
- **Guardrails** — prompt injection detection + output grounding validation
- **JWT auth with RBAC** — admin, developer, user roles
- **In-memory caching** — TTL-based query and retrieval caching

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d db qdrant

# 2. Configure
cd backend
cp .env.example .env
# Edit .env — set BookStack credentials and LLM API key

# 3. Install & run
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Seed database (first time)
cd .. && python scripts/seed_db.py
```

Or run everything with Docker:

```bash
docker compose up -d
```

**Default admin**: admin@bookstack-rag.local / admin1234

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
| POST | `/api/auth/login` | Login → JWT token |
| POST | `/api/auth/register` | Register user |
| POST | `/api/query` | RAG query |
| POST | `/api/query/stream` | Streaming RAG query |
| POST | `/api/ingestion/ingest` | Ingest from BookStack |
| GET | `/api/admin/stats` | System stats (admin) |
| GET | `/api/health` | Health check |

Full API docs at http://localhost:8000/docs

## Documentation

- [docs/setup.md](docs/setup.md) — Installation and setup
- [docs/architecture.md](docs/architecture.md) — System design
- [docs/api.md](docs/api.md) — API reference
- [docs/usage.md](docs/usage.md) — Usage guide

## License

MIT
