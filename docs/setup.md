# Setup Guide

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
