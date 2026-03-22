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
| `LLM_API_KEY` | Yes (not ollama) | API key for the LLM provider |
| `JWT_SECRET_KEY` | Yes | Secret for JWT token signing |
