# Usage Guide

## First-Time Setup

After starting the services, seed the database to create default roles and admin user:

```bash
python scripts/seed_db.py
```

Default admin credentials:
- **Email**: admin@bookstack-rag.local
- **Password**: admin1234 (change via `ADMIN_DEFAULT_PASSWORD` env var)

## Ingesting Documentation

### 1. Configure BookStack credentials

Set these in `backend/.env`:
```
BOOKSTACK_BASE_URL=https://your-bookstack.com
BOOKSTACK_TOKEN_ID=your-token-id
BOOKSTACK_TOKEN_SECRET=your-token-secret
```

### 2. Login and get a token

```bash
TOKEN=$(curl -s http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@bookstack-rag.local","password":"admin1234"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 3. Trigger ingestion

```bash
# Ingest all pages
curl -X POST http://localhost:8000/api/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default"}'

# Ingest specific pages
curl -X POST http://localhost:8000/api/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"page_ids": [1, 5, 10]}'
```

## Querying

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I set up authentication?"}'
```

## Resetting the Database

```bash
./scripts/reset_db.sh
```

This drops and recreates the database, runs migrations, and re-seeds.

## Retrieval Modes

Configure via `RETRIEVAL_MODE` in `.env`:

| Mode | Description |
|---|---|
| `dense` | Vector similarity search only |
| `keyword` | Full-text keyword search (Qdrant text index) |
| `hybrid` | Combined dense + keyword with Reciprocal Rank Fusion (default) |

## LLM Providers

Configure via `LLM_PROVIDER` in `.env`:

| Provider | `LLM_PROVIDER` | `LLM_API_KEY` | `LLM_BASE_URL` |
|---|---|---|---|
| OpenAI | `openai` | Required | Auto |
| Groq | `groq` | Required | Auto |
| OpenRouter | `openrouter` | Required | Auto |
| Ollama | `ollama` | Not needed | Auto (localhost:11434) |
