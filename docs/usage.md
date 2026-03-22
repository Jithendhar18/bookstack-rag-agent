# Usage Guide

## First-Time Setup

After starting the services, seed the database to create default roles and admin user:

```bash
python scripts/seed_db.py
```

Default admin credentials:
- **Username**: `admin`
- **Password**: `admin1234` (change via `ADMIN_DEFAULT_PASSWORD` env var)

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
TOKEN=$(curl -s http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin1234"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 3. Trigger ingestion

```bash
# Ingest all pages
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"bookstack_type": "pages"}'

# Ingest specific pages by ID
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"bookstack_type": "pages", "bookstack_ids": [1, 5, 10]}'

# Force re-embed all pages (ignores content hash)
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"bookstack_type": "pages", "force_reindex": true}'
```

The response returns a `task_id`:
```json
{"task_id": "uuid", "status": "queued", "documents_queued": -1, "message": "Ingestion task started"}
```

### 4. Poll ingestion status

```bash
TASK_ID="<task_id from above>"
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/ingestion/status/$TASK_ID
```

## Querying

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I set up authentication?"}'
```

Optional parameters:

```json
{
  "query": "How do I set up authentication?",
  "session_id": "<uuid>",
  "top_k": 5
}
```

Provide `session_id` from a previous response to continue a conversation.

## Resetting the Database & Vector Store

To completely clear all data (PostgreSQL + Qdrant):

```bash
./scripts/reset_db.sh
```

This script:
- Drops and recreates the PostgreSQL database
- Deletes the Qdrant collection (`bookstack_documents`)
- Runs all pending migrations
- Re-seeds default roles and admin user

After reset, you must re-ingest your documentation.

### Clear vectors only (keep PostgreSQL)

To delete all chunks in Qdrant without touching database records:

```bash
cd backend && python -c "
from app.retrieval.vector_store import VectorStoreManager
manager = VectorStoreManager()
manager.client.delete_collection('bookstack_documents')
print('✓ Qdrant collection deleted')
"
```

Then re-ingest to recreate the vectors.

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
