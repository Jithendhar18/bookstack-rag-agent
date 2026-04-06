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

The pipeline runs in three modes depending on the request:

| Mode | When | BookStack API calls |
|------|------|---------------------|
| **Incremental** (default) | Prior ingestion exists | Only pages with `updated_at ≥ last_ingestion − 20 min` |
| **Full scan** | First ever ingestion (no DB records) | All pages |
| **Force reindex** | `force_reindex: true` | All pages, re-embeds regardless of content hash |

```bash
# Incremental update — fetches only pages changed since last run (with 20-min overlap)
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"bookstack_type": "pages"}'

# Ingest specific pages by ID
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"bookstack_type": "pages", "bookstack_ids": [1, 5, 10]}'

# Force re-embed all pages (ignores content hash, full scan)
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

The response includes the answer and source links:

```json
{
  "answer": "Authentication is configured by...",
  "session_id": "933d92b1-xxxx",
  "latency_ms": 2340.5,
  "sources": [
    {
      "document_title": "Auth Setup Guide",
      "source_url": "http://localhost:6875/books/my-book/page/auth-setup"
    }
  ]
}
```

`source_url` is the direct BookStack page link. Save `session_id` and pass it in the next request to continue the same conversation.

### Streaming query

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I set up authentication?", "session_id": "optional-uuid"}' \
  --no-buffer
```

The response uses Server-Sent Events (SSE). Each event is a JSON object with a `node` field indicating the pipeline stage:

```
data: {"node": "input"}
data: {"node": "query_rewrite"}
data: {"node": "retriever"}
data: {"node": "reranker"}
data: {"node": "context_compressor"}
data: {"node": "llm_reasoning", "answer": "Authentication is configured by...", "sources": [...]}
data: {"node": "response_validator"}
data: {"node": "response", "session_id": "933d92b1-xxxx", "latency_ms": 2340.5}
```

Pipeline node labels for UI display:

| Node | Display Label |
|---|---|
| `input` | Processing... |
| `query_rewrite` | Optimizing query... |
| `retriever` | Searching documents... |
| `reranker` | Ranking results... |
| `context_compressor` | Preparing context... |
| `llm_reasoning` | Generating answer... |
| `response_validator` | Checking answer... |
| `response` | Finalizing... |

The `session_id` in the final event links this query to a chat session for history tracking.

## Chat History

### List your sessions

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/query/history?page=1&page_size=20"
```

Response:
```json
[
  { "id": "uuid", "title": "Who is Rama?", "message_count": 4, "last_message_at": "2026-03-22T12:46:23Z", "created_at": "2026-03-22T12:44:01Z" }
]
```

### Get a full session (messages + source links)

```bash
SESSION_ID="<session_id from history list>"
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/query/history/$SESSION_ID"
```

Each assistant message includes `sources` with `source_url` — the direct BookStack page link.

### Delete a session

```bash
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/query/history/$SESSION_ID"
# Returns 204 No Content
```

## Popular Questions (Admin / Developer)

```bash
ADMIN_TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin1234"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/query/popular?limit=10"
```

Response:
```json
[
  { "query": "Who is Rama?", "count": 12, "last_asked_at": "2026-03-22T12:46:23Z" }
]
```

Aggregated from audit logs — only queries made via `POST /query` (non-streaming) are counted.

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
