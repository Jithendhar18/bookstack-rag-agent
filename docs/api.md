# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

All routes (except `/health*` and auth endpoints) are prefixed with `/api/v1`.

---

## Authentication

All endpoints except `/health`, `/health/detailed`, `POST /api/v1/auth/login`, and `POST /api/v1/auth/register` require a Bearer token:

```
Authorization: Bearer <access_token>
```

### POST /api/v1/auth/register

Register a new user (assigned the `user` role by default).

Request:
```json
{
  "email": "user@example.com",
  "username": "myusername",
  "password": "securepassword",
  "full_name": "User Name",
  "tenant_id": "default"
}
```

Response `201`:
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "myusername",
  "full_name": "User Name",
  "is_active": true,
  "role": "user",
  "tenant_id": "default",
  "created_at": "2026-03-22T07:11:57"
}
```

### POST /api/v1/auth/login

Authenticate and receive JWT tokens. Login uses **`username`** (not email).

Request:
```json
{
  "username": "admin",
  "password": "admin1234"
}
```

Response `200`:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

`expires_in` is in seconds (default: 30 minutes = 1800 s).

### POST /api/v1/auth/refresh

Exchange a refresh token for a new access + refresh token pair.

Request:
```json
{
  "refresh_token": "eyJ..."
}
```

Response `200`: same shape as `/login`.

### GET /api/v1/auth/me

Return the currently authenticated user's profile.

Response `200`: same shape as the `UserResponse` from `/register`.

---

## Query

### POST /api/v1/query

Ask a question against the ingested documentation. Requires authentication.

Request:
```json
{
  "query": "How do I configure authentication?",
  "session_id": null,
  "top_k": 5,
  "filters": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Question (1–2000 chars) |
| `session_id` | UUID | No | Continue an existing chat session |
| `top_k` | int | No | Number of sources to return (1–50, default 5) |
| `filters` | object | No | Additional metadata filters |

Response `200`:
```json
{
  "answer": "Based on the documentation...",
  "sources": [
    {
      "chunk_id": "abc-123",
      "document_title": "Authentication Setup",
      "content": "...",
      "score": 0.85,
      "source_url": null,
      "metadata": {}
    }
  ],
  "session_id": "uuid",
  "trace_id": "trace-uuid",
  "latency_ms": 1234.5
}
```

### POST /api/v1/query/stream

Stream query results via Server-Sent Events. Same request body as `POST /api/v1/query`.

Each SSE event carries:
```json
{"node": "llm_reasoning", "answer": "...", "sources": [], "metadata": {}}
```

Final event: `data: [DONE]`

---

## Ingestion

Requires `admin` or `developer` role.

### POST /api/v1/ingestion/ingest

Trigger a background ingestion job from BookStack.

Request:
```json
{
  "bookstack_type": "pages",
  "bookstack_ids": [1, 2, 3],
  "force_reindex": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `bookstack_type` | string | `"pages"` | One of: `pages`, `books`, `chapters`, `shelves` |
| `bookstack_ids` | int[] | `null` | IDs to ingest. Omit (or `null`) to ingest all |
| `force_reindex` | bool | `false` | Re-embed even if content hash unchanged |

Response `200`:
```json
{
  "task_id": "uuid",
  "status": "queued",
  "documents_queued": -1,
  "message": "Ingestion task started"
}
```

`documents_queued` is `-1` when ingesting all (count unknown at queue time).

### GET /api/v1/ingestion/status/{task_id}

Poll the status of a running or completed ingestion task.

Response `200`:
```json
{
  "task_id": "uuid",
  "status": "SUCCESS",
  "progress": "completed",
  "result": {
    "status": "completed",
    "stats": {}
  }
}
```

Possible `status` values: `PENDING` (unknown task_id), `PROGRESS`, `SUCCESS`, `FAILURE`.

### GET /api/v1/ingestion/documents

List ingested documents with pagination and optional filters.

Query params: `page` (default 1), `page_size` (default 20), `status`, `book_id`.

Results are ordered by `book_id → chapter_id → title`.

### GET /api/v1/ingestion/books

List all distinct books that have at least one ingested page, with page and chunk counts.

### GET /api/v1/ingestion/books/{book_id}

Return a full `Book → Chapter → Page` hierarchy for a given book.

---

## Admin

Requires `admin` role.

### GET /api/v1/admin/metrics

Return system metrics for the current tenant.

Response `200`:
```json
{
  "total_documents": 42,
  "total_chunks": 1234,
  "total_embeddings": 1234,
  "total_users": 3,
  "total_queries": 99,
  "total_books": 5,
  "documents_by_status": {"completed": 40, "failed": 2},
  "documents_by_book": {"1": 10, "2": 32},
  "avg_query_latency_ms": null
}
```

### GET /api/v1/admin/users

List users in the current tenant.

Query params: `page` (default 1), `page_size` (default 20).

### PATCH /api/v1/admin/users/{user_id}

Update a user's profile, active status, or role.

Request (all fields optional):
```json
{
  "full_name": "New Name",
  "is_active": true,
  "role": "developer"
}
```

Valid roles: `admin`, `developer`, `user`.

---

## Health

### GET /health

Basic health check — no authentication required.

```json
{"status": "ok"}
```

### GET /health/detailed

Detailed health check including cache and vector store subsystems.

```json
{
  "status": "ok",
  "checks": {
    "api": "ok",
    "cache": "ok",
    "vector_store": "ok (qdrant, 1234 vectors)"
  }
}
```
