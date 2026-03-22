# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

All routes (except `/health*` and auth endpoints) are prefixed with `/api/v1`.

---

## Role System

| Role | Description |
|---|---|
| `admin` | Full access — all endpoints |
| `developer` | Can trigger ingestion and view popular questions |
| `user` | Can query, view and manage own chat history |

---

## Authentication

All endpoints except `/health`, `/health/detailed`, `POST /api/v1/auth/login`, and `POST /api/v1/auth/register` require a Bearer token:

```
Authorization: Bearer <access_token>
```

Access tokens expire after 30 minutes (1800 s). Use the refresh endpoint before expiry to get a new pair without forcing the user to log in again.

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

| Field | Required | Notes |
|---|---|---|
| `email` | Yes | Valid email format |
| `username` | Yes | Unique, used for login |
| `password` | Yes | Minimum 8 characters |
| `full_name` | No | Display name |
| `tenant_id` | No | Defaults to `"default"` |

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

`expires_in` is in seconds (default: 30 minutes = 1800 s). Store both tokens — use `access_token` for API calls, and call `/auth/refresh` with the `refresh_token` before the access token expires.

### POST /api/v1/auth/refresh

Exchange a refresh token for a new access + refresh token pair. Call this automatically when the access token is about to expire (e.g. within 60 s of expiry).

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

Creates a new chat session automatically if `session_id` is not provided. The returned `session_id` can be passed in subsequent requests to maintain conversation context.

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
      "document_title": "Ramayana.pdf - Part 74",
      "content": "Excerpt from the chunk...",
      "score": 0.85,
      "source_url": "http://your-bookstack/books/my-book/page/page-slug",
      "metadata": {}
    }
  ],
  "session_id": "uuid",
  "trace_id": "trace-uuid",
  "latency_ms": 1234.5
}
```

`source_url` is the direct link to the BookStack page (e.g. `http://localhost:6875/books/my-book/page/my-page`). Use this to build clickable references in the UI. `session_id` must be saved and passed back in subsequent requests to maintain a multi-turn conversation.

### POST /api/v1/query/stream

Stream query results via Server-Sent Events. Same request body as `POST /api/v1/query`.

Each SSE event carries:
```json
{"node": "llm_reasoning", "answer": "...", "sources": [], "metadata": {}}
```

Final event: `data: [DONE]`

Streaming does **not** persist messages to the database. Use `POST /query` (non-streaming) for conversations that need history.

---

## Chat History

All history endpoints are scoped to the authenticated user — users can only access their own sessions.

### GET /api/v1/query/history

List the current user's chat sessions, newest first. Paginated.

Query params: `page` (default 1), `page_size` (default 20, max 100).

Response `200`:
```json
[
  {
    "id": "uuid",
    "title": "Who is Rama?",
    "message_count": 4,
    "last_message_at": "2026-03-22T12:46:23Z",
    "created_at": "2026-03-22T12:44:01Z"
  }
]
```

`title` is auto-set to the first 100 characters of the first query in the session. `message_count` counts both user and assistant turns.

### GET /api/v1/query/history/{session_id}

Return a full chat session including all messages and source links.

Response `200`:
```json
{
  "id": "uuid",
  "title": "Who is Rama?",
  "created_at": "2026-03-22T12:44:01Z",
  "updated_at": "2026-03-22T12:46:23Z",
  "messages": [
    {
      "role": "user",
      "content": "Who is Rama?",
      "sources": [],
      "created_at": "2026-03-22T12:44:01Z"
    },
    {
      "role": "assistant",
      "content": "Rama is...",
      "sources": [
        {
          "chunk_id": "abc-123",
          "document_title": "Ramayana.pdf - Part 74",
          "content": "...",
          "score": 0.92,
          "source_url": "http://your-bookstack/books/ramayan/page/ramayanapdf-part-74",
          "metadata": {}
        }
      ],
      "created_at": "2026-03-22T12:44:03Z"
    }
  ]
}
```

Returns `404` if the session doesn't belong to the authenticated user.

### DELETE /api/v1/query/history/{session_id}

Delete a chat session and all its messages. Returns `204 No Content`.

Returns `404` if the session doesn't belong to the authenticated user.

### GET /api/v1/query/popular

Return the most frequently asked questions across the tenant. Requires `admin` or `developer` role.

Query params: `limit` (default 10, max 50).

Response `200`:
```json
[
  {
    "query": "Who is Rama?",
    "count": 12,
    "last_asked_at": "2026-03-22T12:46:23Z"
  }
]
```

Data is aggregated from the audit log — only queries made via `POST /query` are counted. Results are ordered by frequency descending.

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

### DELETE /api/v1/admin/users/{user_id}

Deactivate (soft-delete) a user. Returns `204 No Content`.

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

---

## Quick Endpoint Reference

| Method | Endpoint | Role Required | Purpose |
|---|---|---|---|
| POST | `/api/v1/auth/register` | None | Create account |
| POST | `/api/v1/auth/login` | None | Get tokens |
| POST | `/api/v1/auth/refresh` | None | Refresh access token |
| GET | `/api/v1/auth/me` | Any | Own profile |
| POST | `/api/v1/query` | Any | RAG query (saves to history) |
| POST | `/api/v1/query/stream` | Any | Streaming RAG query (SSE, no history) |
| GET | `/api/v1/query/history` | Any | List own chat sessions |
| GET | `/api/v1/query/history/{id}` | Owner | Full session with messages + source URLs |
| DELETE | `/api/v1/query/history/{id}` | Owner | Delete session |
| GET | `/api/v1/query/popular` | Admin/Developer | Frequent questions across tenant |
| POST | `/api/v1/ingestion/ingest` | Admin/Developer | Trigger BookStack ingestion |
| GET | `/api/v1/ingestion/status/{task_id}` | Admin/Developer | Poll ingestion task |
| GET | `/api/v1/ingestion/documents` | Admin/Developer | List ingested documents |
| GET | `/api/v1/ingestion/books` | Admin/Developer | List books with counts |
| GET | `/api/v1/ingestion/books/{book_id}` | Admin/Developer | Book hierarchy |
| GET | `/api/v1/admin/metrics` | Admin | System-wide stats |
| GET | `/api/v1/admin/users` | Admin | List users |
| PATCH | `/api/v1/admin/users/{id}` | Admin | Update user |
| GET | `/health` | None | Basic health check |
| GET | `/health/detailed` | None | Full health check |
