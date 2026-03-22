# API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## Authentication

All endpoints (except health and login/register) require a Bearer token:

```
Authorization: Bearer <jwt_token>
```

### POST /api/auth/register

Register a new user.

```json
{
  "email": "user@example.com",
  "username": "user",
  "password": "securepassword",
  "full_name": "User Name"
}
```

### POST /api/auth/login

Login and receive JWT token.

```json
{
  "email": "admin@bookstack-rag.local",
  "password": "admin1234"
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

## Queries

### POST /api/query

Ask a question against ingested documentation.

```json
{
  "query": "How do I configure authentication?",
  "tenant_id": "default"
}
```

Response:
```json
{
  "answer": "Based on the documentation...",
  "sources": [
    {
      "chunk_id": "...",
      "document_title": "Authentication Setup",
      "content": "...",
      "score": 0.85,
      "source_url": "https://bookstack.example.com/books/..."
    }
  ],
  "metadata": {
    "latency_ms": 1234,
    "documents_used": 3
  }
}
```

### POST /api/query/stream

Stream query results (Server-Sent Events).

Same request body as `/api/query`. Returns node-by-node progress events.

## Ingestion

### POST /api/ingestion/ingest

Trigger ingestion from BookStack.

```json
{
  "tenant_id": "default",
  "force_reindex": false,
  "page_ids": [1, 2, 3]
}
```

Omit `page_ids` to ingest all pages.

### GET /api/ingestion/status

Get ingestion status and document counts.

## Admin

### GET /api/admin/stats

Get system statistics (admin only).

### GET /api/admin/users

List all users (admin only).

### PUT /api/admin/users/{user_id}

Update user role or status (admin only).

## Health

### GET /api/health

Health check endpoint (no auth required).

```json
{
  "status": "healthy",
  "database": "connected",
  "qdrant": "connected"
}
```
