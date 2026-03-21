# Configuration Guide

All configuration is done via environment variables. Set them in `backend/.env`.

---

## Quick Start

```bash
# Option 1: Use an AI profile (simplest)
AI_PROFILE=balanced

# Option 2: Configure everything individually
LLM_PROVIDER=openrouter
LLM_MODEL=mistralai/mistral-small
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_ENABLED=true
```

---

## AI Profiles

Set `AI_PROFILE` to apply a tested configuration preset. Individual env vars override profile defaults.

### `cheap` — 100% Free, Local

| Setting | Value |
|---|---|
| LLM_PROVIDER | ollama |
| LLM_MODEL | llama3.2 |
| EMBEDDING_PROVIDER | local |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 |
| RERANKER_ENABLED | false |
| RETRIEVAL_MODE | dense |
| CONTEXT_COMPRESSION_ENABLED | false |
| QUERY_REWRITER_ENABLED | false |

**Best for**: Development, testing, air-gapped environments.

### `balanced` — Cost-Effective Production

| Setting | Value |
|---|---|
| LLM_PROVIDER | openrouter |
| LLM_MODEL | mistralai/mistral-small |
| EMBEDDING_PROVIDER | local |
| EMBEDDING_MODEL | BAAI/bge-base-en-v1.5 |
| RERANKER_ENABLED | true |
| RERANKER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RETRIEVAL_MODE | hybrid |
| CONTEXT_COMPRESSION_ENABLED | true |
| QUERY_REWRITER_ENABLED | true |

**Best for**: Production with moderate traffic. ~$0.10/1K queries.

### `best` — Maximum Quality

| Setting | Value |
|---|---|
| LLM_PROVIDER | openrouter |
| LLM_MODEL | openai/gpt-4o |
| EMBEDDING_PROVIDER | local |
| EMBEDDING_MODEL | BAAI/bge-large-en-v1.5 |
| RERANKER_ENABLED | true |
| RERANKER_MODEL | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RETRIEVAL_MODE | hybrid |
| CONTEXT_COMPRESSION_ENABLED | true |
| QUERY_REWRITER_ENABLED | true |

**Best for**: Knowledge bases where answer quality is critical. ~$2/1K queries.

---

## All Environment Variables

### Core Infrastructure

| Variable | Default | Description |
|---|---|---|
| `AI_PROFILE` | *(none)* | Preset profile: `cheap`, `balanced`, or `best` |
| `DATABASE_URL` | *(required)* | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis for cache and Celery broker |

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | One of: `openai`, `openrouter`, `groq`, `ollama` |
| `LLM_MODEL` | `mistralai/mistral-small` | Model name at the provider |
| `LLM_ENABLED` | `true` | Master switch for LLM reasoning |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |

### LLM Fallback (Optional)

| Variable | Default | Description |
|---|---|---|
| `LLM_FALLBACK_PROVIDER` | *(none)* | Fallback provider if primary fails |
| `LLM_FALLBACK_MODEL` | *(none)* | Model name for fallback provider |

### API Keys

| Variable | Required When | Description |
|---|---|---|
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` or embeddings | OpenAI API key |
| `OPENROUTER_API_KEY` | `LLM_PROVIDER=openrouter` | [OpenRouter](https://openrouter.ai) API key |
| `GROQ_API_KEY` | `LLM_PROVIDER=groq` | [Groq](https://groq.com) API key |
| `LLM_API_KEY` | Manual override | Overrides provider-specific key |
| `LLM_BASE_URL` | Manual override | Overrides provider-specific base URL |

### Embedding Provider

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | One of: `local`, `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Model name |
| `EMBEDDING_DIMENSION` | `768` | Embedding vector dimension |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | One of: `dense`, `keyword`, `hybrid` |
| `RETRIEVAL_TOP_K` | `10` | Number of results to retrieve |
| `VECTOR_STORE` | `qdrant` | Vector store backend |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant server address |
| `QDRANT_COLLECTION_NAME` | `bookstack_documents` | Collection name |

### Reranker

| Variable | Default | Description |
|---|---|---|
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model |
| `RERANKER_TOP_N` | `5` | Results to keep after reranking |

### Pipeline Toggles

| Variable | Default | Description |
|---|---|---|
| `QUERY_REWRITER_ENABLED` | `true` | LLM rewrites user query for better retrieval |
| `CONTEXT_COMPRESSION_ENABLED` | `true` | Dedup + MMR + token truncation |
| `GUARDRAILS_ENABLED` | `true` | Prompt injection detection + output grounding |
| `CACHE_ENABLED` | `true` | Redis caching of query results |

### BookStack

| Variable | Default | Description |
|---|---|---|
| `BOOKSTACK_BASE_URL` | *(required)* | BookStack instance URL |
| `BOOKSTACK_TOKEN_ID` | *(required)* | API token ID |
| `BOOKSTACK_TOKEN_SECRET` | *(required)* | API token secret |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | *(none)* | LangSmith API key |
| `LANGCHAIN_PROJECT` | `bookstack-rag` | LangSmith project name |

### Auth & Security

| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET_KEY` | *(required)* | Secret for JWT signing |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRY_MINUTES` | `60` | Token expiry time |
| `RATE_LIMIT` | `60/minute` | API rate limit |

---

## Override Behavior

Profile defaults apply **only** when the corresponding env var is **not set**:

```bash
# Profile sets LLM_PROVIDER=ollama and LLM_MODEL=llama3.2
AI_PROFILE=cheap

# But you can override specific values:
LLM_MODEL=phi3    # overrides the profile's llama3.2
```

Resolution order:
1. Explicit env var (highest priority)
2. AI profile default
3. Config class default (lowest priority)

---

## Common Configurations

### Free Local (No API keys)

```env
AI_PROFILE=cheap
# Requires: Ollama running locally with llama3.2 pulled
```

### OpenRouter + Local Embeddings

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=mistralai/mistral-small
EMBEDDING_PROVIDER=local
RERANKER_ENABLED=true
```

### Maximum Speed (Groq)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.1-70b-versatile
EMBEDDING_PROVIDER=local
RERANKER_ENABLED=false
CONTEXT_COMPRESSION_ENABLED=false
QUERY_REWRITER_ENABLED=false
```

### High Quality with Fallback

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openai/gpt-4o
LLM_FALLBACK_PROVIDER=groq
LLM_FALLBACK_MODEL=llama-3.1-70b-versatile
GROQ_API_KEY=gsk_...
RERANKER_ENABLED=true
RETRIEVAL_MODE=hybrid
```
