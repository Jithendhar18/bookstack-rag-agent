# Troubleshooting Guide

## LLM Not Responding

### Symptoms
- Queries hang or timeout
- Error: "Connection refused" or "API key invalid"

### Solutions

**OpenRouter / OpenAI / Groq:**
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Test connectivity
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | head -20
```

**Ollama (local):**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the required model
ollama pull llama3.2

# Verify OLLAMA_BASE_URL matches your setup
# Default: http://localhost:11434
```

**Fallback not working:**
- Ensure `LLM_FALLBACK_PROVIDER` is set (not just `LLM_FALLBACK_MODEL`)
- Ensure the fallback provider's API key is also configured
- Check logs for "Falling back to..." messages

---

## Embedding Errors

### "Dimension mismatch" / Qdrant Rejects Vectors

**Cause**: Embedding model changed but Qdrant collection still has old dimension.

**Fix**:
```bash
# Option 1: Delete and recreate collection
curl -X DELETE http://localhost:6333/collections/bookstack_documents

# Then re-ingest
curl -X POST http://localhost:8000/api/v1/ingestion/ingest

# Option 2: Use matching dimension
# bge-small: 384, bge-base: 768, bge-large: 1024
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
```

### Slow First Embedding

**Cause**: SentenceTransformer downloads model on first use (~400MB for bge-base).

**Fix**: Pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

---

## Qdrant Connection Issues

### "Connection refused" to Qdrant

```bash
# Check Qdrant is running
docker compose ps qdrant

# Check URL matches
echo $QDRANT_URL  # should be http://qdrant:6333 (docker) or http://localhost:6333 (local)

# Test connectivity
curl http://localhost:6333/healthz
```

### Collection Not Found

```bash
# List collections
curl http://localhost:6333/collections

# Create via ingestion
curl -X POST http://localhost:8000/api/v1/ingestion/ingest \
  -H "Authorization: Bearer $TOKEN"
```

---

## High Latency

### Diagnosis

Check the response metadata for per-stage timing:
```json
{
  "metadata": {
    "latency_ms": 3400,
    "modules": {
      "query_rewriter": true,
      "reranker": true,
      "context_compressor": true
    }
  }
}
```

### Optimization Steps

1. **Disable optional stages**:
   ```env
   QUERY_REWRITER_ENABLED=false      # saves ~500-1000ms
   CONTEXT_COMPRESSION_ENABLED=false  # saves ~100ms
   RERANKER_ENABLED=false             # saves ~200-500ms
   ```

2. **Reduce retrieval count**:
   ```env
   RETRIEVAL_TOP_K=5    # default: 10
   RERANKER_TOP_N=3     # default: 5
   ```

3. **Use faster LLM**:
   ```env
   LLM_PROVIDER=groq   # Groq is typically fastest
   ```

4. **Enable caching**:
   ```env
   CACHE_ENABLED=true   # default: true
   ```

5. **Use dense-only retrieval** (skip keyword search):
   ```env
   RETRIEVAL_MODE=dense
   ```

---

## Wrong or Low-Quality Answers

### Answers Don't Match Documents

1. **Enable reranker** — significantly improves relevance:
   ```env
   RERANKER_ENABLED=true
   ```

2. **Use hybrid retrieval** — catches keyword matches that dense might miss:
   ```env
   RETRIEVAL_MODE=hybrid
   ```

3. **Increase retrieval count**:
   ```env
   RETRIEVAL_TOP_K=20
   RERANKER_TOP_N=8
   ```

4. **Use a better LLM**:
   ```env
   AI_PROFILE=best
   ```

### Answers Include Hallucinations

1. **Enable guardrails**:
   ```env
   GUARDRAILS_ENABLED=true
   ```

2. **Lower LLM temperature**:
   ```env
   LLM_TEMPERATURE=0.1
   ```

3. **Check document coverage** — the answer might be correct but not in your BookStack docs

---

## Redis Connection Issues

```bash
# Check Redis is running
docker compose ps redis

# Test connectivity
redis-cli -u $REDIS_URL ping

# If Redis is down, the app still works but without caching
# Set CACHE_ENABLED=false to suppress Redis error logs
```

---

## Database Issues

### Migration Errors

```bash
cd backend

# Check current migration state
alembic current

# Reset to a known state
alembic downgrade base
alembic upgrade head
```

### "Relation does not exist"

Run migrations:
```bash
cd backend
alembic upgrade head
python scripts/seed_db.py
```

---

## Docker Issues

### Container Keeps Restarting

```bash
# Check logs
docker compose logs backend

# Common causes:
# - Missing .env file
# - Invalid DATABASE_URL
# - Port conflicts (8000, 6333, 5432, 6379)
```

### Out of Memory (Embedding/Reranker Models)

SentenceTransformer and CrossEncoder models can use significant RAM:

| Model | ~RAM |
|---|---|
| bge-small-en-v1.5 | ~100MB |
| bge-base-en-v1.5 | ~400MB |
| bge-large-en-v1.5 | ~1.3GB |
| ms-marco-MiniLM-L-6-v2 | ~100MB |

**Fix**: Use smaller models or disable reranker:
```env
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_ENABLED=false
```

---

## Ingestion Issues

### Ingestion Hangs / No Documents Imported

```bash
# Check Celery worker is running
docker compose logs celery_worker

# Check BookStack credentials
curl -s "$BOOKSTACK_BASE_URL/api/pages" \
  -H "Authorization: Token $BOOKSTACK_TOKEN_ID:$BOOKSTACK_TOKEN_SECRET" | head -5
```

### Duplicate Documents After Re-ingestion

The pipeline deduplicates by content hash. If you see duplicates:
1. Check `content_hash` field in PostgreSQL
2. Delete the Qdrant collection and re-ingest for a clean slate

---

## Logging

### Enable Debug Logs

```env
LOG_LEVEL=DEBUG
```

### LangSmith Tracing

For detailed pipeline traces:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=bookstack-rag-debug
```

Visit [smith.langchain.com](https://smith.langchain.com) to view traces.

---

## Health Check

Use the built-in health endpoint to verify all components:

```bash
curl http://localhost:8000/api/v1/health | python -m json.tool
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "database": "connected",
    "redis": "connected",
    "qdrant": "connected"
  }
}
```
