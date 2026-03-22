# Architecture

BookStack RAG Agent is a Retrieval-Augmented Generation system that ingests documentation from BookStack and provides an AI-powered Q&A interface.

---

## 1. System Architecture (High Level)

```mermaid
graph TD
    Client["🖥️ Client<br/>(Browser / API)"]
    API["⚡ FastAPI<br/>API Layer"]
    LG["🔀 LangGraph<br/>RAG Pipeline"]
    PG[("🐘 PostgreSQL<br/>Users · Documents · Chunks<br/>Audit · Sessions")]
    QD[("🔷 Qdrant<br/>Vector Store<br/>bookstack_documents")]
    EMB["🧠 SentenceTransformers<br/>BAAI/bge-base-en-v1.5"]
    LLM["💬 LLM Provider<br/>OpenAI · Groq · OpenRouter · Ollama"]
    BS["📚 BookStack<br/>Documentation Source"]
    Cache["📦 In-Memory Cache<br/>TTL 300s"]

    Client <-->|"REST / SSE"| API
    API -->|"Query"| LG
    API -->|"Ingest"| BS
    LG -->|"Retrieve / Store"| QD
    LG -->|"Read / Write"| PG
    LG -->|"Embed query"| EMB
    LG -->|"Generate answer"| LLM
    API -->|"Cache hit?"| Cache
    BS -->|"Pages + HTML"| API

    style Client fill:#e1f5fe,stroke:#0277bd
    style API fill:#fff3e0,stroke:#ef6c00
    style LG fill:#f3e5f5,stroke:#7b1fa2
    style PG fill:#e8f5e9,stroke:#2e7d32
    style QD fill:#e3f2fd,stroke:#1565c0
    style EMB fill:#fce4ec,stroke:#c62828
    style LLM fill:#fff9c4,stroke:#f9a825
    style BS fill:#efebe9,stroke:#4e342e
    style Cache fill:#f1f8e9,stroke:#558b2f
```

---

## 2. Ingestion Pipeline (Detailed)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI
    participant BS as BookStack API
    participant P as ContentParser
    participant CH as SemanticChunker
    participant EMB as SentenceTransformers
    participant QD as Qdrant
    participant PG as PostgreSQL

    C->>API: POST /api/v1/ingestion/ingest
    activate API
    API->>API: Create task_id (UUID)
    API-->>C: 202 Accepted {task_id}
    deactivate API

    Note over API: BackgroundTask starts

    API->>BS: get_all_pages() — paginated (100/batch)
    BS-->>API: List of page summaries

    API->>BS: build_hierarchy_caches()
    BS-->>API: book_names, slugs, URLs, chapters

    loop For each page
        API->>BS: get_page(page_id)
        BS-->>API: Page HTML + metadata

        API->>P: html_to_text(html)
        P-->>API: Plain text

        API->>P: normalize_text(text)
        P-->>API: Cleaned text

        API->>P: compute_hash(text) — SHA-256
        P-->>API: content_hash

        alt Hash unchanged & not force_reindex
            Note over API: Skip — already indexed
        else New or changed content
            API->>PG: Upsert Document (status="processing")

            API->>CH: chunk_text(text)
            Note over CH: Split on headers → paragraphs → sentences<br/>chunk_size=512, overlap=50
            CH-->>API: Text chunks[]

            API->>PG: Create Chunk records

            API->>EMB: embed_batch(chunks)
            Note over EMB: SentenceTransformer encode<br/>batch_size=32, normalize=True
            EMB-->>API: Embedding vectors[]

            API->>QD: add_embeddings(ids, vectors, metadata, texts)
            Note over QD: Upsert PointStruct<br/>batches of 100, 3 retries
            QD-->>API: OK

            API->>PG: Create EmbeddingMetadata records
            API->>PG: Update Document (status="completed")
        end
    end

    Note over API: Task complete — stats stored in memory
    C->>API: GET /api/v1/ingestion/status/{task_id}
    API-->>C: {status, processed, skipped, failed, chunks_created}
```

---

## 3. RAG Query Pipeline (Detailed)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant API as FastAPI
    participant Cache as InMemoryCache
    participant LG as LangGraph Agent
    participant GR as Guardrails
    participant LLM as LLM Provider
    participant RET as Retriever
    participant EMB as Embedding
    participant QD as Qdrant
    participant RR as CrossEncoder Reranker
    participant PG as PostgreSQL

    U->>API: POST /api/v1/query {query, session_id?}
    activate API
    API->>PG: Validate JWT → get User
    API->>Cache: get_query_result(query, tenant_id)

    alt Cache hit
        Cache-->>API: Cached response
        API-->>U: QueryResponse (cached)
    else Cache miss
        API->>LG: run_agent_query(query, tenant_id, session_id)
        activate LG

        Note over LG: Node 1: input
        LG->>GR: check_prompt_injection(query)
        GR-->>LG: {safe: true/false}

        alt Prompt blocked
            LG-->>API: Error: blocked
            API-->>U: 200 {error: "blocked"}
        else Safe query
            Note over LG: Node 2: query_rewrite
            opt QUERY_REWRITER_ENABLED=true
                LG->>LLM: Rewrite query for retrieval
                LLM-->>LG: Optimized query
            end

            Note over LG: Node 3: retriever
            LG->>EMB: embed(rewritten_query)
            EMB-->>LG: Query vector

            LG->>RET: retrieve(query, top_k, tenant_id)
            RET->>QD: Dense / Keyword / Hybrid search
            QD-->>RET: Scored results
            RET-->>LG: Retrieved documents

            alt No documents found
                LG-->>API: No relevant documents
            else Documents found
                Note over LG: Node 4: reranker
                opt RERANKER_ENABLED=true
                    LG->>RR: rerank(query, docs, top_k_rerank)
                    RR-->>LG: Reranked documents
                end

                Note over LG: Node 5: context_compressor
                opt CONTEXT_COMPRESSION_ENABLED=true
                    Note over LG: Deduplicate → MMR select → Trim to token budget
                end

                Note over LG: Node 6: llm_reasoning
                LG->>LLM: System prompt + numbered sources + query
                LLM-->>LG: Generated answer + source refs

                Note over LG: Node 7: response_validator
                opt GUARDRAILS_ENABLED=true
                    LG->>GR: enforce_source_requirement(sources)
                    LG->>GR: validate_output_grounding(answer, sources)
                    GR-->>LG: {grounded, confidence}
                end

                Note over LG: Node 8: response
                LG-->>API: {answer, sources, metadata}
            end
        end
        deactivate LG

        API->>Cache: set_query_result(query, result)
        API->>PG: Create ChatSession (if new)
        API->>PG: Store ChatMessages (user + assistant)
        API->>PG: Create AuditLog
        API-->>U: QueryResponse
    end
    deactivate API
```

---

## 4. LangGraph Internal Flow (Node Level)

```mermaid
graph TD
    START(("▶ START"))
    INPUT["input<br/>━━━━━━━━━━<br/>Validate query<br/>Check prompt injection"]
    QR["query_rewrite<br/>━━━━━━━━━━<br/>LLM rewrites query<br/>for better retrieval"]
    RET["retriever<br/>━━━━━━━━━━<br/>Dense / Keyword / Hybrid<br/>search in Qdrant"]
    RR["reranker<br/>━━━━━━━━━━<br/>CrossEncoder scoring<br/>or NoOp pass-through"]
    CC["context_compressor<br/>━━━━━━━━━━<br/>Dedup → MMR → Token trim<br/>Max 10 docs"]
    LLM["llm_reasoning<br/>━━━━━━━━━━<br/>Build prompt with sources<br/>Generate answer via LLM"]
    RV["response_validator<br/>━━━━━━━━━━<br/>Source requirement check<br/>Grounding validation"]
    RESP["response<br/>━━━━━━━━━━<br/>Compute latency<br/>Build module summary"]
    END(("⏹ END"))

    START --> INPUT

    INPUT -->|"error"| RESP
    INPUT -->|"ok"| QR

    QR --> RET

    RET -->|"no docs / error"| RESP
    RET -->|"docs found"| RR

    RR --> CC
    CC --> LLM
    LLM --> RV
    RV --> RESP

    RESP --> END

    style START fill:#c8e6c9,stroke:#2e7d32
    style END fill:#ffcdd2,stroke:#c62828
    style INPUT fill:#e1f5fe,stroke:#0277bd
    style QR fill:#fff3e0,stroke:#ef6c00
    style RET fill:#e3f2fd,stroke:#1565c0
    style RR fill:#f3e5f5,stroke:#7b1fa2
    style CC fill:#fce4ec,stroke:#c62828
    style LLM fill:#fff9c4,stroke:#f9a825
    style RV fill:#e8f5e9,stroke:#2e7d32
    style RESP fill:#efebe9,stroke:#4e342e
```

### Conditional Edges

| From | Condition | Target |
|---|---|---|
| `input` | `state["error"]` is set | → `response` (short-circuit) |
| `input` | No error | → `query_rewrite` |
| `retriever` | No documents or error | → `response` (short-circuit) |
| `retriever` | Documents found | → `reranker` |

### Toggleable Nodes

| Node | Config Toggle | When Disabled |
|---|---|---|
| `input` (guardrails) | `GUARDRAILS_ENABLED` | Skips injection check |
| `query_rewrite` | `QUERY_REWRITER_ENABLED` | Passes original query through |
| `reranker` | `RERANKER_ENABLED` | Uses `NoOpReranker` (pass-through) |
| `context_compressor` | `CONTEXT_COMPRESSION_ENABLED` | Passes reranked docs through |
| `response_validator` | `GUARDRAILS_ENABLED` | Skips grounding validation |

---

## 5. Module-Level Architecture

```mermaid
graph TD
    subgraph API["API Layer"]
        health["health_routes<br/>/health"]
        auth["auth_routes<br/>/api/v1/auth"]
        query["query_routes<br/>/api/v1/query"]
        ingest["ingestion_routes<br/>/api/v1/ingestion"]
        admin["admin_routes<br/>/api/v1/admin"]
    end

    subgraph Core["Core Services"]
        MW["middleware<br/>RequestContext · CORS"]
        GR["guardrails<br/>Injection · Grounding"]
        CA["cache<br/>InMemoryCache (TTL)"]
        LOG["logging_config"]
    end

    subgraph Agents["LangGraph Agent"]
        graph_mod["graph.py<br/>Build & run graph"]
        nodes["nodes.py<br/>8 pipeline nodes"]
        state["state.py<br/>AgentState TypedDict"]
    end

    subgraph Providers["Provider Layer"]
        factory["factory.py<br/>Singletons"]
        LLM_P["llm/<br/>OpenAICompatible · Ollama"]
        EMB_P["embeddings/<br/>LocalEmbedding"]
        RR_P["rerankers/<br/>CrossEncoder"]
        RET_P["retrievers/<br/>Dense · Keyword · Hybrid"]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        pipe["pipeline.py<br/>IngestionPipeline"]
        bsclient["bookstack_client.py<br/>BookStack API"]
        parser["content_parser.py<br/>HTML → Text"]
        chunker["chunker.py<br/>SemanticChunker"]
    end

    subgraph Storage["Data Layer"]
        db["db/<br/>models · session · seed"]
        vs["retrieval/<br/>VectorStoreManager"]
        jwt["auth/<br/>JWT · Password"]
    end

    subgraph External["External Services"]
        PG[("PostgreSQL")]
        QD[("Qdrant")]
        LLM_EXT["LLM API"]
        BS_EXT["BookStack API"]
    end

    query --> graph_mod
    graph_mod --> nodes
    nodes --> state
    nodes --> factory
    factory --> LLM_P & EMB_P & RR_P & RET_P
    nodes --> GR
    RET_P --> vs
    ingest --> pipe
    pipe --> bsclient & parser & chunker
    pipe --> factory
    pipe --> vs
    pipe --> db
    bsclient --> BS_EXT
    auth --> jwt
    admin --> db
    query --> CA
    query --> db
    vs --> QD
    db --> PG
    LLM_P --> LLM_EXT
    MW -.->|wraps| API

    style API fill:#fff3e0,stroke:#ef6c00
    style Core fill:#e1f5fe,stroke:#0277bd
    style Agents fill:#f3e5f5,stroke:#7b1fa2
    style Providers fill:#fff9c4,stroke:#f9a825
    style Ingestion fill:#e8f5e9,stroke:#2e7d32
    style Storage fill:#efebe9,stroke:#4e342e
    style External fill:#fce4ec,stroke:#c62828
```

---

## 6. Database Schema

```mermaid
erDiagram
    roles {
        uuid id PK
        string name UK
        string description
        timestamp created_at
    }

    permissions {
        uuid id PK
        uuid role_id FK
        string resource
        string action
        timestamp created_at
    }

    users {
        uuid id PK
        string email UK
        string username UK
        string hashed_password
        string full_name
        boolean is_active
        string tenant_id
        uuid role_id FK
        timestamp created_at
        timestamp updated_at
    }

    documents {
        uuid id PK
        int bookstack_id
        string bookstack_type
        string title
        string slug
        int book_id
        string book_name
        int chapter_id
        string chapter_name
        string content_hash
        text html_content
        text plain_content
        string status
        string tenant_id
        jsonb metadata
        timestamp created_at
        timestamp updated_at
        timestamp ingested_at
    }

    chunks {
        uuid id PK
        uuid document_id FK
        int chunk_index
        text content
        string content_hash
        int token_count
        int char_count
        jsonb metadata
        timestamp created_at
    }

    embeddings_metadata {
        uuid id PK
        uuid chunk_id FK
        string vector_store_id
        string model_name
        int dimension
        timestamp created_at
    }

    chat_sessions {
        uuid id PK
        uuid user_id FK
        string title
        string tenant_id
        timestamp created_at
        timestamp updated_at
    }

    chat_messages {
        uuid id PK
        uuid session_id FK
        string role
        text content
        jsonb metadata
        jsonb sources
        int token_count
        timestamp created_at
    }

    audit_logs {
        uuid id PK
        uuid user_id FK
        string action
        string resource
        string resource_id
        jsonb details
        string ip_address
        string tenant_id
        timestamp created_at
    }

    roles ||--o{ permissions : "has"
    roles ||--o{ users : "assigned to"
    users ||--o{ chat_sessions : "owns"
    users ||--o{ audit_logs : "performed"
    chat_sessions ||--o{ chat_messages : "contains"
    documents ||--o{ chunks : "split into"
    chunks ||--o| embeddings_metadata : "has"
```

---

## 7. Database + Vector Store Interaction

```mermaid
graph LR
    subgraph PostgreSQL["PostgreSQL (Relational)"]
        DOC["documents<br/>━━━━━━━━━━<br/>bookstack_id, title<br/>content_hash, status<br/>book_id, chapter_id"]
        CHK["chunks<br/>━━━━━━━━━━<br/>document_id (FK)<br/>chunk_index, content<br/>token_count"]
        EMD["embeddings_metadata<br/>━━━━━━━━━━<br/>chunk_id (FK)<br/>vector_store_id<br/>model_name, dimension"]
    end

    subgraph Qdrant["Qdrant (Vector Store)"]
        COL["bookstack_documents<br/>━━━━━━━━━━<br/>id: vector_store_id<br/>vector: float[768]<br/>payload: {text, tenant_id,<br/>page_id, title, book_name,<br/>source_url, ...}"]
    end

    DOC -->|"1:N"| CHK
    CHK -->|"1:1"| EMD
    EMD ---|"vector_store_id links to"| COL

    subgraph Ingestion Flow
        direction TB
        I1["1. Upsert Document"] --> I2["2. Create Chunks"]
        I2 --> I3["3. Embed + Store in Qdrant"]
        I3 --> I4["4. Create EmbeddingMetadata"]
    end

    subgraph Query Flow
        direction TB
        Q1["1. Embed query"] --> Q2["2. Search Qdrant"]
        Q2 --> Q3["3. Return scored docs"]
        Q3 --> Q4["4. Source URLs from payload"]
    end

    style PostgreSQL fill:#e8f5e9,stroke:#2e7d32
    style Qdrant fill:#e3f2fd,stroke:#1565c0
```

---

## 8. Combined End-to-End Flow

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Admin Client
    participant User as End User
    participant API as FastAPI
    participant BS as BookStack
    participant Pipeline as IngestionPipeline
    participant EMB as Embedding Model
    participant QD as Qdrant
    participant PG as PostgreSQL
    participant LG as LangGraph
    participant LLM as LLM

    rect rgb(232, 245, 233)
        Note over Admin,PG: Phase 1 — Ingestion
        Admin->>API: POST /api/v1/ingestion/ingest
        API-->>Admin: 202 {task_id}
        API->>BS: Fetch all pages
        BS-->>API: Pages HTML
        API->>Pipeline: Parse → Chunk → Embed
        Pipeline->>EMB: embed_batch(chunks)
        EMB-->>Pipeline: Vectors
        Pipeline->>QD: Store vectors + metadata
        Pipeline->>PG: Store documents + chunks
    end

    rect rgb(227, 242, 253)
        Note over User,LLM: Phase 2 — Query
        User->>API: POST /api/v1/query {query}
        API->>LG: run_agent_query()
        LG->>LG: Input validation + guardrails
        LG->>LG: Query rewrite (optional)
        LG->>EMB: Embed query
        EMB-->>LG: Query vector
        LG->>QD: Vector similarity search
        QD-->>LG: Relevant chunks + scores
        LG->>LG: Rerank → Compress → Trim
        LG->>LLM: Prompt with context + query
        LLM-->>LG: Generated answer
        LG->>LG: Validate grounding
        LG-->>API: {answer, sources, metadata}
        API->>PG: Log session + messages + audit
        API-->>User: QueryResponse
    end
```

---

## 9. Retrieval Strategy Detail

```mermaid
graph TD
    Q["Query"]

    subgraph Factory["get_retriever()"]
        MODE{{"RETRIEVAL_MODE"}}
    end

    subgraph Dense["DenseRetriever"]
        D1["Embed query → vector"]
        D2["Qdrant vector search"]
        D3["Filter by SIMILARITY_THRESHOLD"]
    end

    subgraph Keyword["KeywordRetriever"]
        K1["Qdrant full-text search"]
        K2["MatchText filter on 'text' index"]
    end

    subgraph Hybrid["HybridRetriever"]
        H1["Run Dense + Keyword in parallel"]
        H2["Reciprocal Rank Fusion"]
        H3["Weighted merge<br/>DENSE_WEIGHT + BM25_WEIGHT"]
        H4["Normalize scores 0–1"]
    end

    Q --> Factory
    MODE -->|"dense"| D1 --> D2 --> D3
    MODE -->|"keyword"| K1 --> K2
    MODE -->|"hybrid"| H1 --> H2 --> H3 --> H4

    style Dense fill:#e3f2fd,stroke:#1565c0
    style Keyword fill:#fff3e0,stroke:#ef6c00
    style Hybrid fill:#f3e5f5,stroke:#7b1fa2
```

---

## 10. Authentication & Authorization Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant JWT as JWTHandler
    participant PG as PostgreSQL

    Note over C,PG: Registration
    C->>API: POST /api/v1/auth/register {email, username, password}
    API->>PG: Check existing user
    API->>API: hash_password(bcrypt)
    API->>PG: Create User (role="user")
    API->>JWT: create_access_token + create_refresh_token
    API-->>C: {access_token, refresh_token, user}

    Note over C,PG: Authenticated Request
    C->>API: POST /api/v1/query<br/>Authorization: Bearer {access_token}
    API->>JWT: decode_token(token)
    JWT-->>API: {sub, role, tenant_id, type:"access"}
    API->>PG: Lookup user by ID, check is_active
    API->>API: require_roles(["user","admin","developer"])
    API->>API: Process request...
    API-->>C: Response

    Note over C,PG: Token Refresh
    C->>API: POST /api/v1/auth/refresh {refresh_token}
    API->>JWT: decode_token(refresh_token)
    JWT-->>API: {sub, type:"refresh"}
    API->>PG: Lookup user
    API->>JWT: New access_token + refresh_token
    API-->>C: {access_token, refresh_token}
```

---

## Directory Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── config.py               # Environment-based settings (pydantic-settings)
├── alembic/                # Database migrations
├── app/
│   ├── api/                # Route handlers
│   │   ├── auth_routes.py         # Login, register, refresh, me
│   │   ├── query_routes.py        # Query + streaming
│   │   ├── ingestion_routes.py    # Ingest, status, documents, books
│   │   ├── admin_routes.py        # Metrics, users, cache
│   │   └── health_routes.py       # Health + detailed checks
│   ├── agents/             # LangGraph pipeline
│   │   ├── graph.py        # Graph build, run_agent_query, stream_agent_query
│   │   ├── nodes.py        # 8 node implementations (AgentNodes class)
│   │   └── state.py        # AgentState TypedDict
│   ├── auth/               # JWT authentication
│   │   ├── jwt_handler.py  # Token create/decode (python-jose)
│   │   ├── dependencies.py # get_current_user, require_roles
│   │   └── password.py     # Bcrypt hash/verify
│   ├── core/               # Cross-cutting concerns
│   │   ├── middleware.py   # Request ID, timing, logging
│   │   ├── cache.py        # InMemoryCache (TTLCache)
│   │   ├── guardrails.py   # Injection detection, grounding validation
│   │   └── logging_config.py
│   ├── db/                 # Data layer
│   │   ├── models.py       # 9 SQLAlchemy models
│   │   ├── session.py      # Async engine + session factory
│   │   └── seed.py         # Default roles + admin user
│   ├── ingestion/          # Content pipeline
│   │   ├── pipeline.py     # IngestionPipeline orchestrator
│   │   ├── bookstack_client.py  # BookStack API client (httpx)
│   │   ├── content_parser.py    # HTML→text, normalize, hash
│   │   └── chunker.py          # Header/paragraph/sentence chunking
│   ├── providers/          # Pluggable backends
│   │   ├── factory.py      # Singleton factories (get_llm, get_embedding, etc.)
│   │   ├── base.py         # Abstract interfaces
│   │   ├── llm/            # OpenAICompatibleLLM, OllamaLLM
│   │   ├── embeddings/     # LocalEmbedding (SentenceTransformers)
│   │   ├── rerankers/      # CrossEncoderReranker
│   │   └── retrievers/     # Dense, Keyword, Hybrid + RRF merge
│   ├── retrieval/          # Vector store
│   │   └── vector_store.py # VectorStoreManager (Qdrant client)
│   └── schemas/            # Pydantic request/response models
```

---

## Key Design Decisions

1. **Single vector store (Qdrant)** — no FAISS/PGVector abstraction overhead
2. **Local embeddings** — SentenceTransformers (BAAI/bge-base-en-v1.5), no external API calls
3. **Flexible LLM** — supports OpenAI, Groq, OpenRouter, Ollama via single `LLM_PROVIDER` config
4. **In-memory cache** — TTL-based caching without Redis dependency
5. **Synchronous ingestion** — FastAPI `BackgroundTasks`, no Celery/Redis
6. **Toggleable pipeline nodes** — every non-essential node can be disabled via env var
7. **Hybrid retrieval** — Dense + keyword search with Reciprocal Rank Fusion
