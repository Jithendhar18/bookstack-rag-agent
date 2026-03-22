# Frontend Integration Guide

Everything needed to build a UI on top of the BookStack RAG Agent API.

**Base URL**: `http://localhost:8000/api/v1`  
**Interactive docs**: `http://localhost:8000/docs`

---

## AI Coding Prompt

> Paste this into Cursor, Copilot Chat, or Claude to bootstrap your frontend:

---

```
You are building a React + TypeScript frontend chat application for a BookStack RAG Agent.

## Stack
- React 18 + TypeScript
- Tailwind CSS
- Axios for API calls
- React Router v6

## API Base URL
http://localhost:8000/api/v1

All requests except /auth/login and /auth/register require:
  Authorization: Bearer <access_token>

---

## AUTH FLOW

POST /auth/login
Body:     { "username": "admin", "password": "admin1234" }
Response: { "access_token": "eyJ...", "refresh_token": "eyJ...", "expires_in": 1800 }

Store both tokens in localStorage.
On any 401 response: call POST /auth/refresh with { "refresh_token": "..." }, get new tokens, retry.

GET /auth/me
Response: { "id": "uuid", "username": "admin", "email": "...", "role": "admin" | "developer" | "user" }
Call this on app load to verify the token and get the user's role.

---

## CORE CHAT FLOW

### Send a message (saves to history)
POST /query
Body:
{
  "query": "Who is Rama?",
  "session_id": null        ← null for new chat, pass UUID to continue existing chat
}
Response:
{
  "answer": "Rama is the seventh avatar of Vishnu...",
  "session_id": "933d92b1-xxxx",   ← SAVE THIS. Reuse for next message in same chat.
  "latency_ms": 2340.5,
  "sources": [
    {
      "document_title": "Ramayana.pdf - Part 74",
      "source_url": "http://localhost:6875/books/ramayan/page/ramayanapdf-part-74",
      "score": 0.92,
      "chunk_id": "abc-123",
      "content": "short excerpt...",
      "metadata": {}
    }
  ]
}

CRITICAL: source_url is the reference link shown to users. Always render it as <a href={source_url}>.
Never show chunk_id or raw content as the reference. If source_url is null, show only document_title.

### Streaming (real-time, does NOT save to history)
POST /query/stream
Body: same as POST /query
Media type: text/event-stream (SSE)

Events arrive in this order — use node name to show progress:
  data: {"node": "input",              "answer": "", "sources": []}
  data: {"node": "query_rewrite",      "answer": "", "sources": []}
  data: {"node": "retriever",          "answer": "", "sources": []}
  data: {"node": "reranker",           "answer": "", "sources": []}
  data: {"node": "context_compressor", "answer": "", "sources": []}
  data: {"node": "llm_reasoning",      "answer": "Rama is...", "sources": [{source_url: "..."}]}  ← ANSWER IS HERE
  data: {"node": "response",           "answer": "Rama is...", "sources": [...]}
  data: [DONE]

Sources (with source_url) only appear from llm_reasoning onwards.
NOTE: streaming does not persist to DB. Use POST /query for history.

---

## CHAT HISTORY FLOW

### List sessions (sidebar)
GET /query/history?page=1&page_size=20
Response: [
  {
    "id": "uuid",
    "title": "Who is Rama?",        ← auto-set from first query (first 100 chars)
    "message_count": 4,
    "last_message_at": "2026-03-22T12:46:23Z",
    "created_at": "2026-03-22T12:44:01Z"
  }
]

### Load a past session
GET /query/history/{session_id}
Response:
{
  "id": "uuid",
  "title": "Who is Rama?",
  "created_at": "...",
  "updated_at": "...",
  "messages": [
    { "role": "user",      "content": "Who is Rama?", "sources": [], "created_at": "..." },
    {
      "role": "assistant",
      "content": "Rama is...",
      "sources": [
        {
          "document_title": "Ramayana.pdf - Part 74",
          "source_url": "http://localhost:6875/books/ramayan/page/ramayanapdf-part-74",
          "score": 0.92
        }
      ],
      "created_at": "..."
    }
  ]
}
Sources are on assistant messages only. user messages always have sources: [].

### Delete a session
DELETE /query/history/{session_id}
Response: 204 No Content

---

## ADMIN / DEVELOPER ENDPOINTS

### Popular questions (role: admin or developer)
GET /query/popular?limit=10
Response: [{ "query": "Who is Rama?", "count": 12, "last_asked_at": "..." }]
Show as "Trending Questions" on admin dashboard.

### Metrics (role: admin)
GET /admin/metrics
Response: { "total_documents": 42, "total_chunks": 1234, "total_users": 3, "total_queries": 99, "total_books": 5 }

### User management (role: admin)
GET  /admin/users?page=1&page_size=20
PATCH /admin/users/{id}   Body: { "role": "developer", "is_active": true, "full_name": "New Name" }

---

## TYPESCRIPT TYPES

interface LoginResponse { access_token: string; refresh_token: string; expires_in: number; }

interface User {
  id: string; email: string; username: string; full_name: string | null;
  role: "admin" | "developer" | "user"; is_active: boolean; tenant_id: string;
}

interface SourceDocument {
  chunk_id: string; document_title: string; content: string;
  score: number; source_url: string | null; metadata: Record<string, unknown>;
}

interface QueryResponse {
  answer: string; sources: SourceDocument[];
  session_id: string; trace_id: string | null; latency_ms: number;
}

interface ChatSessionListItem {
  id: string; title: string | null; message_count: number;
  last_message_at: string | null; created_at: string;
}

interface ChatMessage {
  role: "user" | "assistant"; content: string;
  sources: SourceDocument[]; created_at: string;
}

interface ChatSession {
  id: string; title: string | null;
  created_at: string; updated_at: string | null; messages: ChatMessage[];
}

interface FrequentQuestion { query: string; count: number; last_asked_at: string; }

interface AdminMetrics {
  total_documents: number; total_chunks: number; total_embeddings: number;
  total_users: number; total_queries: number; total_books: number;
  documents_by_status: Record<string, number>; documents_by_book: Record<string, number>;
}

---

## PAGES (React Router)

/login          → Login form (no auth)
/chat           → Main chat (redirect here after login)
/chat/:id       → Same chat view with a specific session loaded
/admin          → Admin dashboard (role: admin or developer only)

---

## AXIOS CLIENT

const api = axios.create({ baseURL: "http://localhost:8000/api/v1" });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    if (error.response?.status === 401) {
      const refresh = localStorage.getItem("refresh_token");
      if (refresh) {
        const { data } = await axios.post(
          "http://localhost:8000/api/v1/auth/refresh",
          { refresh_token: refresh }
        );
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        error.config.headers.Authorization = `Bearer ${data.access_token}`;
        return api(error.config);
      }
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

---

## CHAT STATE

const [sessionId, setSessionId] = useState<string | null>(null);
const [messages, setMessages]   = useState<ChatMessage[]>([]);

async function sendMessage(text: string) {
  // Optimistically add user message
  setMessages(m => [...m, { role: "user", content: text, sources: [], created_at: new Date().toISOString() }]);

  const { data } = await api.post<QueryResponse>("/query", {
    query: text,
    session_id: sessionId,   // null on first message
  });

  if (!sessionId) setSessionId(data.session_id);  // save for subsequent messages

  setMessages(m => [...m, {
    role: "assistant",
    content: data.answer,
    sources: data.sources,   // source_url inside each source → render as <a href>
    created_at: new Date().toISOString(),
  }]);
}

async function loadSession(id: string) {
  const { data } = await api.get<ChatSession>(`/query/history/${id}`);
  setSessionId(data.id);
  setMessages(data.messages);  // sources already contain source_url
}

function newChat() { setSessionId(null); setMessages([]); }

---

## SOURCE LINK COMPONENT (CRITICAL)

function SourceList({ sources }: { sources: SourceDocument[] }) {
  const withUrl = sources.filter(s => s.source_url);
  if (!withUrl.length) return null;
  return (
    <div className="mt-2 text-sm">
      <p className="font-semibold text-gray-500">Sources:</p>
      <ul className="list-none space-y-1">
        {withUrl.map(src => (
          <li key={src.chunk_id}>
            <a
              href={src.source_url!}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              📄 {src.document_title}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

---

## STREAMING (OPTIONAL)

const nodeLabels: Record<string, string> = {
  input: "Understanding question...",
  query_rewrite: "Reformulating...",
  retriever: "Searching documentation...",
  reranker: "Ranking results...",
  context_compressor: "Preparing context...",
  llm_reasoning: "Generating answer...",
  response: "Done",
};

async function sendMessageStream(text: string, token: string) {
  const res = await fetch("http://localhost:8000/api/v1/query/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ query: text, session_id: sessionId }),
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      const raw = part.slice(6).trim();
      if (raw === "[DONE]") { setProgressStep(""); return; }

      const event = JSON.parse(raw);
      setProgressStep(nodeLabels[event.node] ?? "");
      if (event.answer) setAnswer(event.answer);
      if (event.sources?.length) setSources(event.sources);  // source_url is here
    }
  }
}
// Streaming does NOT save to history. Call POST /query afterwards if you need persistence.

---

## ROLE GUARDS

function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => { api.get("/auth/me").then(r => setUser(r.data)).catch(() => setUser(null)); }, []);
  return {
    user,
    isAdmin: user?.role === "admin",
    isDeveloper: user?.role === "developer" || user?.role === "admin",
  };
}

// In router:
// { path: "/admin", element: isAdmin || isDeveloper ? <AdminPage /> : <Navigate to="/chat" /> }

---

## ERROR HANDLING

| Status | Meaning | Action |
|--------|---------|--------|
| 400 | Validation error | Show response.detail |
| 401 | Token expired | Auto-refresh via /auth/refresh |
| 403 | Wrong role | Show "Not authorized" |
| 404 | Not found | Show "Not found" |
| 422 | Body validation | Show detail[0].msg |
| 500 | Server error | Show generic error |

All errors: { "detail": "Error message" }

---

## RULES
1. source_url is ALWAYS the reference link — render as <a href={source_url}>document_title</a>
2. Never display chunk_id or raw content chunk as a reference to the user
3. Always pass session_id from the first POST /query response into all subsequent messages
4. Sources only appear on role === "assistant" messages
5. Guard /admin with role check from GET /auth/me
6. On app load: call GET /auth/me → redirect to /login if it fails
```

---

## UI Layout

```
App
├── /login                → Login form
├── /chat                 → New chat (default after login)
├── /chat/:id             → Restore past session
└── /admin                → Admin dashboard (admin/developer only)

Chat Page:
  ┌─────────────────┬──────────────────────────────────┐
  │ Sidebar (300px) │  Chat area                        │
  │                 │                                   │
  │ [New Chat]      │  [User bubble]   right-aligned    │
  │                 │  [Assistant]     left-aligned      │
  │ Past sessions:  │    Answer text                    │
  │  • Who is Rama? │    📄 Source link 1               │
  │  • What is...   │    📄 Source link 2               │
  │                 │                                   │
  │                 │  [Input bar] [Send]               │
  └─────────────────┴──────────────────────────────────┘

Admin Page:
  Metrics cards | Popular Questions table | Users table
```

## When to Call Each Endpoint

| User action | Endpoint |
|---|---|
| App loads | `GET /auth/me` |
| Login form submit | `POST /auth/login` |
| Sidebar mounts | `GET /query/history` |
| User taps past chat | `GET /query/history/:id` |
| User sends message | `POST /query` |
| User deletes chat | `DELETE /query/history/:id` |
| Admin opens dashboard | `GET /admin/metrics` |
| Admin views trending | `GET /query/popular` |
| Admin manages users | `GET /admin/users` |
| Any 401 received | `POST /auth/refresh` |
