# 🚀 LM Studio Optimization Bridge (V5.0)

> **What LLMs alone cannot do with a single API call — this bridge does.**

The V5 bridge is an **intelligent inference middleware** that sits between AI agents (OpenCode, Continue, Cursor, Copilot) and LM Studio. It transforms raw API calls into **structured, optimized, tool-augmented responses** — handling the heavy lifting that a pure API call cannot.

## 🎯 The Core Problem

An LLM API call is **dumb**: it takes text in, gives text out. It cannot:
- Rerank or retrieve relevant context before answering
- Decide whether to use a tool vs generate text
- Structure output into typed schemas (JSON, code, plans)
- Switch between fast/cheap and slow/thinking modes dynamically
- Embed and search your codebase or knowledge base

The V5 bridge solves this by wrapping every API call in a **smart routing layer** that inspects the request and applies the right pipeline.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              AGENTIC FRAMEWORKS                                  │
│        OpenCode · Continue · Cursor · LangChain                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  OpenAI-compatible API  (port 1234)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              🔀 REQUEST ROUTER                                   │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │  /rerank      │  │  /embed      │  │  /chat/complete   │    │
│  │  (semantic    │  │  (generate   │  │  (standard        │    │
│  │   reranking)  │  │   vectors)   │  │   proxy + enhance)│    │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬─────────┘    │
│         │                 │                     │               │
│         ▼                 ▼                     ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              🧠 INFERENCE ORCHESTRATOR                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │   │
│  │  │  Fast Path │  │ Thinking   │  │  Tool / MCP Path  │ │   │
│  │  │  (direct   │  │ Pipeline   │  │  (delegate to     │ │   │
│  │  │  LM Studio)│  │ (iterate,  │  │  external tools)  │ │   │
│  │  │            │  │  reflect)  │  │                   │ │   │
│  │  └────────────┘  └────────────┘  └────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           📐 STRUCTURED OUTPUT FORMATTING                │   │
│  │   JSON Schema · TypeScript · Python Pydantic · SQL      │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  LM Studio REST API v1  (port 1234)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LM STUDIO                                    │
│        (Inference Engine · GGUF · CUDA/Vulkan)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ What's New in V5

| Feature | V4 | V5 |
|---------|----|----|
| **Streaming** | SSE with truncation bugs | SSE + structured chunks + heartbeat |
| **Context** | 10-turn prune only | Semantic reranking + embedding retrieval |
| **Tools** | None | MCP-first tool dispatch |
| **Output** | Text formatting only | JSON/Code/Plan schemas |
| **Thinking** | One-shot | Fast · Think · Architect modes |
| **Input enrichment** | None | Rerank + embed before inference |
| **Stop tokens** | Broke code streaming | Explicit block markers only |

---

## 🔀 Routing Modes

Every incoming request is classified into one of three modes:

### ⚡ Fast Mode
When `mode=fast` or the prompt is a short question:
- Direct passthrough to LM Studio with minimal overhead
- Temperature ≤ 0.3, max_tokens capped by hardware
- Response cached in LRU

### 🧠 Think Mode
When `mode=think` or prompt contains `why`/`explain`/`analyze`:
- Iterative reflection loop: generate → score → refine
- Structured output: bullet points, step-by-step
- `reasoning_content` delta extracted and surfaced separately

### 🏗️ Architect Mode
When `mode=architect` or prompt is a complex multi-step task:
- Breaks task into sub-tasks
- Each sub-task dispatched as a tool call or sub-prompt
- Final answer assembled from all sub-results
- Supports MCP tool chains

---

## 📥 Input Pipeline (What LLMs Can't Do Alone)

### 1. Semantic Reranking (`/rerank`)
```
Request:  "how do I sort a list in Python?"
Context:  [file1.py, file2.py, docs/api.md, ...]

→ Embed all context chunks
→ Compute cosine similarity to query
→ Reorder: [docs/api.md, file1.py, ...]
→ Inject top-3 chunks into messages
```

### 2. Embedding Generation (`/embed`)
```
Request:  "summarize the codebase"

→ Chunk the codebase into paragraphs
→ POST each chunk to embedding endpoint
→ Return vector array for downstream RAG
```

### 3. URL Context Fetch
If the prompt contains URLs, the bridge fetches them **before** sending to the model:
```
Request:  "summarize https://httpbin.org/html"

→ httpx.get() the URL content
→ Inject as system message: "Context from https://...: {content}"
→ Then forward to LM Studio
```

---

## 📤 Output Pipeline (Structured, Not Just Text)

### JSON Schema Mode
When `response_format={"type": "json", "schema": {...}}`:
- Validates output against schema
- Retries on failure (up to 2 attempts)
- Strips markdown fences, returns raw JSON

### Code Block Mode
When detected code output:
- Auto-closes unclosed fences
- Applies syntax hints if `language` tag present
- Never injects `stop=["```"]` (V4 bug fixed)

### Plan Mode
When task is multi-step:
- Returns structured steps with status markers
- Each step can be executed independently
- Assembles into final answer

---

## 🔧 MCP / Tool Integration

The bridge ships with an MCP server (`mcp_server.py`) that exposes LM Studio as a tool:

```json
{
  "tools": [
    {
      "name": "complete",
      "description": "Streaming chat completion with context reranking",
      "input_schema": {
        "model": "string",
        "messages": "array",
        "mode": "fast|think|architect",
        "stream": "boolean",
        "context_docs": "optional array of doc chunks"
      }
    },
    {
      "name": "embed",
      "description": "Generate embeddings for text chunks",
      "input_schema": {
        "texts": "array of strings"
      }
    },
    {
      "name": "rerank",
      "description": "Rerank context chunks by relevance to query",
      "input_schema": {
        "query": "string",
        "chunks": "array of strings",
        "top_k": "number"
      }
    }
  ]
}
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
export LMSTUDIO_BASE_URL=http://192.168.1.12:1234   # Your LM Studio server
export BRIDGE_PORT=1234                              # Bridge listens here
export BRIDGE_HOST=0.0.0.0                           # Accept from any NIC
```

### 3. Start
```bash
python proxy.py
```

### 4. Use from OpenCode/Continue

```python
client = OpenAI(
    base_url="http://192.168.1.12:1234/v1",
    api_key="lm-studio"
)

# Fast mode (simple question)
client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=True
)

# Think mode (reasoning)
client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Explain why the sky is blue"}],
    extra_body={"mode": "think"}
)

# Architect mode (structured plan)
client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Build a web scraper in Python"}],
    extra_body={"mode": "architect", "response_format": {"type": "json", "schema": {
        "steps": [{"order": "int", "action": "string", "tool": "string"}]
    }}}
)
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Main proxy with routing + optimization |
| `/v1/models` | GET | List models (cached 2s TTL) |
| `/api/v1/models` | GET | Native LM Studio model list |
| `/api/v1/hardware` | GET | Detected hardware profile |
| `/api/v1/embed` | POST | Generate embeddings |
| `/api/v1/rerank` | POST | Rerank context chunks |
| `/api/v1/mcp/tools` | GET | List available MCP tools |
| `/api/v1/benchmark/{model}` | POST | Run streaming benchmark |

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LMSTUDIO_BASE_URL` | `http://192.168.1.12:1234` | LM Studio server |
| `BRIDGE_PORT` | `1234` | Bridge listening port |
| `BRIDGE_HOST` | `0.0.0.0` | Bind address |
| `CACHE_SIZE` | `100` | LRU cache entries |
| `MAX_CONVERSATION_TURNS` | `10` | Context window |
| `RERANK_TOP_K` | `3` | Context chunks to inject |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |

---

## 🔮 Roadmap

- [ ] **Embedding endpoint** — `/api/v1/embed` with local embedding model
- [ ] **Reranking endpoint** — `/api/v1/rerank` with cross-encoder scoring
- [ ] **URL pre-fetch middleware** — auto-fetch URLs in prompts before inference
- [ ] **MCP tool registry** — dynamic tool discovery and dispatch
- [ ] **Retry with schema validation** — JSON output guaranteed or retries
- [ ] **Thought decomposition** — Architect mode sub-task graph
- [ ] **Flash Attention detection** — auto-enable based on GPU compute
- [ ] **Redis cache backend** — for multi-instance deployments
- [ ] **Vector store integration** — Chroma/Pinecone for RAG

---

## ⚖️ License
MIT License. Created by [Alireza Ghaffari](https://github.com/ghaffaria).
