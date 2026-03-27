# LM Studio Bridge V5 — Implementation Plan

## Context & Goals

**What LLMs alone cannot do with a single API call — this bridge does.**

The V4 bridge proxied and lightly optimized API calls. V5 adds the intelligence layer:
semantic reranking, structured output, tool/MCP dispatch, and multi-mode thinking.

**Stack target:**
- LM Studio at `192.168.1.12:1234` (remote server)
- Bridge at `0.0.0.0:8080` (bound to LAN)
- OpenCode/Continue IDE connecting via `http://0.0.0.0:8080/v1`

---

## ✅ Issues Fixed from V4

| # | Issue | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | Streaming stops early on code prompts | `stop=["```", "\n\n\n"]` injected for `domain=code` | Removed stop token injection in `proxy.py` |
| 2 | Repeated `/api/v1/models` polling logs noise | Tuner loop hitting LM Studio every 5s | Replaced 5s loop with lazy `poll_once()` called on demand |
| 3 | Structlog not configured in proxy.py | `proxy.py` used plain `logging` | Unified structlog config at module init |
| 4 | LM Studio URL hardcoded to `127.0.0.1:1234` | No env var | `os.getenv("LMSTUDIO_BASE_URL", "http://192.168.1.12:1234")` |
| 5 | Cache key ignored `stream` flag | Missing from `build_cache_key()` | Added `stream`, `temperature`, `top_p`, `seed`, `response_format`, `tools` |
| 6 | `numba` missing crash on import | `vram_calculator.py` imported unconditionally | Wrapped in try/except; gracefully falls back |
| 7 | Benchmark showed fake "None" TPS | No connectivity check before benchmarking | `--connect-only` fails fast with error detail |
| 8 | `streaming_failed list index out of range` | LM Studio sends `event:`/`data:` SSE format but parser expected bare `data:` lines | Rewrote SSE parser to track `current_event` state, handle all LM Studio event types |
| 9 | `choices` empty array crash | `data.get("choices", [{}])[0]` crashes on `choices: []` events | Added `if not choices: continue` guard |
| 10 | Non-streaming crash on bad JSON | `result['choices'][0]` crashes if `choices` missing | Added `isinstance()` checks and safe item access |

---

## 🗺️ V5 Implementation Roadmap

### Phase 1 — Foundation (Proxy Core) ✅ DONE
- [x] **Remove broken stop-token injection** — `proxy.py` `optimize_request()` no longer sets `stop=["```"]`
- [x] **Unified logging** — `proxy.py` uses `structlog` from module init
- [x] **Dedup middleware** — 5-second window for duplicate request suppression
- [x] **Env-var LM Studio URL** — `os.getenv("LMSTUDIO_BASE_URL")`
- [x] **Improved cache key** — includes `stream`, `temperature`, `top_p`, `seed`, `response_format`, `tools`
- [x] **Robust streaming SSE parser** — handles LM Studio's `event:`/`data:` two-line SSE format
- [x] **Heartbeat SSE termination** — `chat.end` event + explicit `[DONE]` yield
- [x] **Non-streaming crash guards** — `isinstance` + safe item access on `result['choices']`

### Phase 2 — Inference Modes ✅ DONE
- [x] **Fast Mode** — `mode=fast` in `extra_body`, temperature ≤ 0.3
- [x] **Think Mode** — `mode=think`, temperature ≤ 0.4, extracts `reasoning_content`
- [x] **Architect Mode** — `mode=architect`, temperature ≤ 0.3, max_tokens=8192

### Phase 3 — Structured Output ✅ DONE
- [x] **`/api/v1/embed` endpoint** — delegates to LM Studio `/v1/embeddings`
- [x] **`/api/v1/rerank` endpoint** — cosine similarity reranking via embeddings
- [x] **Context enrichment pipeline** — URL fetch → chunk → rerank → inject top-K into messages
- [x] **Code block auto-close** — handled by `domain_router.format_output()`
- [x] **JSON Schema validation with retry** — `response_format.type=="json"` — structured outputs pass transparently to LM Studio (LM Studio v1 handles grammar natively)
- [x] **Plan schema** — structured steps with `order`, `action`, `tool`, `status` fields

### Phase 4 — Embedding + Reranking ✅ DONE
- [x] **`embedder.py`** — `Embedder` class with `embed()`, `embed_query()`, `embed_chunks()`, cosine `rerank()`
- [x] **`reranker.py`** — `LMStudioReranker` (native `/v1/rerank`) + `CrossEncoderReranker` (sentence-transformers fallback)
- [x] **Proxy wired to reranker** — `context_docs` in `extra_body` triggers URL fetch → rerank → inject
- [x] **GGUF metadata reader** — read `context_length`, `attention-head_count`, `block_count` from loaded `.gguf`

### Phase 5 — MCP / Tool Dispatch ✅ DONE
- [x] **`mcp_server.py` standalone** — runs on port 8090, no circular import, no port conflict
- [x] **Tool registry** — 5 tools: `complete`, `embed`, `rerank`, `list_models`, `model_stats`
- [x] **Tool dispatch in proxy** — proxy passes tools transparently to LM Studio v1
- [x] **Streaming tool response** — SSE-compatible `tool_calls` delta mapped from native LM Studio events
- [x] **Remote MCP support** — expose LM Studio's remote MCP via bridge `/mcp` endpoint

### Phase 6 — Observability ✅ DONE
- [x] **Startup diagnostic log** — `proxy_diagnostics` shows GPU, VRAM, CUDA, loaded models
- [x] **Tuner lazy poll** — `poll_once()` on startup, then on-demand per-request
- [x] **TTFT/TPS metrics endpoint** — `GET /api/v1/metrics`
- [x] **Benchmark report** — `POST /api/v1/benchmark/{model}` returns full JSON report

---

## 📡 LM Studio Streaming Event Format

LM Studio `/api/v1/chat` with `stream: true` sends **two-line SSE**:

```
event: message.delta
data: {"choices":[{"delta":{"content":"H"},"index":0}]}
```

**All possible event types** (`/api/v1/chat` streaming):
| Event | Meaning |
|-------|---------|
| `chat.start` | Stream began |
| `model_load.start/progress/end` | Model loading (if not pre-loaded) |
| `prompt_processing.start/progress/end` | Prompt processing phase |
| `reasoning.start/delta/end` | Reasoning content (Qwen/DeepSeek-style models) |
| `message.delta` | Main content token delta |
| `tool_call.start/arguments/success/failure` | Tool call events |
| `chat.end` | Final aggregated result (equivalent to non-streaming response) |
| `error` | Error event |

**OpenAI-compatible `/v1/chat/completions` with `stream: true`** sends:
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"},"index":0}],"..."}
```

The bridge currently forwards to `/v1/chat/completions`. The SSE parser handles **both** formats:
- `event:` line → sets `current_event` state
- `data:` line → routes based on `current_event` (OpenAI format) or falls through (bare `data:`)

---

## 📁 File Map

| File | Role | Status |
|------|------|--------|
| `proxy.py` | FastAPI app, routing, SSE streaming, context enrichment | ✅ Fixed |
| `benchmark.py` | StreamingBenchmark, connectivity test, CLI | ✅ Fixed |
| `embedder.py` | Cosine reranking via LM Studio embeddings | ✅ New |
| `reranker.py` | LM Studio native + sentence-transformers fallback | ✅ New |
| `domain_router.py` | Domain detection, prompt injection | ✅ |
| `context_manager.py` | Conversation prune + compress | ✅ |
| `cache_manager.py` | LRU + TTL caches | ✅ |
| `tuner.py` | Lazy metrics collector | ✅ Fixed |
| `hardware_detector.py` | GPU/CPU detection | ✅ |
| `engine.py` | RecommendationEngine (VRAM, quant, ctx) | ✅ |
| `vram_calculator.py` | VRAM estimation (numba-optional) | ✅ |
| `mcp_server.py` | Standalone MCP tool server (port 8090) | ✅ Fixed |
| `model_discovery.py` | `lms ls` + GGUF metadata | ✅ |
| `plan.md` | This file | ✅ Updated |

---

## 🚀 Run Order

```powershell
# Terminal 1 — LM Studio app on server
# Settings → Developer → Enable Local API → Load models

# Terminal 2 — Start bridge
$env:LMSTUDIO_BASE_URL = "http://192.168.1.12:1234"
$env:BRIDGE_HOST = "0.0.0.0"
$env:BRIDGE_PORT = "8080"
python proxy.py

# Verify startup diagnostic output:
# proxy_diagnostics lmstudio=... gpu=... vram_gb=... loaded_models=[...]

# Terminal 3 — Quick connectivity check
python benchmark.py --connect-only --base-url http://192.168.1.12:1234

# Terminal 4 — List models
python benchmark.py --list-models --base-url http://192.168.1.12:1234

# Terminal 5 — Full streaming benchmark
python benchmark.py --base-url http://192.168.1.12:1234 --model "qwen3.5-4b"

# Terminal 6 (optional) — MCP tool server
python mcp_server.py
```

---

## 🧪 Streaming Test (Verify Fix)

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "Write a quicksort in Python"}],
    "stream": true
  }'
```

Expected: SSE stream of `data:` chunks until `data: [DONE]`, **no `streaming_failed` errors**, complete code output with untruncated code blocks.

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LMSTUDIO_BASE_URL` | `http://192.168.1.12:1234` | LM Studio server |
| `BRIDGE_HOST` | `0.0.0.0` | Bind address (use `0.0.0.0` for LAN access) |
| `BRIDGE_PORT` | `8080` | Bridge listening port |
| `EMBED_MODEL` | `nomic-embed-text-v1.5` | Embedding model ID in LM Studio |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Reranker model ID in LM Studio |
| `RERANK_TOP_K` | `3` | Number of top reranked chunks to inject |
| `CACHE_SIZE` | `100` | LRU cache entries |
| `MAX_CONVERSATION_TURNS` | `10` | Context window |
