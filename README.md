# LM Studio Optimization Proxy

OpenAI-compatible FastAPI proxy that sits in front of LM Studio and improves requests automatically:

- Detects the prompt “domain” (code / reasoning / business) and applies safer defaults (temperature, token caps)
- Prunes/compresses long chats to reduce token usage
- Optionally injects “relevant context” from provided chunks/URLs using reranking
- Supports streaming (SSE) and passes through tools/structured outputs to LM Studio
- Exposes helper endpoints for hardware info, embeddings, reranking, and LM Studio preset generation

The main server is [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py).

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Configure (Windows PowerShell):

```powershell
$env:LMSTUDIO_BASE_URL="http://127.0.0.1:1234"
$env:BRIDGE_HOST="0.0.0.0"
$env:BRIDGE_PORT="8080"
```

Configure (macOS/Linux):

```bash
export LMSTUDIO_BASE_URL="http://127.0.0.1:1234"
export BRIDGE_HOST="0.0.0.0"
export BRIDGE_PORT="8080"
```

Run:

```bash
python proxy.py
```

## OpenAI SDK Example

Point your client to the proxy (not LM Studio directly):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080", api_key="lm-studio")

resp = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Write a Python function to merge two dicts."}],
    stream=True,
)

for event in resp:
    if event.choices and event.choices[0].delta and event.choices[0].delta.content:
        print(event.choices[0].delta.content, end="")
```

## Context Enrichment (Optional)

If you include `extra_body.context_docs`, the proxy can:

- Fetch URLs (first ~4000 chars)
- Rerank your chunks against the user prompt (top `RERANK_TOP_K`)
- Inject a `system` message: `Relevant context: ...`

Example request body (OpenAI-compatible shape):

```json
{
  "model": "qwen3.5-4b",
  "stream": true,
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Summarize the page and extract key facts." }
  ],
  "extra_body": {
    "context_docs": [
      { "url": "https://httpbin.org/html" },
      { "text": "Local notes: the user cares about pricing + limits." }
    ]
  }
}
```

## API Endpoints

Main OpenAI-compatible proxy:

- `POST /v1/chat/completions` — proxies to LM Studio with optimization and optional context enrichment

Utilities:

- `GET /api/v1/models` — fetches LM Studio’s model list (native API)
- `GET /api/v1/model-stats` — lightweight list of id/state/size from LM Studio
- `GET /api/v1/hardware` — detected hardware profile (GPU/VRAM/CPU/RAM)
- `POST /api/v1/embed` — embeddings via LM Studio `/v1/embeddings`
- `POST /api/v1/rerank` — simple cosine-similarity rerank (via embeddings)
- `POST /api/v1/benchmark/{model_id}` — starts an async benchmark task
- `GET /api/v1/benchmark/compare?model_id=...` — compares streaming vs non-streaming
- `POST /api/v1/presets/generate?model_id=...&use_case=balanced&params_b=4.0` — generates an LM Studio preset JSON and saves it locally
- `GET /api/v1/presets/list` — lists generated preset JSON files

## Configuration

Environment variables used by the proxy:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LMSTUDIO_BASE_URL` | `http://192.168.1.12:1234` | Where LM Studio is reachable |
| `BRIDGE_HOST` | `0.0.0.0` | Bind host for the proxy |
| `BRIDGE_PORT` | `8080` | Bind port for the proxy |
| `EMBED_MODEL` | `nomic-embed-text-v1.5` | Embedding model id sent to `/v1/embeddings` |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Reranker model id (used when injecting context) |
| `RERANK_TOP_K` | `3` | How many reranked chunks to inject |

## Extra Tools (Optional)

Config wizard (GUI by default, TUI available):

```bash
python cli.py
python cli.py --tui
```

Standalone MCP-like tool server:

```bash
python mcp_server.py
```

It starts on `http://0.0.0.0:8090` and exposes:

- `GET /tools`
- `POST /tools/call`

## License

MIT License. Created by [Alireza Ghaffari](https://github.com/ghaffaria).
