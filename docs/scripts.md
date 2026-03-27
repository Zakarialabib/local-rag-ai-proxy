# Script Reference (Proxy API Project)

This repository contains two main things:

- **The proxy server** (FastAPI) that sits in front of LM Studio and exposes OpenAI-compatible endpoints.
- **A “config wizard” toolkit** (GUI/TUI/CLI) that can analyze your hardware + local models and export LM Studio preset JSON files.

If you only care about the proxy server, you can keep a small subset of files (see “What You Can Drop”).

This documentation is about **this proxy-api repository only** (it does not document the Continue extension).

## What Impacts This Project The Most

If you are troubleshooting “proxy didn’t respond correctly” or deciding what to keep/drop, these are the highest-impact parts:

- **Streaming path**: `POST /v1/chat/completions` streaming handling in [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py). This is where SSE parsing and output compatibility issues appear first.
- **Request mutation**: `optimize_request()` in [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) changes `max_tokens`/`temperature` based on domain and mode; this can change behavior compared to talking to LM Studio directly.
- **Context manipulation**: [context_manager.py](file:///c:/Users/dell/Desktop/proxy-api/context_manager.py) pruning/compression can remove earlier turns and change answers.
- **Prompt injection**: [domain_router.py](file:///c:/Users/dell/Desktop/proxy-api/domain_router.py) injects/extends system prompts; bad prompts can degrade tool use or JSON correctness.
- **Installation dependencies**: [requirements.txt](file:///c:/Users/dell/Desktop/proxy-api/requirements.txt) currently has conflicts/omissions that can break runtime behavior even if the code compiles.

## Quick Run

```bash
pip install -r requirements.txt
python proxy.py
```

Environment variables:

| Variable | Default | Where used | Meaning |
|----------|---------|------------|---------|
| `LMSTUDIO_BASE_URL` | `http://192.168.1.12:1234` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | Where the proxy forwards requests (LM Studio base URL) |
| `BRIDGE_HOST` | `0.0.0.0` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | Host to bind the proxy |
| `BRIDGE_PORT` | `8080` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | Port to bind the proxy |
| `EMBED_MODEL` | `nomic-embed-text-v1.5` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | Embedding model id passed to LM Studio `/v1/embeddings` |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | Reranker model id used for context injection |
| `RERANK_TOP_K` | `3` | [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) | How many top chunks to inject as “Relevant context” |

## Proxy API (Endpoints)

Main OpenAI-compatible endpoint:

- `POST /v1/chat/completions` — forwards to `LMSTUDIO_BASE_URL/v1/chat/completions` with optimizations:
  - domain detection (code/reasoning/business)
  - chat pruning + optional compression
  - optional context injection using `extra_body.context_docs`
  - streaming SSE pass-through (plus some event normalization)

Utility endpoints:

- `GET /api/v1/models` — returns LM Studio model list (native API, uncached)
- `GET /api/v1/model-stats` — smaller model summary (id/state/size)
- `GET /api/v1/hardware` — detected hardware profile (GPU/VRAM/CPU/RAM)
- `POST /api/v1/embed` — forwards to `LMSTUDIO_BASE_URL/v1/embeddings`
- `POST /api/v1/rerank` — embedding-based cosine similarity rerank
- `POST /api/v1/benchmark/{model_id}` — runs benchmark in background task
- `GET /api/v1/benchmark/compare?model_id=...` — benchmark helper
- `POST /api/v1/presets/generate?...` — generates an LM Studio preset JSON and saves it locally
- `GET /api/v1/presets/list` — lists generated preset JSON files

Note: `benchmark.py` tries `GET /v1/models` by default, but the current [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) does not expose `GET /v1/models`. Use `GET /api/v1/models` instead, or point the benchmark directly at LM Studio.

## Script Inventory

### Server / API

- [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py)
  - Purpose: FastAPI server that proxies LM Studio with optimization (domain routing, context pruning, optional rerank injection, caching, streaming handling).
  - How to run: `python proxy.py` (binds using `BRIDGE_HOST`/`BRIDGE_PORT`).
  - Key dependencies:
    - [hardware_detector.py](file:///c:/Users/dell/Desktop/proxy-api/hardware_detector.py) for hardware info
    - [engine.py](file:///c:/Users/dell/Desktop/proxy-api/engine.py) for preset recommendations
    - [tuner.py](file:///c:/Users/dell/Desktop/proxy-api/tuner.py) for lazy loaded-model polling
    - [cache_manager.py](file:///c:/Users/dell/Desktop/proxy-api/cache_manager.py) for response/model-list caching
    - [domain_router.py](file:///c:/Users/dell/Desktop/proxy-api/domain_router.py) for prompt “domain” detection and formatting
    - [context_manager.py](file:///c:/Users/dell/Desktop/proxy-api/context_manager.py) for pruning/compression
    - [reranker.py](file:///c:/Users/dell/Desktop/proxy-api/reranker.py) for LM Studio rerank calls (context injection)

- [mcp_server.py](file:///c:/Users/dell/Desktop/proxy-api/mcp_server.py)
  - Purpose: Small FastAPI “tool server” that exposes endpoints like `/tools` and `/tools/call` to call LM Studio actions (complete/embed/rerank/list_models).
  - How to run: `python mcp_server.py` (listens on port `8090`).
  - Notes:
    - It calls LM Studio directly (hardcoded `LMSTUDIO_BASE` inside this file).
    - It does not call `proxy.py` unless you change its base URL.

### Benchmarking

- [benchmark.py](file:///c:/Users/dell/Desktop/proxy-api/benchmark.py)
  - Purpose: Benchmark helper for streaming/non-streaming, simple “capability detection” (reasoning/tool use/vision/url), truncation detection.
  - How to use:
    - Imported by `proxy.py` for the `/api/v1/benchmark/...` endpoints.
    - Can also be run as a CLI tool:
      - `python benchmark.py --connect-only --base-url http://127.0.0.1:1234`
      - `python benchmark.py --list-models --base-url http://127.0.0.1:1234`
      - `python benchmark.py --base-url http://127.0.0.1:8080 --model qwen3.5-4b --prompt code`

### Core “optimization” helpers (used by proxy)

- [cache_manager.py](file:///c:/Users/dell/Desktop/proxy-api/cache_manager.py)
  - Purpose: LRU cache for non-streaming responses + short TTL cache for model list.
  - Used by: `proxy.py`.

- [context_manager.py](file:///c:/Users/dell/Desktop/proxy-api/context_manager.py)
  - Purpose: trims long conversations to the last N turns; optional “compress” step that summarizes older turns into a synthetic system message.
  - Used by: `proxy.py`.

- [domain_router.py](file:///c:/Users/dell/Desktop/proxy-api/domain_router.py)
  - Purpose: “domain” detection via regex (code/json/data/business/reasoning), system prompt injection, light output formatting.
  - Used by: `proxy.py`.
  - Customization: edits/overrides can be added via [custom_prompts.json](file:///c:/Users/dell/Desktop/proxy-api/custom_prompts.json).

- [custom_prompts.json](file:///c:/Users/dell/Desktop/proxy-api/custom_prompts.json)
  - Purpose: overrides built-in domain system prompts (e.g. code/reasoning/creative).
  - Used by: `domain_router.py` (loaded at import time).

- [reranker.py](file:///c:/Users/dell/Desktop/proxy-api/reranker.py)
  - Purpose: rerank utilities:
    - `LMStudioReranker` calls `LMSTUDIO_BASE/v1/rerank` (if your LM Studio build exposes it).
    - `CrossEncoderReranker` is an optional local fallback (requires `sentence_transformers`, not in `requirements.txt`).
  - Used by: `proxy.py` for context injection when `extra_body.context_docs` is provided.

- [embedder.py](file:///c:/Users/dell/Desktop/proxy-api/embedder.py)
  - Purpose: embedding client for `LMSTUDIO_BASE/v1/embeddings` + cosine scoring + a convenience `rerank()` implementation.
  - Used by: `proxy.py` constructs an `Embedder`, but current proxy flow uses `reranker.py` for injection and uses `/api/v1/rerank` for the simpler embedding-based rerank endpoint.
  - Dependencies: uses `numpy` at runtime; ensure it is installed (see “Project-impacting issues” below).

- [tuner.py](file:///c:/Users/dell/Desktop/proxy-api/tuner.py)
  - Purpose: lazy polling of LM Studio `/api/v1/models` to record which models are loaded.
  - Used by: `proxy.py` startup (lifespan).

### Hardware / recommendation engine (used by proxy and by wizard UI)

- [hardware_detector.py](file:///c:/Users/dell/Desktop/proxy-api/hardware_detector.py)
  - Purpose: cross-platform hardware detection (CPU/RAM + GPU/VRAM + CUDA version + compute capability).
  - Used by: `proxy.py`, `gui.py`, `tui.py`.

- [models.py](file:///c:/Users/dell/Desktop/proxy-api/models.py)
  - Purpose: Pydantic models/enums for hardware profile and recommendation output.
  - Used by: the recommendation engine and exporter tooling.

- [vram_calculator.py](file:///c:/Users/dell/Desktop/proxy-api/vram_calculator.py)
  - Purpose: estimates VRAM use for weights + KV cache + runtime overhead; used by the recommendation engine.
  - Used by: `engine.py`.

- [engine.py](file:///c:/Users/dell/Desktop/proxy-api/engine.py)
  - Purpose: produces “top recommended” model configurations based on your hardware and a rough VRAM math model.
  - Used by: `proxy.py` preset generation endpoint, `gui.py`, `tui.py`.

### Model discovery + preset exporting (wizard toolchain)

- [model_discovery.py](file:///c:/Users/dell/Desktop/proxy-api/model_discovery.py)
  - Purpose:
    - `get_local_models()` parses `lms ls` output (LM Studio CLI) to list installed models.
    - `extract_model_specs()` tries to read GGUF metadata (via `gguf` library) or fallback `config.json`.
  - Used by: `gui.py`, `tui.py`.

- [exporters.py](file:///c:/Users/dell/Desktop/proxy-api/exporters.py)
  - Purpose: builds and writes LM Studio “preset JSON” format from a `ModelRecommendation`.
  - Used by: `gui.py`, `tui.py`.

- [export_utils.py](file:///c:/Users/dell/Desktop/proxy-api/export_utils.py)
  - Purpose: older export helpers (YAML/preset json) that take generic dicts.
  - Used by: not imported by the current GUI/TUI code paths (kept as legacy utilities).

- [recommender.py](file:///c:/Users/dell/Desktop/proxy-api/recommender.py)
  - Purpose: older heuristic recommender that estimates VRAM and returns settings dict.
  - Used by: not used by the current `engine.py` path (kept as legacy logic).

- [model_profile.py](file:///c:/Users/dell/Desktop/proxy-api/model_profile.py)
  - Purpose: interactive Questionary prompts to build a “model profile” by asking the user questions.
  - Used by: legacy/standalone usage (not used by proxy server).

- [hardware_utils.py](file:///c:/Users/dell/Desktop/proxy-api/hardware_utils.py)
  - Purpose: Rich terminal hardware report + fallback GPU detection. Similar scope to `hardware_detector.py`, but returns plain dicts.
  - Used by: standalone script / older flows.

### Wizard entrypoints (optional)

- [cli.py](file:///c:/Users/dell/Desktop/proxy-api/cli.py)
  - Purpose: single entrypoint for the config wizard.
  - How to use:
    - `python cli.py` launches the GUI (CustomTkinter) if installed
    - `python cli.py --tui` launches the Textual TUI

- [gui.py](file:///c:/Users/dell/Desktop/proxy-api/gui.py)
  - Purpose: modern GUI for scanning models and exporting LM Studio presets.
  - How to use: `python cli.py` (recommended) or `python gui.py` (if it has its own `__main__` later).

- [tui.py](file:///c:/Users/dell/Desktop/proxy-api/tui.py)
  - Purpose: Textual terminal UI for scanning models and exporting LM Studio presets.
  - How to use: `python cli.py --tui`.

## Non-code / Supporting Files

- [requirements.txt](file:///c:/Users/dell/Desktop/proxy-api/requirements.txt)
  - Purpose: Python dependencies for proxy + wizard tools.
  - Risk: currently contains duplicate/conflicting `nvidia-ml-py` entries and is missing some runtime deps used by the code (see next section).

- [README.md](file:///c:/Users/dell/Desktop/proxy-api/README.md)
  - Purpose: high-level description + quick start.

- [plan.md](file:///c:/Users/dell/Desktop/proxy-api/plan.md)
  - Purpose: implementation plan + assumptions + historical notes about fixes. Not required at runtime.

- [issues.md](file:///c:/Users/dell/Desktop/proxy-api/issues.md)
  - Purpose: captured logs / debugging notes. Not required at runtime.

- [.gitignore](file:///c:/Users/dell/Desktop/proxy-api/.gitignore)
  - Purpose: ignore patterns for Python caches/venvs/logs/editor settings.

- [templates/lmstudio_config_template.yaml](file:///c:/Users/dell/Desktop/proxy-api/templates/lmstudio_config_template.yaml)
  - Purpose: template for an LM Studio configuration YAML (looks like Jinja-style placeholders). This is only relevant if you have a script that renders it (none of the current main paths render this template).

- [output/lmstudio_config.yaml](file:///c:/Users/dell/Desktop/proxy-api/output/lmstudio_config.yaml)
  - Purpose: example/exported config values (wizard output artifact). Not required at runtime.

## Project-impacting Issues / Risks To Watch

These are the kinds of problems that most often “break the proxy” or cause surprising behavior compared to calling LM Studio directly:

- **Missing endpoint parity**: benchmark/docs sometimes assume `GET /v1/models`, but the proxy only implements `GET /api/v1/models` currently.
- **Dependency gaps**: [embedder.py](file:///c:/Users/dell/Desktop/proxy-api/embedder.py) imports `numpy`, but `numpy` is not listed in [requirements.txt](file:///c:/Users/dell/Desktop/proxy-api/requirements.txt). Any path that instantiates/uses `Embedder` can fail at runtime on a clean machine.
- **Dependency conflicts**: [requirements.txt](file:///c:/Users/dell/Desktop/proxy-api/requirements.txt) declares `nvidia-ml-py` twice with different version constraints; pip resolution can become unpredictable.
- **Hidden runtime bug (Maxwell path)**: [vram_calculator.py](file:///c:/Users/dell/Desktop/proxy-api/vram_calculator.py) references `logger.debug(...)` but does not define/import `logger`. This only triggers on certain GPUs (`cuda_compute < 6.0`).
- **Hardcoded LM Studio base**: [mcp_server.py](file:///c:/Users/dell/Desktop/proxy-api/mcp_server.py) uses a hardcoded `LMSTUDIO_BASE` constant instead of `LMSTUDIO_BASE_URL`, so it can silently point at the wrong server.
- **“Dedup” middleware not enforcing**: request de-duplication logic exists in [proxy.py](file:///c:/Users/dell/Desktop/proxy-api/proxy.py) but it currently does not short-circuit the request pipeline, so it does not reduce load.
- **Prompt injection side-effects**: domain routing can add/modify system prompts; this can reduce JSON validity or tool calling depending on the model. Customize via [custom_prompts.json](file:///c:/Users/dell/Desktop/proxy-api/custom_prompts.json).

## What You Can Drop (Common Minimal Setup)

If the goal is “only run the proxy server” (no wizard UI, no preset export), you typically keep:

- `proxy.py`
- `cache_manager.py`
- `context_manager.py`
- `domain_router.py`
- `custom_prompts.json` (optional, only if you want custom prompts)
- `hardware_detector.py` and `models.py` (required for `/api/v1/hardware` and preset generation endpoint)
- `tuner.py`
- `reranker.py` (only needed if you use `extra_body.context_docs` injection)
- `requirements.txt`

Candidates to drop if you do not need them:

- Wizard UI: `cli.py`, `gui.py`, `tui.py`
- Model discovery/export tooling: `model_discovery.py`, `exporters.py`, `export_utils.py`, `model_profile.py`, `recommender.py`, `hardware_utils.py`
- Benchmark tooling: `benchmark.py` (but then remove the benchmark endpoints/import from `proxy.py`)
- Templates/output: `templates/`, `output/` (only used by the wizard/config export workflows)
- Planning notes: `plan.md`, `issues.md` (documentation only, not runtime code)

## Notes / Known Limitations (Proxy)

- The proxy forwards most request fields to LM Studio, but also mutates some fields (like `max_tokens` and `temperature`) based on detected domain and optional `mode` values.
- For streaming requests, the proxy normalizes some SSE event variants into an OpenAI-like `data: { "choices": [ { "delta": ... } ] }` shape.
- For non-streaming requests, the proxy formats plain text output for some domains, but it intentionally does not modify `tool_calls` or structured output payloads.
