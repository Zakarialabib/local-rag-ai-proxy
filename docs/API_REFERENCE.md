# 📚 LM Studio Proxy API Reference (Port 8080)


All endpoints—both OpenAI-compatible and custom—are accessible via the FastAPI proxy at `127.0.0.1:8080`.

---

## 🟢 Dashboard API Coverage (All Implemented)

The web dashboard provides direct UI access to all major API endpoints:

- `/api/v1/retrieve` and `/api/v1/rerank` (Retrieval & Rerank tab)
- `/v1/embeddings` and `/api/v1/embed` (Embeddings tab)
- `/v1/agent/sessions` and related (Agent Orchestration tab)
- `/v1/ace/generate` and trace (ACE Context tab)
- `/v1/chat/completions` (Benchmark/Diagnostics tab)

All features are fully implemented and can be tested interactively from the dashboard.

---

## 🏛️ Models

### `GET /v1/models`
Returns all models currently loaded in LM Studio in the standard OpenAI format.

### `GET /api/v1/model-stats`
Custom endpoint to retrieve VRAM usage and loading state—useful for dashboard health monitoring.

### `POST /api/v1/models/load`
Manually load a model by its identifier. Supports `context_length`, `ttl`, and `gpu` configurations.

---

## 💬 Chat & Completions

### `POST /v1/chat/completions` (Recommended)
The primary endpoint for all chat interactions.
*   **Streaming**: Full support with standard OpenAI metadata (`id`, `model`, etc.).
*   **Reasoning**: Qwen 3.5 4B-specific reasoning chunks are mapped to `choices[0].delta.reasoning_content`.
*   **Caching**: Automatic LRU caching of responses to reduce repeat-latency.

### `POST /v1/responses` (Optimized Batch)
A specialized endpoint for instruction-heavy prompts.
*   **Parameters**: `instructions`, `messages`, `max_output_tokens`, `temperature`.
*   **Note**: ❌ **Streaming is NOT currently implemented** for this endpoint.

---

## 🏗️ Retrieval System

### `POST /api/v1/retrieve`
The core two-stage retrieval pipeline. Optimized for matching queries against local document sets.

```json
{
  "query": "How to sort in Python?",
  "docs": ["path/to/docs/file1.md", "README.md"],
  "retrieval": {
    "top_k": 3,
    "chunk_size": 900,
    "chunk_overlap": 150,
    "max_chunks": 32
  }
}
```

*   **Stage 1 (Embedding)**: Fast semantic search using Qwen 4B (optimized with MRL).
*   **Stage 2 (Reranking)**: Precision scoring with Qwen 4B Reranker.

### `POST /api/v1/rerank`
Manual reranking of provided text chunks.
*   **Parameters**: `query`, `chunks` (list of strings or dicts), `top_k`.

---

## 🧬 Embeddings

### `POST /v1/embeddings`
OpenAI-compatible embedding generation. Supports single strings or lists of strings.

### `POST /api/v1/embed` (Performance Hook)
Returns raw embedding vectors without the OpenAI wrapper. Ideal for high-throughput batching.

---

## 🏎️ Hardware & Tuning

### `GET /api/v1/hardware`
Returns detected GPU/CPU stats, CUDA version, and VRAM availability.

### `POST /api/v1/presets/generate`
Generates an optimized LM Studio `.preset.json` file specifically for your hardware profile.

---

## 🤖 Orchestration (v1/agent)

### `POST /v1/agent/sessions`
Initializes a multi-turn agent session with a specific `workflow` (e.g., `coding_sprint`).

### `POST /v1/agent/sessions/{id}/turn`
Executes an agent turn using thinking-aware orchestration. Supports tool calls.

---

## 🛸 ACE (Active Context Engineering)

### `POST /v1/ace/generate`
Streams context-enriched completions.
*   **Headers**: `X-ACE-Version: v3-active-context`.
*   **Events**: `context_injected`, `thought_detected`, `generation`.

---

## 🎯 Phase 6: Perfection Path API (Autonomous Optimization)

All Phase 6 endpoints are available at `http://127.0.0.1:5000/api/perfection/`.

### 1. Constraint Probing (Capability Envelope)

#### `GET /api/perfection/probing/envelope`
Returns hardware capability envelope (context vs TTFT curves).

**Response**: Safe zone, cliff edge, degradation curve, trend analysis.

#### `POST /api/perfection/probing/trigger`
Force immediate probing run across context levels.

---

### 2. Model Sharding (Residency Orchestration)

#### `GET /api/perfection/sharding/residency`
Current model residency breakdown (VRAM, SystemRAM, Disk).

**Response**: Model locations, VRAM usage, free space, predicted next task.

#### `POST /api/perfection/sharding/evict`
Manually evict model to free VRAM for remediation.

---

### 3. Pre-warming (Cold-Start Mitigation)

#### `GET /api/perfection/prewarming/metrics`
Cold/hot TTFT metrics and idle time prediction.

#### `POST /api/perfection/prewarming/trigger`
Manually trigger pre-warming cycle.

---

### 4. Truncation Monitoring (Output Quality)

#### `GET /api/perfection/truncation/patterns`
Truncation rate by task type with remediation recommendations.

#### `POST /api/perfection/truncation/fix`
Apply remediation (e.g., increase max_tokens) for truncation pattern.

---

### 5. Reranker Dilemma (0.6B vs 4B)

#### `GET /api/perfection/reranker/stats`
0.6B vs 4B usage statistics, confidence thresholds, swap history.

#### `POST /api/perfection/reranker/swap`
Manually trigger 4B reranker swap for high-confidence queries.

---

### 6. VRAM Management (Defragmentation)

#### `GET /api/perfection/vram/fragmentation`
Fragmentation status, scheduled grooming, recovery history.

#### `POST /api/perfection/vram/groom-now`
Trigger immediate memory grooming.

---

### 7. Preset Evolution (Learning)

#### `GET /api/perfection/presets/lineage`
Preset mutation history (family tree) with performance deltas.

#### `POST /api/perfection/presets/rollback`
Rollback to previous preset variant.

#### `POST /api/perfection/presets/mutate`
Trigger preset mutation experiment.

---

### 8. Streaming Mode (Hybrid Routing)

#### `GET /api/perfection/streaming/mode-distribution`
Streaming vs batch-and-stream distribution by task type.

#### `POST /api/perfection/streaming/switch-mode`
Force streaming mode switch.

---

### 9. Vision Capability (Masking)

#### `GET /api/perfection/vision/capability-status`
Vision capability declared vs runtime (with VRAM constraint analysis).

#### `POST /api/perfection/vision/enable`
Enable vision by evicting non-critical models.

---

### 10. Resilience & Fallback Chains

#### `GET /api/perfection/resilience/status`
Current resilience mode, fallback chain state, activation history.

#### `POST /api/perfection/resilience/activate-level`
Manually activate fallback level for testing or recovery.

#### `POST /api/perfection/resilience/reset`
Reset to ideal mode if recovery successful.

---

**See also**: [Technical Details](TECHNICAL_DETAILS.md) for Phase 6 architecture, [Dashboard Integration](DASHBOARD_INTEGRATION.md) for UI visualization panels.

---

**Next Step**: Read the [Technical Details](TECHNICAL_DETAILS.md) for architecture and Qwen 4B tuning specs.
