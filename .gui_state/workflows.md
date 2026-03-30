# GUI Workflow Map

- Event: `probe_result`
- Selected model: `liquid/lfm2.5-1.2b`
- Main role: `qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2`
- Embed role: `text-embedding-qwen3-embedding-4b`
- Rerank role: `qwen3-reranker-0.6b`
- Profile status: `predicted`
- Mode: `fast`
- Probe mode: `quick`
- Timeout policy: `fast_short`
- Request counters: `{"models_refresh": 2, "probe_quick": 1, "runtime_health_refresh": 4}`
- Retrieval sources enabled: `0`

## Workflows

- `Runtime`: bridge health, role loading, loaded-model reuse.
- `Responses`: OpenAI-style responses request builder and result viewer.
- `Chat`: compatibility path for `chat.completions`.
- `Retrieval`: source registration, chunk preview, rerank preview, injected context preview.
- `Benchmark`: connectivity, model list, probe, smoke tests.
- `Profile`: hardware-aware prediction, preset load/export, env preview.
