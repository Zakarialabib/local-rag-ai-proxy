# 🚀 Getting Started with LM Studio Proxy

Welcome! This guide provides everything you need to set up, launch, and verify the LM Studio Proxy API—a high-performance, OpenAI-compatible bridge with advanced retrieval and Qwen 3.5 4B optimizations.

---

## 🛠️ Step 1: Environment Setup

Ensure you have your environment variables configured in a `.env` file in the project root.

```powershell
# Core Settings
LMSTUDIO_BASE_URL="http://192.168.1.12:1234"  # Where LM Studio runs
BRIDGE_HOST="127.0.0.1"                      # Proxy server host
BRIDGE_PORT="8080"                           # Proxy server port

# Default Models (optimized for Qwen 4B stack)
MAIN_MODEL="qwen3.5-4b"
EMBED_MODEL="text-embedding-qwen3-embedding-4b"
RERANK_MODEL="qwen.qwen3-reranker-4b"
```

> [!TIP]
> Use `127.0.0.1:8080` for the proxy and `192.168.1.12:1234` (or your local IP) for LM Studio to avoid port conflicts.

---

## ⚡ Step 2: Launch the Services

### 1. Start the Proxy Server
In your first terminal:
```bash
python proxy.py
```
Wait for: `INFO: Uvicorn running on http://127.0.0.1:8080`

### 2. Start the Glassmorphic Dashboard
In a second terminal:
```bash
python webapp/app.py
```

Then navigate to: **[http://127.0.0.1:8090](http://127.0.0.1:8090)**

---

## 🚦 Dashboard Features (All Implemented)

The Glassmorphic Dashboard now provides a complete UI for:

- **Retrieval & Rerank**: Test document retrieval and reranking endpoints interactively.
- **Embeddings**: Generate and inspect embeddings from any model.
- **Agent Orchestration**: Create, branch, and manage agent sessions and turns.
- **ACE Context**: Generate and trace Active Context Engineering sessions.
- **Benchmark/Diagnostics**: Run pregenerated input tests and view results.

All features are fully implemented and available in the sidebar. Results, logs, and test flows are interactive and persist in the UI.

---

## 🧪 Step 3: Verify Connections

Run these quick tests to ensure all endpoints are live and compatible.

### Check Model List (OpenAI Format)
```bash
curl http://127.0.0.1:8080/v1/models | jq '.data[].id'
```

### Test Embeddings (Qwen 4B)
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-qwen3-embedding-4b", "input": "Hello Qwen!"}'
```

### Test Chat Completion (Streaming)
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [{"role": "user", "content": "How fast are you?"}],
    "stream": true
  }'
```

---

## 🔌 Integration Examples

### Official OpenAI SDK (Python)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none"
)

# Chat with Qwen 4B
response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Hi!"}]
)
print(response.choices[0].message.content)
```

### LangChain
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:8080/v1",
    api_key="none",
    model="text-embedding-qwen3-embedding-4b"
)
```

---

## ⭐ Step 3b: Phase 6 Perfection Path Configuration

Enable autonomous optimization for 8GB VRAM constraints and 26s TTFT:

### 1. Set Environment Variables for Phase 6

```powershell
# In your .env file, add Phase 6 settings:

# Perfection Path: Constraint Probing
ENABLE_PROBING="true"
PROBING_CONTEXT_LEVELS="512,1024,2048,4096,8192"

# Perfection Path: Model Sharding
ENABLE_MODEL_SHARDING="true"
MAX_RESIDENT_VRAM="6.5"  # GB (reserve 1.5GB for CUDA overhead)
DOMAIN_AWARE_SCHEDULING="true"

# Perfection Path: Pre-warming
ENABLE_PREWARMING="true"
IDLE_THRESHOLD="30"  # seconds before predicting model eviction
WARM_TIMEOUT="5"  # seconds allowed for model to warm

# Perfection Path: Truncation Handling
ENABLE_TRUNCATION_MONITOR="true"
TRUNCATION_ALERT_THRESHOLD="0.4"  # Alert if >40% of tasks truncate

# Perfection Path: Reranker Swapping
ENABLE_CONFIDENCE_ROUTING="true"
RERANKER_CONFIDENCE_THRESHOLD="0.7"  # Swap to 4B if <0.7

# Perfection Path: VRAM Management
ENABLE_VRAM_GROOMING="true"
FRAGMENTATION_THRESHOLD="500"  # MB loss before triggering groom
GROOM_IDLE_WINDOW="300"  # Schedule during 5+ min idle

# Perfection Path: Preset Evolution
ENABLE_PRESET_EVOLUTION="true"
MUTATION_RATE="0.15"  # 15% degradation triggers mutation
MUTATION_CHECK_INTERVAL="86400"  # 24 hours

# Perfection Path: Streaming Hybrid
ENABLE_STREAMING_HYBRID="true"
TPS_THRESHOLD="10"  # True stream if >10 TPS, else batch-and-stream

# Perfection Path: Vision Masking
ENABLE_VISION_MASKING="true"
VISION_VRAM_REQUIRED="2.0"  # GB needed for image processing

# Perfection Path: Fallback Chains
ENABLE_FALLBACK_CHAINS="true"
FALLBACK_LEVELS="ideal,pressure,emergency,retrieval,circuit_break"
```

### 2. Initialize Phase 6 Databases

No manual initialization needed; databases auto-create on startup:

```bash
# Existing databases (Phases 1-5) auto-initialize:
# - tool_ecosystem.db
# - tool_analytics.db  
# - perfection_index.db
# - alerts.db

# Phase 6 additions (auto-create):
# - perfection_path.db (probing envelopes, preset lineage)
# - vram_events.db (fragmentation, grooming history)
```

### 3. Start Phase 6 Background Services

#### Start Proxy (Gateway)
```bash
python proxy.py
# Waits for: INFO: Uvicorn running on http://127.0.0.1:8080
```

#### Start Webapp (Main Dashboard + Phase 6)
```bash
python webapp/app.py
# Waits for: Running on http://127.0.0.1:5000

# Phase 6 services auto-start:
# - ProbeScheduler (constraint probing)
# - FluidOrchestrator (model sharding)
# - HardwareTuner (VRAM grooming + presets)
# - StreamingRouter (hybrid streaming)
```

#### Optional: Start GUI for Benchmarking
```powershell
# In PowerShell on Windows:
python gui.py

# Features available:
# - Constraint Probing (auto-runs on startup)
# - Benchmark with cold/hot TTFT separation
# - Truncation detection
# - Preset recommendations
```

### 4. Verify Phase 6 is Running

#### Check Dashboard Panels
Open **http://localhost:5000**:
1. Click **Management** → **Perfection Path**
2. You should see 10 new panels:
   - ⛰️ Capability Envelope
   - 🧩 Model Residency (VRAM Tetris)
   - 🔥 Pre-warming Status
   - 📄 Truncation-Aware Streaming
   - ♻️ Reranker Dilemma
   - 🧹 VRAM Grooming
   - 🔀 Preset Evolution (Lineage Tree)
   - 📡 Streaming Mode Decision
   - 👁️ Vision Capability Masking
   - 🛡️ Resilience Mode Indicator (top bar)

#### Check Phase 6 API Endpoints
```bash
# Test Perfection Path endpoints
curl http://127.0.0.1:5000/api/perfection/probing/envelope
# Expected: { "safe_zone": "0-4096", "cliff_edge": 8192, ... }

curl http://127.0.0.1:5000/api/perfection/sharding/residency
# Expected: { "models": [...], "total_vram": 6.2, "free": 1.8, ... }

curl http://127.0.0.1:5000/api/perfection/resilience/status
# Expected: { "mode": "ideal", "next_levels": [...], ... }
```

### 5. Monitor Phase 6 in Action

#### Real-Time Monitoring
```bash
# Tail logs for Phase 6 events
tail -f logs/perfection_path.log | grep -E "PROBING|SHARDING|PREWARMING|TRUNCATION|RERANKER|GROOMING|EVOLUTION|STREAMING|VISION|FALLBACK"

# Example outputs:
# 14:32 [PREWARMING] Cold model detected (45s idle). Triggering pre-warm...
# 14:33 [PROBING] Context 8192: TTFT 5.2s (cliff detected!)
# 14:34 [SHARDING] Code task detected. Keeping main+embed, evicting reranker.
# 14:35 [GROOMING] Fragmentation 520MB detected. Scheduling groom for 2:30 AM.
# 14:36 [STREAMING] TPS 7.2/s < 10. Switching to batch-and-stream mode.
```

#### Test Fallback Chains (Optional)

Force OOM to test fallback escalation:

```python
# In a test script:
from webapp.remediation_callbacks import remediation_callback_registry
from webapp.tool_analytics import remediation_engine

# Simulate OOM failure
remediation_engine.trigger_remediation("fallback", {
    "anomaly_type": "oom_detected",
    "current_level": 1,
    "models": ["Qwen3.5-4B (Q4_K_M)", "Embed4B", "Rerank0.6B"]
})

# Watch GUI:
# 1. 🟢 IDEAL → TRY unload reranker
# 2. 🟢 IDEAL (still fails) → 🟡 PRESSURE (reduce to Q3_K_M)
# 3. 🟡 PRESSURE (still fails) → 🟠 EMERGENCY (load Lfm2.5-1.2B)
# 4. 🟠 EMERGENCY (still fails) → 🔴 RETRIEVAL (no generation)
# 5. 🔴 RETRIEVAL (still fails) → ⚫ CIRCUIT_BREAK (manual restart)
```

---

## Phase 6 Feature Usage Examples

### Example 1: Auto-Tune for Code Generation

```bash
# GUI automatically detects code task pattern
# 1. Benchmark runs: "Generate Python function for sorting"
# 2. Result: truncation_rate 51%, max_tokens 512 insufficient
# 3. Webapp analyzes: Code tasks need higher token budgets
# 4. Remedy: Sets max_tokens 1024 for future code tasks
# 5. Next code task: completes without truncation ✓

# Dashboard shows:
# Truncation Monitor → Code: 51% → 0% (after fix)
# Remediation Actions: "INCREASE_MAX_TOKENS" (code) ✓
```

### Example 2: VRAM Fragmentation Recovery

```bash
# Windows WDDM causes memory leak over 8+ hours
# 1. Available VRAM (CUDA reports): 1.2GB
# 2. Actual allocatable: 0.7GB (500MB fragmented)
# 3. Webapp detects >500MB gap
# 4. Trigger grooming during idle window (no requests 5+ min)
# 5. Unload all models sequentially, reload in order
# 6. Result: Recover 480MB, available now 1.15GB ✓

# Dashboard shows:
# VRAM Grooming → Fragmentation: 520MB → Alert: Grooming scheduled
# System Maintenance modal during groom (30s)
# After: Fragmentation 40MB (recovered 480MB)
```

### Example 3: Preset Auto-Evolution

```bash
# System learns optimal context_length over 7 days
# Day 1: context: 2048 (baseline, TTFT 26s, TPS 7/s)
# Day 2: Benchmark regression +10% TTFT → Suggest context: 1536
# Day 3: Test mutation (context 1536): TTFT 18s, TPS 9/s ✓
# Day 4: Activate new preset (context 1536)
# Day 5: Continue monitoring, new baseline: 18s
# Day 6: Regression -8% TPS → Suggest batch_size increase
# Day 7: Test/validate, lineage tree shows evolution path

# Dashboard shows:
# Preset Evolution → Tree view with mutations:
#   Base (TTFT 26s)
#    ├─ Context↓ (TTFT 18s) ✓
#    ├─ Batch↑ (TTFT 16s, TPS 12/s) ✓
#    └─ Temp↓ (TTFT 20s) ✗
# User can "Roll back to Day 3" anytime
```

### Example 4: Cold-Start Elimination

```bash
# User requests benchmark after 45 minutes idle
# 1. Runtime detects: Model last used 45 min ago
# 2. Prediction: Model likely evicted to system RAM
# 3. Action: Send dummy inference ("Hello")
# 4. Measure: TTFT 26s (cold!)
# 5. Auto-warm loop: Re-send dummy until TTFT <2s
# 6. Result: Model warm, TTFT now 1.9s
# 7. Then real benchmark runs with hot metrics

# Dashboard shows:
# Pre-warming Status → Cold TTFT: 26s | Hot TTFT: 1.9s
# Pre-warming Log: "14:32 - Pre-warming triggered (45s idle). 26s → 1.9s ✓"
```

---

## 🔧 Troubleshooting Phase 6

### Issue: Perfection Path panels not showing

**Solution**: Restart webapp with Phase 6 enabled:
```bash
# Check .env has ENABLE_*=true flags
grep ENABLE_ .env | head -10

# Restart
python webapp/app.py  # Should see:
# INFO: Loading Phase 6 Perfection Path components
# INFO: Probing scheduler: ACTIVE
# INFO: Model sharding orchestrator: ACTIVE
# INFO: Hardware tuner (grooming + evolution): ACTIVE
```

### Issue: VRAM grooming not triggering

**Solution**: Manually trigger or adjust idle window:
```powershell
# Trigger immediately (for testing)
$response = curl -X POST http://127.0.0.1:5000/api/perfection/vram/groom-now

# Or adjust idle threshold in .env
GROOM_IDLE_WINDOW=60  # Groom after 1 minute idle (instead of 5 min)
```

### Issue: Preset mutations making things worse

**Solution**: Disable auto-apply, enable manual review:
```powershell
# In .env:
ENABLE_PRESET_EVOLUTION=true
AUTO_APPLY_MUTATIONS=false  # Require manual approval
NOTIFICATION_ON_MUTATION=true  # Alert user before applying
```

---

## 📊 Next Steps

1. **Benchmark System**: Run `python gui.py`, execute full benchmark → probe results show capability envelope
2. **Monitor Evolution**: Leave running for 7 days, watch preset mutations in Dashboard
3. **Test Fallback**: Deliberately crash a model level to observe fallback chain escalation
4. **Tune Thresholds**: Adjust PERFECTION_* env vars based on real system behavior

---

**Return to**: [API Reference](API_REFERENCE.md) | [Technical Details](TECHNICAL_DETAILS.md) | [Dashboard Integration](DASHBOARD_INTEGRATION.md)
*   **Streaming for `/v1/responses`**: Only non-streaming is currently supported.
*   **Persistent Agent Tools**: Tools must be registered manually via `register_tool()` in the current version.

### 502 Bad Gateway?
Check if LM Studio is actually running at the IP/Port specified in `LMSTUDIO_BASE_URL`.

### Permission Errors on Logs?
Windows may block log rotation if the `.log` file is open. Close any log viewers and restart `proxy.py`.

---

**Next Step**: Read the [API Reference](API_REFERENCE.md) for a full map of advanced retrieval and orchestration endpoints.
