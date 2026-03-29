# Tool Ecosystem: Monitoring, Analytics & Resilience

## Overview

A comprehensive tool monitoring, health tracking, and automated remediation system for the proxy-api agent framework. Provides real-time visibility into tool behavior, automatic anomaly detection, and intelligent remediation actions.

**Phase 6 Integration**: Remediation engine now orchestrates Perfection Path fallback chains, VRAM management, and autonomous optimization.

## Features

### 1. **Tool Health Monitoring** 🏥
- Real-time tool status tracking (Healthy/Degraded/Isolated/Failed)
- Error rate calculation per tool
- Circuit breaker isolation (auto-isolation after 5 consecutive failures)
- Health timeline with historical events
- Cluster-based health assessment

### 2. **Perfection Index** 📊
- **Global Index**: Overall system quality (0.0-1.0 scale)
- **Quality Score**: Execution success weighted by velocity (0-100)
- **Reliability Index**: Success vs. error rate differential
- **Velocity Metrics**: Calls per minute tracking
- **Per-Tool Metrics**: Individual tool performance tracking

### 3. **Anomaly Detection** 🔍
- **High Error Rate Detection**: Alert when error rate > 30%
- **Slow Execution Detection**: Alert when avg latency > 500ms
- **Sudden Spike Detection**: Alert when recent avg > 2.5x historical avg
- **Degradation Tracking**: Trend analysis over time
- **Severity Scoring**: 0.0-1.0 severity scale

### 4. **Remediation Engine** 🛠️ (Phase 6 Enhanced)
- **Automatic Actions**: Triggered on anomalies detected
- **Action Types**:
  - `RESTART_TOOL`: Restart failed tool
  - `THROTTLE_CALLS`: Reduce call rate
  - `INCREASE_TIMEOUT`: Extend execution timeout
  - `ISOLATE_TOOL`: Circuit breaker isolation
  - `RESET_CIRCUIT`: Manual circuit breaker reset
  - `LOG_ISSUE`: Detailed logging
  - `ALERT_OPERATOR`: Send alert
  - `FALLBACK_STRATEGY`: Use alternative tool
  - **NEW:** `FALLBACK_CHAIN_NEXT`: Activate next level in fallback chain
  - **NEW:** `EVICT_MODELS`: Free VRAM by unloading non-critical models
  - **NEW:** `TRIGGER_VRAM_GROOM`: Schedule or execute memory defragmentation
  - **NEW:** `REDUCE_QUANTIZATION`: Downgrade model quantization level
  - **NEW:** `PRESET_MUTATION`: Auto-apply evolved preset variant
- **Callback Registration**: Custom handlers per action
- **Remediation History**: Audit trail of all actions

### 5. **Rate Limiter** ⏱️
- Token bucket algorithm
- Configurable rate (default: 20 tokens/minute)
- Automatic token refill
- Per-tool rate limiting

### 6. **ACID Audit Trail** 📝
- Complete session history
- Event logging for all tool calls
- Tool execution lifecycle tracking
- Error message storage
- Session metadata preservation

### 7. **Dashboard UI** 📈
- Real-time health widget
- Perfection index visualization
- Anomaly detection display
- Remediation actions log
- Health timeline with events
- Auto-refresh (5-second interval)
- **NEW:** Perfection Path visualization (10 autonomous optimization panels)

## API Endpoints

### Health & Status
```
GET /api/tools/health
  → Current tool health status, error rates, circuit breaker states

GET /api/tools/health/timeline
  → Historical health events (last 100)

GET /api/tools/perfection
  → Perfection index, quality scores, reliability metrics
```

### Analytics
```
POST /api/tools/analytics/detect-anomalies
  → Run anomaly detection on all/specific tool
  → Body: { "tool_name": "optional_tool" }

GET /api/tools/analytics/<tool_name>
  → Tool-specific history (lifecycle, anomalies, remediation)

GET /api/tools/remediation/pending
  → Get pending remediation actions (last 50)
```

### Perfection Index Management
```
POST /api/tools/perfection/reset
  → Reset all metrics and indices
```

### ACID Audit
```
GET /api/acid/sessions
  → List ACID sessions (last 50)

GET /api/acid/sessions/<session_id>/timeline
  → Session event timeline
```

## Usage Examples

### Python Integration

```python
from webapp.tool_analytics import anomaly_detector, remediation_engine, RemediationAction
from webapp.perfection_index import perfection_tracker

# Record tool execution
perfection_tracker.record_tool_execution(
    tool_name="my_tool",
    latency_ms=45.2,
    success=True,
    session_id="sess_123",
    agent="agent_123"
)

# Detect anomalies
anomalies = anomaly_detector.detect_anomalies("my_tool")

# Register remediation handler
def handle_isolate(tool_name):
    print(f"Isolating {tool_name}")

remediation_engine.register_action_callback(
    RemediationAction.ISOLATE_TOOL,
    handle_isolate
)

# Trigger remediation
actions = remediation_engine.trigger_remediation("my_tool", anomalies)
```

### API Integration

```javascript
// JavaScript/Frontend Integration
async function loadHealthStatus() {
    const response = await fetch('/api/tools/health');
    const data = await response.json();
    console.log(`System health: ${data.overall_health * 100}%`);
}

// Detect anomalies on-demand
async function detectAnomalies() {
    const response = await fetch('/api/tools/analytics/detect-anomalies', {
        method: 'POST',
        body: JSON.stringify({})
    });
    const data = await response.json();
    console.log(`Anomalies found: ${data.anomalies_detected}`);
}

// Auto-refresh health every 5 seconds
setInterval(loadHealthStatus, 5000);
```

## Database Files

- **ACID Audit**: `.gui_state/acid_sessions.db`
- **Tool Analytics**: `webapp/tool_analytics.db`
- **Perfection Index**: `webapp/perfection_index.db`

## Configuration

### Tool Health Monitor
- Failure threshold: 5 consecutive failures trigger circuit breaker
- Circuit break timeout: 60 seconds (auto-recovery)
- Isolation event history: Last 100 events

### Rate Limiter
- Default rate: 20 tokens/minute (configurable)
- Token refill: Continuous (1 token per 3 seconds)
- Maximum tokens: Equals configured rate

### Anomaly Detector
- Error rate threshold: 30%+
- Slow execution threshold: 500ms avg
- Spike detection: 2.5x recent vs historical

## Performance Characteristics

- **Memory**: ~5-10MB (in-memory history limited to 100 per tool)
- **Disk**: SQLite databases grow ~1MB per 10k events
- **CPU**: <1% overhead for monitoring
- **Response time**: Analytics queries <50ms

## Integration Points

### Agent Tools
```python
# In shared/agent_tools.py AgentTool.execute():
from webapp.tool_analytics import anomaly_detector, analytics_store
from webapp.perfection_index import perfection_tracker

start_time = time.time()
try:
    result = self.fn(*args, **kwargs)
    duration = (time.time() - start_time) * 1000
    perfection_tracker.record_tool_execution(
        self.name, duration, True, session_id=session_id
    )
except Exception as e:
    duration = (time.time() - start_time) * 1000
    perfection_tracker.record_tool_execution(
        self.name, duration, False, error=str(e), session_id=session_id
    )
    analytics_store.log_lifecycle(self.name, "execute", duration, False, str(e))
    raise
```

## Testing

Run comprehensive tests:
```bash
# Individual system tests
python test_health.py                 # Health monitoring
python test_perfection.py             # Perfection metrics
python test_circuit_breaker.py        # Circuit breaker
python test_rate_limiter.py           # Rate limiting
python test_analytics.py              # Anomaly detection
python test_acid.py                   # ACID persistence

# Full integration test
python test_full_integration.py       # All systems together
```

---

## Phase 6: Perfection Path Integration with Remediation Engine

The remediation engine now orchestrates all 10 Perfection Path concepts via enhanced action callbacks.

### 1. Constraint Probing + Trend Analysis
**Remediation Action**: `ALERT_CLIFF_EDGE`
```python
# When TrendAnalyzer detects TTFT spike >300% at context 8192:
remediation_engine.trigger_remediation("probing", {
    "anomaly_type": "cliff_edge_detected",
    "context_level": 8192,
    "ttft_spike": 5.2,
    "safe_envelope": "0-4096"
})
# Action: Alert severity 0.7 (HIGH), recommend context reduction
```

### 2. Model Sharding + Domain Routing
**Remediation Action**: `FALLBACK_CHAIN_NEXT`
```python
# When code generation hits OOM (Level 1 ideal mode fails):
remediation_engine.trigger_remediation("model_sharding", {
    "anomaly_type": "oom_detected",
    "current_level": 1,
    "reason": "Full stack OOM"
})
# Action: Move to Level 2 (Qwen3.5-4B Q3_K_M), unload reranker
# Result: GUI shows 🟡 YELLOW (Pressure), operations continue
```

### 3. Pre-warming + Runtime Prediction
**Remediation Action**: `ALERT_COLD_START`
```python
# When time_since_generation > 30s:
remediation_engine.trigger_remediation("prewarming", {
    "anomaly_type": "cold_model_predicted",
    "idle_duration": 45,
    "recommendation": "trigger_prewarm"
})
# Action: Signal GUI to pre-warm before user task, separates TTFT metrics
```

### 4. Truncation Handling + Context Management
**Remediation Action**: `INCREASE_MAX_TOKENS`
```python
# When analytics detects code task truncating >40% of time:
remediation_engine.trigger_remediation("truncation", {
    "anomaly_type": "truncation_rate_high",
    "task_type": "code_generation",
    "current_max": 512,
    "suggested_max": 1024
})
# Action: Auto-apply higher max_tokens for code domain
# Result: Subsequent code tasks use max_tokens: 1024
```

### 5. Reranker Dilemma + Confidence Routing
**Remediation Action**: `DEEP_RERANKER_SWAP`
```python
# When 0.6B reranker confidence < 0.7 on document set:
remediation_engine.trigger_remediation("reranker", {
    "anomaly_type": "low_confidence",
    "confidence_score": 0.65,
    "action": "swap_to_4b",
    "cost": "5s_latency",
    "benefit": "+0.14_confidence"
})
# Action: Unload embed, load 4B reranker, re-rank high-value subset
# Result: Critical documents get better ranking (swap cost justified)
```

### 6. VRAM Management + Fragmentation
**Remediation Action**: `TRIGGER_VRAM_GROOM`
```python
# When available VRAM gap > 500MB detected:
remediation_engine.trigger_remediation("vram_management", {
    "anomaly_type": "fragmentation_detected",
    "reported_vram": 1200,
    "allocatable_vram": 700,
    "loss_mb": 500,
    "schedule": "next_idle_window"
})
# Action: Schedule/execute sequential model unload-reload
# Result: Recover fragmented space, GUI shows grooming progress
```

### 7. Preset Evolution + A/B Testing
**Remediation Action**: `PRESET_MUTATION`
```python
# When TrendAnalyzer detects regression (TTFT +15%):
remediation_engine.trigger_remediation("presets", {
    "anomaly_type": "regression_detected",
    "baseline_ttft": 18,
    "current_ttft": 20.7,
    "suggested_mutation": {"context_length": 2048 -> 1536}
})
# Action: Auto-apply variant preset during idle window, benchmark
# Result: If improvement confirmed, new preset becomes default
```

### 8. Streaming Mode + TPS Threshold
**Remediation Action**: `SWITCH_STREAMING_MODE`
```python
# When code task TPS < 10/s during first 10 tokens:
remediation_engine.trigger_remediation("streaming", {
    "anomaly_type": "tps_low",
    "measured_tps": 7.2,
    "threshold": 10,
    "action": "switch_to_batch_stream"
})
# Action: Batch completion in background, then rapid stream
# Result: Hides 26s latency, preserves IDE UX feeling
```

### 9. Vision Masking + Capability Detection
**Remediation Action**: `MASK_VISION_CAPABILITY`
```python
# When free_vram < 2GB after model load:
remediation_engine.trigger_remediation("vision", {
    "anomaly_type": "insufficient_vram_for_vision",
    "free_vram": 0.8,
    "required": 2,
    "action": "mask_capability"
})
# Action: Hide vision capability in API, log constraint
# Result: Prevents "works in probe, fails in prod" confusion
```

### 10. Autonomous Fallback Chains
**Remediation Action**: `FALLBACK_CHAIN_NEXT` + `REDUCE_QUANTIZATION` + `EVICT_MODELS`
```python
# Cascading fallback on failures:
Level 1 (IDEAL) - OOM?
  → Remediation: evict reranker, reduce to Q3_K_M
     → Level 2 (PRESSURE) - Still fails?
        → Load Lfm2.5-1.2B instead
           → Level 3 (EMERGENCY) - Still fails?
              → Keep embed+rerank, no generation
                 → Level 4 (RETRIEVAL) - Still fails?
                    → Circuit break (Level 5)

# Each level automatically triggers next if execution fails:
remediation_engine.trigger_remediation("fallback", {
    "anomaly_type": "execution_failure",
    "current_level": 1,
    "execution_error": "CUDA OOM",
    "action": "next_level"
})
# Result: GUI Resilience Indicator updates 🟢 → 🟡 → 🟠 → 🔴 → ⚫
```

---

### Enhanced Callback Registry

New callbacks added to `remediation_callbacks.py`:

```python
class RemediationCallbackRegistry:
    
    def __init__(self):
        # Existing callbacks
        self.handlers[RemediationAction.ISOLATE_TOOL] = self.handle_isolate
        self.handlers[RemediationAction.RESET_CIRCUIT] = self.handle_reset
        ...
        
        # NEW Phase 6 callbacks
        self.handlers[RemediationAction.FALLBACK_CHAIN_NEXT] = self.handle_fallback_next
        self.handlers[RemediationAction.EVICT_MODELS] = self.handle_model_eviction
        self.handlers[RemediationAction.TRIGGER_VRAM_GROOM] = self.handle_vram_groom
        self.handlers[RemediationAction.REDUCE_QUANTIZATION] = self.handle_quantization_reduce
        self.handlers[RemediationAction.PRESET_MUTATION] = self.handle_preset_mutation
        self.handlers[RemediationAction.ALERT_CLIFF_EDGE] = self.handle_cliff_edge
        self.handlers[RemediationAction.ALERT_COLD_START] = self.handle_cold_start
        self.handlers[RemediationAction.DEEP_RERANKER_SWAP] = self.handle_reranker_swap
        self.handlers[RemediationAction.MASK_VISION_CAPABILITY] = self.handle_vision_mask
        self.handlers[RemediationAction.SWITCH_STREAMING_MODE] = self.handle_streaming_switch

    def handle_fallback_next(self, tool_name, details):
        """Execute next level in fallback chain"""
        # Implementation in webapp/remediation_callbacks.py
        pass

    def handle_model_eviction(self, tool_name, details):
        """Free VRAM by unloading non-critical models"""
        pass

    def handle_vram_groom(self, tool_name, details):
        """Schedule/execute memory defragmentation"""
        pass

    # ... other Phase 6 handlers ...
```

---

### Integration Flow (Phase 6)

```
Anomaly Detected
    ↓
TrendAnalyzer / AnomalyDetector
    ├─ Constraint cliff-edge?
    ├─ Model sharding failure?
    ├─ Cold-start TTFT?
    ├─ Truncation rate high?
    ├─ Reranker confidence low?
    ├─ VRAM fragmentation?
    ├─ Preset regression?
    ├─ Streaming TPS low?
    ├─ Vision insufficient VRAM?
    └─ Tool execution OOM?
         ↓
RemediationEngine decides action
    │
    ├─ FALLBACK_CHAIN_NEXT → Execute fallback level
    ├─ EVICT_MODELS → Signal GUI to unload models
    ├─ TRIGGER_VRAM_GROOM → Schedule defragmentation
    ├─ REDUCE_QUANTIZATION → Downgrade quantization
    ├─ PRESET_MUTATION → Apply evolved preset
    ├─ ALERT_CLIFF_EDGE → Send severity HIGH alert
    ├─ ALERT_COLD_START → Trigger pre-warm
    ├─ DEEP_RERANKER_SWAP → Swap to 4B reranker
    ├─ MASK_VISION_CAPABILITY → Hide vision in API
    └─ SWITCH_STREAMING_MODE → Toggle batch-and-stream
         ↓
RemediationCallbackRegistry
    │
    ├─ Execute callback handler
    ├─ Log action to ACID audit
    ├─ Update circuit breaker state
    ├─ Store remediation event in analytics
    └─ Broadcast status update via SSE
         ↓
GUI receives update
    │
    ├─ Resilience Indicator updates
    ├─ VRAM Tetris refreshes
    ├─ Streaming Mode changes
    ├─ Presets update
    ├─ Alerts displayed
    └─ User sees smooth operation despite degradation
```

---

### Testing Phase 6 Integration

```bash
# Run comprehensive integration test
python test_full_integration.py

# Specific Phase 6 tests (to be created)
python test_perfection_path.py          # End-to-end perfection path
python test_fallback_chains.py          # Fallback escalation
python test_vram_management.py          # Fragmentation + grooming
python test_preset_evolution.py         # Mutation + A/B testing
```

---

## Future Enhancements

1. **Machine Learning**: Predictive anomaly detection
2. **Alerting**: Email/Slack notifications for severity
3. **Dashboards**: Grafana/Kibana integration
4. **Historical Analysis**: Generate regression reports
5. **Custom Rules**: User-defined anomaly thresholds
6. **Tool Clustering**: Group similar tools
7. **Capacity Planning**: Resource usage predictions
8. **Perfection Path AutoML**: Learn optimal parameter boundaries per hardware
9. **Multi-GPU Support**: Extend model sharding to multi-card setups
10. **Adaptive Quantization**: Auto-select quantization based on workload patterns

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Web Server                      │
│                    (33+ endpoints)                       │
└────────┬────────────────────────┬───────────────────────┘
         │                        │
    ┌────▼──────┐          ┌─────▼────────┐
    │  API      │          │  Dashboard   │
    │  Routes   │          │  (HTML/CSS)  │
    └────┬──────┘          └──────────────┘
         │
┌────────▼──────────────────────────────────┐
│          Monitoring Core System            │
├──────────────────────────────────────────┤
│  Tool Health Monitor                      │
│  ├─ CircuitBreaker                        │
│  ├─ RateLimiter                           │
│  └─ Status Tracking                       │
│                                            │
│  Perfection Index                         │
│  ├─ Metrics Recording                     │
│  ├─ Index Calculation                     │
│  └─ Trend Analysis                        │
│                                            │
│  Analytics Engine                         │
│  ├─ Anomaly Detection                     │
│  ├─ Remediation Engine                    │
│  └─ Action Callbacks                      │
└────────┬──────────────────────────────────┘
         │
┌────────▼──────────────────────────────────┐
│         Persistent Storage (SQLite)        │
├──────────────────────────────────────────┤
│  acid_sessions.db      (Audit Trail)     │
│  tool_analytics.db     (Lifecycle/Anom)  │
│  perfection_index.db   (Metrics/Index)   │
└──────────────────────────────────────────┘
```

## Support

For issues or feature requests, refer to the main README or contact the development team.
