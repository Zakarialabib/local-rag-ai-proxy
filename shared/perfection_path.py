# Phase 6: Perfection Path - Autonomous Optimization for 8GB VRAM + 26s TTFT
"""
Coordinates all 10 Perfection Path concepts:
1. Constraint Probing
2. Model Sharding
3. Pre-warming
4. Truncation Handling
5. Reranker Swapping
6. VRAM Management
7. Preset Evolution
8. Streaming Hybrid
9. Vision Masking
10. Fallback Chains
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import sqlite3
import os

# ============================================================================
# 1. CONSTRAINT PROBING - Capability Envelope Detection
# ============================================================================

@dataclass
class ProbeResult:
    """Results from a single context length probe"""
    context_length: int
    ttft_ms: float
    tps: float
    timestamp: float
    success: bool
    error: Optional[str] = None

@dataclass
class CapabilityEnvelope:
    """Hardware capability boundaries"""
    safe_zone_min: int = 512
    safe_zone_max: int = 4096
    cliff_edge: int = 8192
    degradation_rate: float = 1.05  # TTFT multiplier per 2K context
    trend: str = "stable"  # stable, shrinking, expanding
    last_probed: float = 0.0
    probes: List[ProbeResult] = field(default_factory=list)

class ProbeScheduler:
    """Runs constraint probing to detect VRAM pressure cliff edges"""
    
    def __init__(self, model_id: str, base_url: str = "http://127.0.0.1:8080"):
        self.model_id = model_id
        self.base_url = base_url
        self.envelope = CapabilityEnvelope()
        self.context_levels = [512, 1024, 2048, 4096, 8192]
        
    async def probe_context_levels(self) -> CapabilityEnvelope:
        """Gradually increase context until TTFT jumps >5s, identifying cliff edge"""
        import httpx
        
        print("[PROBING] Starting constraint probing...")
        
        for context_length in self.context_levels:
            prompt = "Explain quantum computing briefly." * (context_length // 256)
            
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt[:context_length]}],
                "temperature": 0.7,
                "max_tokens": 256,
                "stream": True
            }
            
            start = time.time()
            first_token_time = None
            token_count = 0
            
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/chat/completions", 
                                            json=payload, timeout=120) as resp:
                        if resp.status_code != 200:
                            continue
                        
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                
                                if "content" in delta and first_token_time is None:
                                    first_token_time = time.time()
                                if "content" in delta:
                                    token_count += 1
                            except:
                                pass
            except Exception as e:
                result = ProbeResult(context_length, 0, 0, time.time(), False, str(e))
                self.envelope.probes.append(result)
                continue
            
            ttft = (first_token_time - start) * 1000 if first_token_time else 0
            generation_time = (time.time() - first_token_time) if first_token_time else 0.001
            tps = token_count / generation_time if generation_time > 0 else 0
            
            result = ProbeResult(context_length, ttft, tps, time.time(), True)
            self.envelope.probes.append(result)
            
            print(f"[PROBING] Context {context_length}: TTFT {ttft:.0f}ms, TPS {tps:.1f}/s")
            
            # Detect cliff edge: TTFT jumps >5000ms (jump from <2s to 26s for cold model)
            if ttft > 5000 and context_length > self.envelope.safe_zone_max:
                self.envelope.cliff_edge = context_length
                print(f"[PROBING] CLIFF DETECTED at context {context_length}ms (TTFT {ttft:.0f}ms)")
                break
        
        self.envelope.last_probed = time.time()
        self._detect_trend()
        return self.envelope
    
    def _detect_trend(self):
        """Detect if envelope is shrinking (thermal throttle) or expanding"""
        if len(self.envelope.probes) < 2:
            self.envelope.trend = "stable"
            return
        
        latest_ttft = self.envelope.probes[-1].ttft_ms
        prev_ttft = self.envelope.probes[0].ttft_ms
        
        if latest_ttft > prev_ttft * 1.2:
            self.envelope.trend = "shrinking"  # Cliff getting closer (throttling)
        elif latest_ttft < prev_ttft * 0.9:
            self.envelope.trend = "expanding"  # Cliff moving farther (cooling)
        else:
            self.envelope.trend = "stable"


# ============================================================================
# 2. MODEL SHARDING - Temporal Residency Orchestration
# ============================================================================

class ModelResidency(Enum):
    VRAM = "vram"
    SYSTEM_RAM = "system_ram"
    DISK = "disk"

@dataclass
class ModelStatus:
    name: str
    residency: ModelResidency
    size_gb: float
    vram_gb: float = 0.0
    last_used: float = 0.0

class FluidOrchestrator:
    """Manages temporal model residency based on task prediction"""
    
    def __init__(self, max_resident_vram_gb: float = 6.5):
        self.max_resident_vram = max_resident_vram_gb
        self.cuda_overhead_gb = 1.5
        self.models: Dict[str, ModelStatus] = {}
        self.task_history: List[str] = []
        
    def predict_next_task(self) -> str:
        """Predict next task from history (simple recency)"""
        if not self.task_history:
            return "code_generation"
        return self.task_history[-1]
    
    def get_models_for_task(self, task_type: str) -> Dict[str, list]:
        """Return which models to keep resident per task type"""
        strategies = {
            "code_generation": ["main_model", "embed_model"],
            "document_retrieval": ["embed_model", "rerank_model"],
            "reasoning": ["main_model"],
            "tool_use": ["main_model", "embed_model"],
            "business": ["main_model", "embed_model", "rerank_model"],
        }
        return {
            "keep_resident": strategies.get(task_type, ["main_model", "embed_model"]),
            "evict": ["rerank_model"] if task_type == "code_generation" else []
        }
    
    def calculate_residency(self) -> Dict[str, Any]:
        """Calculate current VRAM utilization and residency"""
        total_vram_used = 0
        vram_resident = []
        system_ram = []
        disk = []
        
        for model_name, status in self.models.items():
            if status.residency == ModelResidency.VRAM:
                total_vram_used += status.vram_gb
                vram_resident.append(model_name)
            elif status.residency == ModelResidency.SYSTEM_RAM:
                system_ram.append(model_name)
            else:
                disk.append(model_name)
        
        free_vram = 8.0 - total_vram_used - self.cuda_overhead_gb
        
        return {
            "total_vram_used": total_vram_used,
            "free_vram": max(0, free_vram),
            "cuda_overhead": self.cuda_overhead_gb,
            "max_resident_vram": self.max_resident_vram,
            "vram_resident": vram_resident,
            "system_ram": system_ram,
            "disk": disk
        }


# ============================================================================
# 3. PRE-WARMING - Cold-Start Mitigation
# ============================================================================

@dataclass
class WarmingMetrics:
    cold_ttft_ms: float
    hot_ttft_ms: float
    warm_at_seconds: float
    time_since_generation: float

class PrewarmerService:
    """Detects and mitigates cold-start (model evicted to system RAM)"""
    
    def __init__(self, idle_threshold_s: int = 30, warm_timeout_s: int = 5):
        self.idle_threshold = idle_threshold_s
        self.warm_timeout = warm_timeout_s
        self.last_generation_time = 0
        self.cold_ttft_history: List[float] = []
        self.hot_ttft_history: List[float] = []
        
    def is_model_likely_cold(self) -> bool:
        """Predict if model was evicted to system RAM"""
        idle_time = time.time() - self.last_generation_time
        return idle_time > self.idle_threshold
    
    async def trigger_prewarm(self, model_id: str, base_url: str) -> WarmingMetrics:
        """Send dummy inference to warm model, measure TTFT"""
        import httpx
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 64,
            "stream": True
        }
        
        best_ttft = float('inf')
        max_attempts = 3
        
        for attempt in range(max_attempts):
            start = time.time()
            first_token_time = None
            
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream("POST", f"{base_url}/v1/chat/completions",
                                            json=payload, timeout=60) as resp:
                        if resp.status_code != 200:
                            continue
                        
                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and first_token_time is None:
                                first_token_time = time.time()
                                break
            except:
                continue
            
            if first_token_time:
                ttft = (first_token_time - start) * 1000
                best_ttft = min(best_ttft, ttft)
                
                if ttft < self.warm_timeout * 1000:
                    break  # Model is warm
                
                await asyncio.sleep(250 / 1000)  # 250ms between attempts
        
        return WarmingMetrics(
            cold_ttft_ms=best_ttft,
            hot_ttft_ms=0,
            warm_at_seconds=time.time(),
            time_since_generation=time.time() - self.last_generation_time
        )


# ============================================================================
# 4. TRUNCATION MONITORING - Output Quality
# ============================================================================

@dataclass
class TruncationPattern:
    task_type: str
    truncation_rate: float
    avg_tokens: int
    max_tokens_current: int
    recommended_max_tokens: int
    issues: List[str] = field(default_factory=list)

class TruncationMonitor:
    """Detects and suggests remediation for truncated outputs"""
    
    def __init__(self):
        self.task_patterns: Dict[str, list] = {}
        
    def detect_truncation_issues(self, content: str, token_count: int) -> List[str]:
        """Identify specific truncation problems"""
        issues = []
        
        # Check for unclosed code blocks
        open_blocks = content.count("```") % 2
        if open_blocks == 1:
            issues.append("unclosed_code_block")
        
        # Check for unclosed JSON
        if content.count("{") > content.count("}"):
            issues.append("unclosed_json")
        
        # Check for unclosed reasoning tags
        if content.count("<reasoning>") > content.count("</reasoning>"):
            issues.append("unclosed_reasoning")
        
        # Check for abrupt ending
        if len(content.strip()) > 20 and content.strip()[-1] not in ".!?,;:}])":
            issues.append("incomplete_sentence")
        
        return issues
    
    def record_pattern(self, task_type: str, token_count: int, max_tokens: int, 
                      issues: List[str], success: bool):
        """Record truncation pattern for analysis"""
        if task_type not in self.task_patterns:
            self.task_patterns[task_type] = []
        
        self.task_patterns[task_type].append({
            "token_count": token_count,
            "max_tokens": max_tokens,
            "issues": issues,
            "success": success,
            "timestamp": time.time()
        })
    
    def analyze_patterns(self) -> List[TruncationPattern]:
        """Analyze recorded patterns and suggest remediation"""
        patterns = []
        
        for task_type, records in self.task_patterns.items():
            if not records:
                continue
            
            failed_count = sum(1 for r in records if not r["success"] or r["issues"])
            truncation_rate = failed_count / len(records) if records else 0
            avg_tokens = sum(r["token_count"] for r in records) / len(records)
            all_issues = set()
            for r in records:
                all_issues.update(r["issues"])
            
            max_tokens_current = records[0]["max_tokens"]
            recommended = int(avg_tokens * 1.2) if truncation_rate > 0.3 else max_tokens_current
            
            patterns.append(TruncationPattern(
                task_type=task_type,
                truncation_rate=truncation_rate,
                avg_tokens=int(avg_tokens),
                max_tokens_current=max_tokens_current,
                recommended_max_tokens=recommended,
                issues=list(all_issues)
            ))
        
        return patterns


# ============================================================================
# 5. RESILIENCE MODES & FALLBACK CHAINS
# ============================================================================

class ResilienceMode(Enum):
    IDEAL = ("ideal", "🟢", 1, ["Qwen3.5-4B Q4_K_M", "Embed4B", "Rerank0.6B"])
    PRESSURE = ("pressure", "🟡", 2, ["Qwen3.5-4B Q3_K_M", "Embed4B"])
    EMERGENCY = ("emergency", "🟠", 3, ["Lfm2.5-1.2B", "Embed4B"])
    RETRIEVAL = ("retrieval", "🔴", 4, ["Embed4B", "Rerank0.6B"])
    CIRCUIT_BREAK = ("circuit_break", "⚫", 5, [])
    
    def __init__(self, name_str, emoji, level, models):
        self.name_str = name_str
        self.emoji = emoji
        self.level = level
        self.models = models

class FallbackOrchestrator:
    """Manages cascading fallback levels on failures"""
    
    def __init__(self):
        self.current_mode = ResilienceMode.IDEAL
        self.activation_history: List[Dict] = []
        
    def activate_next_level(self, reason: str) -> ResilienceMode:
        """Move to next fallback level"""
        current_level = self.current_mode.level
        next_levels = [m for m in ResilienceMode if m.level == current_level + 1]
        
        if not next_levels:
            self.current_mode = ResilienceMode.CIRCUIT_BREAK
        else:
            self.current_mode = next_levels[0]
        
        self.activation_history.append({
            "timestamp": time.time(),
            "from_level": current_level,
            "to_level": self.current_mode.level,
            "mode": self.current_mode.name_str,
            "reason": reason
        })
        
        return self.current_mode


# Database persistence for Phase 6
class PerfectionPathDB:
    """SQLite storage for Phase 6 metrics"""
    
    def __init__(self, db_path: str = "webapp/perfection_path.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Probing results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS probes (
                id INTEGER PRIMARY KEY,
                model TEXT,
                context_length INTEGER,
                ttft_ms REAL,
                tps REAL,
                timestamp REAL,
                success BOOLEAN,
                error TEXT
            )
        """)
        
        # Model residency
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS residency_events (
                id INTEGER PRIMARY KEY,
                model TEXT,
                event_type TEXT,
                previous_location TEXT,
                new_location TEXT,
                reason TEXT,
                timestamp REAL
            )
        """)
        
        # Truncation patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS truncation_events (
                id INTEGER PRIMARY KEY,
                task_type TEXT,
                token_count INTEGER,
                max_tokens INTEGER,
                issues TEXT,
                success BOOLEAN,
                timestamp REAL
            )
        """)
        
        # Fallback activations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fallback_events (
                id INTEGER PRIMARY KEY,
                from_level INTEGER,
                to_level INTEGER,
                mode TEXT,
                reason TEXT,
                timestamp REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_probe(self, model: str, probe: ProbeResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO probes (model, context_length, ttft_ms, tps, timestamp, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model, probe.context_length, probe.ttft_ms, probe.tps, 
              probe.timestamp, probe.success, probe.error))
        conn.commit()
        conn.close()
