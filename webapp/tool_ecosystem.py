import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ToolStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ISOLATED = "isolated"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ToolCategory(str, Enum):
    FILE_IO = "file_io"
    CODE_ANALYSIS = "code_analysis"
    EXECUTION = "execution"
    RETRIEVAL = "retrieval"
    EXTERNAL = "external"
    SYSTEM = "system"


@dataclass
class ToolMetrics:
    name: str
    category: ToolCategory
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    circuit_opened_at: Optional[float] = None
    rate_limiter_tokens: float = 20.0
    rate_limiter_last_refill: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.calls == 0:
            return 1.0
        return self.successes / self.calls

    @property
    def error_rate(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.failures / self.calls

    @property
    def avg_latency_ms(self) -> float:
        if self.successes + self.failures == 0:
            return 0.0
        return self.total_latency_ms / max(1, self.successes + self.failures)


@dataclass
class ToolCluster:
    cluster_id: str
    tools: List[str] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_sec: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_sec = timeout_sec
        self.failure_counts: Dict[str, int] = {}
        self.opened_at: Dict[str, float] = {}

    def record_failure(self, tool_name: str) -> None:
        self.failure_counts[tool_name] = self.failure_counts.get(tool_name, 0) + 1
        if self.failure_counts[tool_name] >= self.failure_threshold:
            self.opened_at[tool_name] = time.time()

    def record_success(self, tool_name: str) -> None:
        self.failure_counts[tool_name] = 0
        self.opened_at.pop(tool_name, None)

    def is_open(self, tool_name: str) -> bool:
        opened_at = self.opened_at.get(tool_name)
        if not opened_at:
            return False
        if time.time() - opened_at > self.timeout_sec:
            self.opened_at.pop(tool_name, None)
            self.failure_counts[tool_name] = 0
            return False
        return True

    def reset(self, tool_name: str) -> None:
        self.failure_counts[tool_name] = 0
        self.opened_at.pop(tool_name, None)


class RateLimiter:
    def __init__(self, tokens_per_minute: int = 20):
        self.tokens_per_minute = tokens_per_minute

    def refill(self, metrics: ToolMetrics) -> None:
        now = time.time()
        elapsed = now - metrics.rate_limiter_last_refill
        if elapsed <= 0:
            return
        tokens_to_add = elapsed * (self.tokens_per_minute / 60.0)
        metrics.rate_limiter_tokens = min(self.tokens_per_minute, metrics.rate_limiter_tokens + tokens_to_add)
        metrics.rate_limiter_last_refill = now

    def consume(self, metrics: ToolMetrics, required: float = 1.0) -> bool:
        self.refill(metrics)
        if metrics.rate_limiter_tokens >= required:
            metrics.rate_limiter_tokens -= required
            return True
        return False


class ToolHealthMonitor:
    def __init__(self):
        self._lock = threading.RLock()
        self.tools: Dict[str, ToolMetrics] = {}
        self.clusters: Dict[str, ToolCluster] = {}
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.isolation_events: List[Dict[str, Any]] = []

    def register_tool(self, tool_name: str, category: ToolCategory, cluster_id: Optional[str] = None):
        with self._lock:
            if tool_name not in self.tools:
                self.tools[tool_name] = ToolMetrics(name=tool_name, category=category)
            if cluster_id:
                cluster = self.clusters.setdefault(cluster_id, ToolCluster(cluster_id=cluster_id))
                if tool_name not in cluster.tools:
                    cluster.tools.append(tool_name)

    def record_execution(self, tool_name: str, latency_ms: float, success: bool, error_msg: Optional[str] = None) -> None:
        with self._lock:
            if tool_name not in self.tools:
                self.tools[tool_name] = ToolMetrics(name=tool_name, category=ToolCategory.SYSTEM)
            metrics = self.tools[tool_name]
            metrics.calls += 1
            metrics.total_latency_ms += latency_ms
            if success:
                metrics.successes += 1
                metrics.last_error = None
                self.circuit_breaker.record_success(tool_name)
            else:
                metrics.failures += 1
                metrics.last_error = error_msg
                self.circuit_breaker.record_failure(tool_name)
                if self.circuit_breaker.is_open(tool_name):
                    self.isolation_events.append({
                        "tool": tool_name,
                        "at": time.time(),
                        "state": "isolated",
                        "error": error_msg,
                    })

    def can_execute_tool(self, tool_name: str) -> Tuple[bool, str]:
        with self._lock:
            if self.circuit_breaker.is_open(tool_name):
                return False, "circuit_open"
            metrics = self.tools.get(tool_name)
            if not metrics:
                return True, "not_registered"
            if not self.rate_limiter.consume(metrics):
                return False, "rate_limited"
            return True, "ok"

    def get_tool_status(self, tool_name: str) -> Dict[str, Any]:
        with self._lock:
            metrics = self.tools.get(tool_name)
            if not metrics:
                return {"tool": tool_name, "status": ToolStatus.UNKNOWN.value}
            status = ToolStatus.HEALTHY
            if self.circuit_breaker.is_open(tool_name):
                status = ToolStatus.ISOLATED
            elif metrics.error_rate > 0.5:
                status = ToolStatus.FAILED
            elif metrics.error_rate > 0.2:
                status = ToolStatus.DEGRADED
            return {
                "tool": tool_name,
                "status": status.value,
                "calls": metrics.calls,
                "successes": metrics.successes,
                "failures": metrics.failures,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "last_error": metrics.last_error,
                "circuit_open": self.circuit_breaker.is_open(tool_name),
            }

    def reset_tool(self, tool_name: str) -> Dict[str, Any]:
        with self._lock:
            self.circuit_breaker.reset(tool_name)
            return self.get_tool_status(tool_name)

    def get_health_report(self) -> Dict[str, Any]:
        with self._lock:
            tool_statuses = {t: self.get_tool_status(t) for t in self.tools}
            cluster_data = {}
            for cid, cluster in self.clusters.items():
                tools = [self.get_tool_status(t) for t in cluster.tools]
                degraded = sum(1 for t in tools if t["status"] != ToolStatus.HEALTHY.value)
                cluster_data[cid] = {
                    "tools": tools,
                    "degraded_count": degraded,
                    "total_tools": len(tools),
                }
            overall_health = 1.0
            if tool_statuses:
                overall_health = sum(1.0 - s["error_rate"] for s in tool_statuses.values()) / len(tool_statuses)
            return {
                "timestamp": time.time(),
                "overall_health": max(0.0, min(1.0, overall_health)),
                "tools": tool_statuses,
                "clusters": cluster_data,
                "isolation_events": list(self.isolation_events[-100:]),
            }

    def get_health_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical health timeline (last N records)."""
        with self._lock:
            # Return isolation events as timeline (these are the main health events)
            return [
                {
                    "timestamp": e.get("timestamp", time.time()),
                    "tool": e.get("tool", "unknown"),
                    "event_type": e.get("event_type", "isolation"),
                    "reason": e.get("reason", ""),
                }
                for e in list(self.isolation_events[-limit:])
            ]


tool_health_monitor = ToolHealthMonitor()

