import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).parent / "perfection_index.db"


@dataclass
class MetricPoint:
    timestamp: float
    tool_name: str
    metric_type: str
    value: float
    session_id: Optional[str] = None
    agent: Optional[str] = None
    context: Optional[Dict[str, Any]] = field(default_factory=dict)


class PerfectionIndexStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    tool_name TEXT,
                    metric_type TEXT,
                    value REAL,
                    session_id TEXT,
                    agent TEXT,
                    context TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indices (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    entity_type TEXT,
                    entity_id TEXT,
                    quality_score REAL,
                    reliability_index REAL,
                    velocity REAL,
                    perfection_index REAL
                )
                """
            )

    def add_metric(self, metric: MetricPoint) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics (timestamp, tool_name, metric_type, value, session_id, agent, context) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    metric.timestamp,
                    metric.tool_name,
                    metric.metric_type,
                    metric.value,
                    metric.session_id,
                    metric.agent,
                    json.dumps(metric.context or {}),
                ),
            )

    def add_index(self, entity_type: str, entity_id: str, quality_score: float, reliability_index: float, velocity: float, perfection_index: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO indices (timestamp, entity_type, entity_id, quality_score, reliability_index, velocity, perfection_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), entity_type, entity_id, quality_score, reliability_index, velocity, perfection_index),
            )

    def query_metrics(self, tool_name: Optional[str] = None, since: Optional[float] = None) -> List[Dict[str, Any]]:
        q = "SELECT timestamp, tool_name, metric_type, value, session_id, agent, context FROM metrics"
        conditions = []
        values = []
        if tool_name:
            conditions.append("tool_name = ?")
            values.append(tool_name)
        if since is not None:
            conditions.append("timestamp >= ?")
            values.append(since)
        if conditions:
            q += " WHERE " + " AND ".join(conditions)
        q += " ORDER BY timestamp ASC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(q, tuple(values)).fetchall()
        results = []
        for row in rows:
            results.append({
                "timestamp": row[0],
                "tool_name": row[1],
                "metric_type": row[2],
                "value": row[3],
                "session_id": row[4],
                "agent": row[5],
                "context": json.loads(row[6] or "{}"),
            })
        return results

    def reset(self) -> None:
        """Clear all metrics and indices from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM metrics")
            conn.execute("DELETE FROM indices")


class PerfectionIndexTracker:
    def __init__(self, store: Optional[PerfectionIndexStore] = None):
        self.store = store or PerfectionIndexStore()
        self.lock = threading.RLock()

    def record_tool_execution(self, tool_name: str, latency_ms: float, success: bool, session_id: Optional[str] = None, agent: Optional[str] = None, error: Optional[str] = None) -> None:
        timestamp = time.time()
        self.store.add_metric(MetricPoint(timestamp, tool_name, "latency", latency_ms, session_id, agent, {"error": error}))
        self.store.add_metric(MetricPoint(timestamp, tool_name, "success", 1.0 if success else 0.0, session_id, agent, {"error": error}))
        if not success:
            self.store.add_metric(MetricPoint(timestamp, tool_name, "failure", 1.0, session_id, agent, {"error": error}))

    def calculate_tool_indices(self, tool_name: str, window_minutes: int = 60) -> Dict[str, float]:
        since = time.time() - window_minutes * 60.0
        entries = self.store.query_metrics(tool_name=tool_name, since=since)
        total_calls = len([e for e in entries if e["metric_type"] == "success" or e["metric_type"] == "failure"])
        successes = len([e for e in entries if e["metric_type"] == "success" and e["value"] >= 1.0])
        failures = len([e for e in entries if e["metric_type"] == "failure" and e["value"] >= 1.0])
        latencies = [e["value"] for e in entries if e["metric_type"] == "latency"]
        velocity = total_calls / max(1.0, window_minutes)
        success_rate = successes / max(1, total_calls)
        error_rate = failures / max(1, total_calls)
        avg_latency = sum(latencies) / max(1, len(latencies)) if latencies else 0.0
        quality_score = (success_rate * 0.4 + (1 - error_rate) * 0.3 + (1.0 - min(avg_latency / 1000.0, 1.0)) * 0.3) * 100
        reliability_index = max(0.0, min(1.0, success_rate - error_rate * 0.5))
        perfection_index = (quality_score / 100.0 * 0.5 + reliability_index * 0.5)
        self.store.add_index("tool", tool_name, quality_score, reliability_index, velocity, perfection_index)
        return {
            "tool": tool_name,
            "quality_score": round(quality_score, 2),
            "reliability_index": round(reliability_index, 4),
            "velocity": round(velocity, 4),
            "perfection_index": round(perfection_index, 4),
            "calls": total_calls,
            "successes": successes,
            "failures": failures,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def get_system_health_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        with self.lock:
            tool_names = list({m["tool_name"] for m in self.store.query_metrics()})
            metrics = [self.calculate_tool_indices(tool, window_minutes=window_minutes) for tool in tool_names]
            if not metrics:
                return {
                    "perfection_index": 0.0,
                    "quality_score": 0.0,
                    "reliability_index": 0.0,
                    "velocity": 0.0,
                    "tools": [],
                }
            avg_quality = sum(m["quality_score"] for m in metrics) / len(metrics)
            avg_reliability = sum(m["reliability_index"] for m in metrics) / len(metrics)
            avg_velocity = sum(m["velocity"] for m in metrics)
            overall_perfection = (avg_quality / 100.0 * 0.5 + avg_reliability * 0.5)
            return {
                "perfection_index": round(overall_perfection, 4),
                "quality_score": round(avg_quality, 2),
                "reliability_index": round(avg_reliability, 4),
                "velocity": round(avg_velocity, 4),
                "tools": metrics,
            }

    def get_index_report(self) -> Dict[str, Any]:
        """Get comprehensive perfection index report."""
        summary = self.get_system_health_summary()
        return {
            "timestamp": time.time(),
            "global_index": summary.get("perfection_index", 0.0),
            "quality_score": summary.get("quality_score", 0.0),
            "reliability_index": summary.get("reliability_index", 0.0),
            "tool_metrics": {m["tool"]: m for m in summary.get("tools", [])},
            "trends": {"velocity": summary.get("velocity", 0.0)},
        }

    def reset_metrics(self) -> None:
        """Reset all accumulated metrics."""
        with self.lock:
            self.store.reset()


perfection_tracker = PerfectionIndexTracker()
perfection_index = perfection_tracker  # Alias for convenience


