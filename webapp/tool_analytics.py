"""Tool monitoring, remediation, and analytics system."""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

DEFAULT_ANALYTICS_DB = Path(__file__).parent / "tool_analytics.db"


class RemediationAction(str, Enum):
    """Type of remediation action to take."""
    RESTART_TOOL = "restart_tool"
    THROTTLE_CALLS = "throttle_calls"
    INCREASE_TIMEOUT = "increase_timeout"
    ISOLATE_TOOL = "isolate_tool"
    RESET_CIRCUIT = "reset_circuit"
    LOG_ISSUE = "log_issue"
    ALERT_OPERATOR = "alert_operator"
    FALLBACK_STRATEGY = "fallback_strategy"


class AnomalyType(str, Enum):
    """Type of detected anomaly."""
    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_EXECUTION = "slow_execution"
    SUDDEN_SPIKE = "sudden_spike"
    DEGRADATION = "degradation"
    MISSING_DATA = "missing_data"
    VARIANCE_SPIKE = "variance_spike"


@dataclass
class RemediationAction_Record:
    """Record of remediation action taken."""
    tool_name: str
    action: RemediationAction
    timestamp: float
    reason: str
    result: str = "pending"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action": self.action.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "result": self.result,
            "metadata": self.metadata or {},
        }


@dataclass
class Anomaly:
    """Detected anomaly record."""
    tool_name: str
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    timestamp: float
    details: Dict[str, Any]
    suggested_fixes: List[RemediationAction] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "details": self.details,
            "suggested_fixes": [a.value for a in (self.suggested_fixes or [])],
        }


class ToolAnalyticsStore:
    """Persistent storage for analytics data."""

    def __init__(self, db_path: Path = DEFAULT_ANALYTICS_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Lifecycle events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    tool_name TEXT,
                    event_type TEXT,
                    duration_ms REAL,
                    success INTEGER,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            # Anomalies
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    tool_name TEXT,
                    anomaly_type TEXT,
                    severity REAL,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            """)
            # Remediation actions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS remediation_actions (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    tool_name TEXT,
                    action TEXT,
                    reason TEXT,
                    result TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_tool ON lifecycle_events(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_tool ON anomalies(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_remediation_tool ON remediation_actions(tool_name)")

    def log_lifecycle(self, tool_name: str, event_type: str, duration_ms: float, success: bool, error: Optional[str] = None, metadata: Optional[Dict] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO lifecycle_events (timestamp, tool_name, event_type, duration_ms, success, error_message, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tool_name, event_type, duration_ms, 1 if success else 0, error or "", json.dumps(metadata or {}))
            )

    def log_anomaly(self, anomaly: Anomaly):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO anomalies (timestamp, tool_name, anomaly_type, severity, details) VALUES (?, ?, ?, ?, ?)",
                (anomaly.timestamp, anomaly.tool_name, anomaly.anomaly_type.value, anomaly.severity, json.dumps(anomaly.details))
            )

    def log_remediation(self, action: RemediationAction_Record):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO remediation_actions (timestamp, tool_name, action, reason, result, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (action.timestamp, action.tool_name, action.action.value, action.reason, action.result, json.dumps(action.metadata or {}))
            )

    def get_tool_history(self, tool_name: str, limit: int = 100) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            # Lifecycle
            lifecycle = conn.execute(
                "SELECT timestamp, event_type, duration_ms, success FROM lifecycle_events WHERE tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, limit)
            ).fetchall()
            # Anomalies
            anomalies = conn.execute(
                "SELECT timestamp, anomaly_type, severity FROM anomalies WHERE tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, limit // 2)
            ).fetchall()
            # Remediation
            remediations = conn.execute(
                "SELECT timestamp, action, result FROM remediation_actions WHERE tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, limit // 2)
            ).fetchall()
        return {
            "lifecycle": [{"timestamp": r[0], "event": r[1], "duration_ms": r[2], "success": bool(r[3])} for r in lifecycle],
            "anomalies": [{"timestamp": r[0], "type": r[1], "severity": r[2]} for r in anomalies],
            "remediation": [{"timestamp": r[0], "action": r[1], "result": r[2]} for r in remediations],
        }


class AnomalyDetector:
    """Detect anomalies in tool behavior."""

    def __init__(self):
        self.lock = threading.RLock()
        self.history: Dict[str, List[float]] = {}  # tool_name -> latencies
        self.error_counts: Dict[str, int] = {}

    def record_execution(self, tool_name: str, latency_ms: float, error: bool = False):
        with self.lock:
            if tool_name not in self.history:
                self.history[tool_name] = []
            self.history[tool_name].append(latency_ms)
            # Keep last 100
            if len(self.history[tool_name]) > 100:
                self.history[tool_name] = self.history[tool_name][-100:]

            if error:
                self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
            else:
                # Success resets error count
                pass

    def detect_anomalies(self, tool_name: str) -> List[Anomaly]:
        """Detect anomalies for given tool."""
        with self.lock:
            anomalies = []
            latencies = self.history.get(tool_name, [])

            if not latencies or len(latencies) < 3:
                return anomalies

            # Check 1: High error rate
            errors = self.error_counts.get(tool_name, 0)
            error_rate = errors / max(len(latencies), 1)
            if error_rate > 0.3:
                anomalies.append(Anomaly(
                    tool_name=tool_name,
                    anomaly_type=AnomalyType.HIGH_ERROR_RATE,
                    severity=min(1.0, error_rate),
                    timestamp=time.time(),
                    details={"error_rate": error_rate, "error_count": errors},
                    suggested_fixes=[RemediationAction.ISOLATE_TOOL, RemediationAction.LOG_ISSUE]
                ))

            # Check 2: Slow execution (avg > 500ms)
            avg_latency = sum(latencies) / len(latencies)
            if avg_latency > 500:
                anomalies.append(Anomaly(
                    tool_name=tool_name,
                    anomaly_type=AnomalyType.SLOW_EXECUTION,
                    severity=min(1.0, avg_latency / 5000.0),
                    timestamp=time.time(),
                    details={"avg_latency_ms": avg_latency, "samples": len(latencies)},
                    suggested_fixes=[RemediationAction.INCREASE_TIMEOUT, RemediationAction.LOG_ISSUE]
                ))

            # Check 3: Sudden spike (last execution >> average)
            if len(latencies) >= 5:
                recent_avg = sum(latencies[-5:]) / 5
                oldest_avg = sum(latencies[:-5]) / (len(latencies) - 5) if len(latencies) > 5 else recent_avg
                if recent_avg > oldest_avg * 2.5:
                    anomalies.append(Anomaly(
                        tool_name=tool_name,
                        anomaly_type=AnomalyType.SUDDEN_SPIKE,
                        severity=min(1.0, (recent_avg / oldest_avg - 1) / 2),
                        timestamp=time.time(),
                        details={"recent_avg": recent_avg, "historical_avg": oldest_avg},
                        suggested_fixes=[RemediationAction.LOG_ISSUE, RemediationAction.INCREASE_TIMEOUT]
                    ))

            return anomalies


class RemediationEngine:
    """Apply remediation actions based on anomalies and health state."""

    def __init__(self, analytics_store: ToolAnalyticsStore):
        self.store = analytics_store
        self.lock = threading.RLock()
        self.pending_actions: List[RemediationAction_Record] = []
        self.action_callbacks: Dict[RemediationAction, Callable] = {}

    def register_action_callback(self, action: RemediationAction, callback: Callable):
        """Register a callback for a remediation action."""
        with self.lock:
            self.action_callbacks[action] = callback

    def trigger_remediation(self, tool_name: str, anomalies: List[Anomaly]) -> List[RemediationAction_Record]:
        """Trigger remediation based on anomalies."""
        actions = []
        with self.lock:
            for anomaly in anomalies:
                for suggested_action in (anomaly.suggested_fixes or []):
                    action = RemediationAction_Record(
                        tool_name=tool_name,
                        action=suggested_action,
                        timestamp=time.time(),
                        reason=f"{anomaly.anomaly_type.value} (severity: {anomaly.severity:.2f})",
                    )
                    # Try to execute callback if registered
                    if suggested_action in self.action_callbacks:
                        try:
                            self.action_callbacks[suggested_action](tool_name)
                            action.result = "executed"
                        except Exception as e:
                            action.result = f"failed: {str(e)}"
                    else:
                        action.result = "no_handler"

                    self.store.log_remediation(action)
                    actions.append(action)
                    self.pending_actions.append(action)

            # Keep last 1000 pending actions
            if len(self.pending_actions) > 1000:
                self.pending_actions = self.pending_actions[-1000:]

        return actions

    def get_pending_actions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get pending remediation actions."""
        with self.lock:
            return [a.to_dict() for a in self.pending_actions[-limit:]]


# Global instances
analytics_store = ToolAnalyticsStore()
anomaly_detector = AnomalyDetector()
remediation_engine = RemediationEngine(analytics_store)
