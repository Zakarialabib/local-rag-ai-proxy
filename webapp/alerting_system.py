"""Alerting System - Send notifications for tool anomalies and issues"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_ALERTS_DB = Path(__file__).parent / "alerts.db"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class Alert:
    """Alert record."""
    id: str
    timestamp: float
    tool_name: str
    anomaly_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    channel_sent: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "channel_sent": self.channel_sent or [],
        }


class AlertStore:
    """Persistent alert storage."""

    def __init__(self, db_path: Path = DEFAULT_ALERTS_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    tool_name TEXT,
                    anomaly_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    channel_sent TEXT,
                    acknowledged_at REAL,
                    acknowledged_by TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_tool ON alerts(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")

    def store_alert(self, alert: Alert):
        """Store alert in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO alerts (id, timestamp, tool_name, anomaly_type, severity, 
                   message, details, acknowledged, channel_sent) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert.id,
                    alert.timestamp,
                    alert.tool_name,
                    alert.anomaly_type,
                    alert.severity.value,
                    alert.message,
                    json.dumps(alert.details),
                    1 if alert.acknowledged else 0,
                    json.dumps(alert.channel_sent or []),
                ),
            )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Mark alert as acknowledged."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ? WHERE id = ?",
                (time.time(), acknowledged_by, alert_id),
            )

    def get_recent_alerts(self, limit: int = 100, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        query = "SELECT * FROM alerts"
        params = []
        if acknowledged is not None:
            query += " WHERE acknowledged = ?"
            params.append(1 if acknowledged else 0)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        alerts = []
        for row in rows:
            alerts.append({
                "id": row[0],
                "timestamp": row[1],
                "tool_name": row[2],
                "anomaly_type": row[3],
                "severity": row[4],
                "message": row[5],
                "details": json.loads(row[6]),
                "acknowledged": bool(row[7]),
            })
        return alerts

    def get_tool_alerts(self, tool_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts for a specific tool."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, limit),
            ).fetchall()

        alerts = []
        for row in rows:
            alerts.append({
                "tool_name": row[2],
                "anomaly_type": row[3],
                "severity": row[4],
                "timestamp": row[1],
                "message": row[5],
            })
        return alerts


class AlertManager:
    """Manage alerts and send to multiple channels."""

    def __init__(self, store: Optional[AlertStore] = None):
        self.store = store or AlertStore()
        self.channels: Dict[AlertChannel, Callable] = {}
        self.lock = threading.RLock()
        self._register_default_channels()

    def _register_default_channels(self):
        """Register default alert channels."""
        import logging
        logger = logging.getLogger(__name__)

        def log_alert(alert: Alert):
            level = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
            }.get(alert.severity, logging.INFO)
            logger.log(level, f"Alert: {alert.message}")

        def dashboard_alert(alert: Alert):
            # Broadcast to dashboard via SSE (will be handled by dashboard_events)
            pass

        self.register_channel(AlertChannel.LOG, log_alert)
        self.register_channel(AlertChannel.DASHBOARD, dashboard_alert)

    def register_channel(self, channel: AlertChannel, handler: Callable):
        """Register an alert channel handler."""
        with self.lock:
            self.channels[channel] = handler

    def send_alert(
        self,
        tool_name: str,
        anomaly_type: str,
        severity: float,
        message: str,
        details: Dict[str, Any] = None,
        channels: List[AlertChannel] = None,
    ):
        """Send alert through configured channels."""
        import uuid

        # Map severity (0.0-1.0) to AlertSeverity
        if severity >= 0.8:
            alert_severity = AlertSeverity.CRITICAL
        elif severity >= 0.6:
            alert_severity = AlertSeverity.HIGH
        elif severity >= 0.4:
            alert_severity = AlertSeverity.MEDIUM
        else:
            alert_severity = AlertSeverity.LOW

        alert = Alert(
            id=f"alert_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            tool_name=tool_name,
            anomaly_type=anomaly_type,
            severity=alert_severity,
            message=message,
            details=details or {},
            channel_sent=[]
        )

        # Store alert
        self.store.store_alert(alert)

        # Send through channels
        channels = channels or [AlertChannel.LOG, AlertChannel.DASHBOARD]
        for channel in channels:
            try:
                if channel in self.channels:
                    self.channels[channel](alert)
                    alert.channel_sent.append(channel.value)
            except Exception as e:
                pass  # Continue to next channel on error

        return alert

    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        alerts = self.store.get_recent_alerts(limit=1000)
        recent_alerts = [a for a in alerts if time.time() - a["timestamp"] < 3600]  # Last hour

        severity_counts = {
            "critical": len([a for a in recent_alerts if a["severity"] == "critical"]),
            "high": len([a for a in recent_alerts if a["severity"] == "high"]),
            "medium": len([a for a in recent_alerts if a["severity"] == "medium"]),
            "low": len([a for a in recent_alerts if a["severity"] == "low"]),
        }

        return {
            "total_alerts": len(alerts),
            "recent_alerts_1h": len(recent_alerts),
            "severity_distribution": severity_counts,
            "most_alerted_tools": self._get_most_alerted_tools(alerts),
            "pending_acknowledgment": len([a for a in recent_alerts if not a["acknowledged"]]),
        }

    def _get_most_alerted_tools(self, alerts: List[Dict], limit: int = 5) -> List[Dict]:
        """Get tools with most alerts."""
        tool_counts = {}
        for alert in alerts:
            tool = alert["tool_name"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"tool": tool, "alerts": count} for tool, count in sorted_tools[:limit]]


# Global alert manager instance
alert_manager = AlertManager()
