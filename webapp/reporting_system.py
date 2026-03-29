"""Reporting System - Generate trend reports and detect regressions"""

import json
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REPORTS_DB = Path(__file__).parent / "reports.db"


class ReportType(str, Enum):
    """Report types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class RegressionType(str, Enum):
    """Regression detection types."""
    LATENCY_INCREASE = "latency_increase"
    ERROR_RATE_INCREASE = "error_rate_increase"
    PERFECTION_DECREASE = "perfection_decrease"
    ANOMALY_SPIKE = "anomaly_spike"


@dataclass
class TrendMetrics:
    """Trend metrics for a tool."""
    tool_name: str
    period: str  # e.g., "2024-01-15"
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float
    anomaly_count: int
    avg_perfection: float
    health_score: float
    request_count: int


@dataclass
class RegressionAlert:
    """Regression detection alert."""
    tool_name: str
    regression_type: RegressionType
    severity: float  # 0.0-1.0
    current_value: float
    baseline_value: float
    change_percent: float
    detected_at: float
    details: Dict[str, Any]


class ReportStore:
    """Persistent report storage."""

    def __init__(self, db_path: Path = DEFAULT_REPORTS_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trend_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    period TEXT,
                    avg_latency REAL,
                    p50_latency REAL,
                    p95_latency REAL,
                    p99_latency REAL,
                    error_rate REAL,
                    anomaly_count INTEGER,
                    avg_perfection REAL,
                    health_score REAL,
                    request_count INTEGER,
                    timestamp REAL,
                    UNIQUE(tool_name, period)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    regression_type TEXT,
                    severity REAL,
                    current_value REAL,
                    baseline_value REAL,
                    change_percent REAL,
                    detected_at REAL,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT,
                    tool_name TEXT,
                    period_start REAL,
                    period_end REAL,
                    content TEXT,
                    generated_at REAL,
                    format TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trend_tool ON trend_metrics(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trend_period ON trend_metrics(period DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_regression_tool ON regression_alerts(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type)")

    def store_trend(self, metrics: TrendMetrics):
        """Store trend metrics."""
        import time
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trend_metrics 
                (tool_name, period, avg_latency, p50_latency, p95_latency, p99_latency,
                 error_rate, anomaly_count, avg_perfection, health_score, request_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.tool_name,
                metrics.period,
                metrics.avg_latency,
                metrics.p50_latency,
                metrics.p95_latency,
                metrics.p99_latency,
                metrics.error_rate,
                metrics.anomaly_count,
                metrics.avg_perfection,
                metrics.health_score,
                metrics.request_count,
                time.time(),
            ))

    def store_regression(self, regression: RegressionAlert):
        """Store regression alert."""
        import time
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO regression_alerts
                (tool_name, regression_type, severity, current_value, baseline_value, change_percent, detected_at, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                regression.tool_name,
                regression.regression_type.value,
                regression.severity,
                regression.current_value,
                regression.baseline_value,
                regression.change_percent,
                regression.detected_at,
                json.dumps(regression.details),
            ))

    def get_trend_history(self, tool_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get trend history for a tool."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM trend_metrics WHERE tool_name = ? AND period >= ? ORDER BY period DESC",
                (tool_name, cutoff_date[:10]),
            ).fetchall()

        trends = []
        for row in rows:
            trends.append({
                "tool_name": row[1],
                "period": row[2],
                "avg_latency": row[3],
                "p50_latency": row[4],
                "p95_latency": row[5],
                "p99_latency": row[6],
                "error_rate": row[7],
                "anomaly_count": row[8],
                "avg_perfection": row[9],
                "health_score": row[10],
                "request_count": row[11],
            })
        return trends

    def get_regressions(self, tool_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get regression alerts."""
        query = "SELECT * FROM regression_alerts"
        params = []
        if tool_name:
            query += " WHERE tool_name = ?"
            params.append(tool_name)
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        regressions = []
        for row in rows:
            regressions.append({
                "tool_name": row[1],
                "regression_type": row[2],
                "severity": row[3],
                "current_value": row[4],
                "baseline_value": row[5],
                "change_percent": row[6],
                "detected_at": row[7],
                "details": json.loads(row[8]),
            })
        return regressions


class TrendAnalyzer:
    """Analyze trends and detect regressions."""

    def __init__(self, store: Optional[ReportStore] = None):
        self.store = store or ReportStore()
        self.baseline_window = 7  # days for baseline calculation

    def analyze_tool(self, tool_name: str, analytics_data: Dict[str, Any]) -> TrendMetrics:
        """Analyze tool analytics data to create trend metrics."""
        history = self.store.get_trend_history(tool_name, days=1)

        # Calculate latency percentiles
        latencies = analytics_data.get("latencies", [])
        if not latencies:
            latencies = [0]  # Avoid empty list error

        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0

        return TrendMetrics(
            tool_name=tool_name,
            period=datetime.now().isoformat()[:10],
            avg_latency=statistics.mean(latencies) if latencies else 0,
            p50_latency=p50,
            p95_latency=p95,
            p99_latency=p99,
            error_rate=analytics_data.get("error_rate", 0),
            anomaly_count=analytics_data.get("anomaly_count", 0),
            avg_perfection=analytics_data.get("avg_perfection", 1.0),
            health_score=analytics_data.get("health_score", 1.0),
            request_count=analytics_data.get("request_count", 0),
        )

    def detect_regressions(self, tool_name: str, current: TrendMetrics) -> List[RegressionAlert]:
        """Detect regressions by comparing current metrics to baseline."""
        baseline_history = self.store.get_trend_history(tool_name, days=self.baseline_window)

        if not baseline_history:
            return []  # No baseline data

        # Calculate baseline averages
        baseline_latency = statistics.mean([m["avg_latency"] for m in baseline_history])
        baseline_error_rate = statistics.mean([m["error_rate"] for m in baseline_history])
        baseline_perfection = statistics.mean([m["avg_perfection"] for m in baseline_history])

        regressions = []

        # Detect latency increase
        if baseline_latency > 0 and current.avg_latency > baseline_latency * 1.5:
            change_percent = ((current.avg_latency - baseline_latency) / baseline_latency) * 100
            severity = min(1.0, change_percent / 100.0)
            regressions.append(RegressionAlert(
                tool_name=tool_name,
                regression_type=RegressionType.LATENCY_INCREASE,
                severity=severity,
                current_value=current.avg_latency,
                baseline_value=baseline_latency,
                change_percent=change_percent,
                detected_at=datetime.now().timestamp(),
                details={"threshold_multiplier": 1.5},
            ))

        # Detect error rate increase
        if current.error_rate > baseline_error_rate + 0.05:
            change_percent = ((current.error_rate - baseline_error_rate) / max(baseline_error_rate, 0.01)) * 100
            severity = min(1.0, current.error_rate)
            regressions.append(RegressionAlert(
                tool_name=tool_name,
                regression_type=RegressionType.ERROR_RATE_INCREASE,
                severity=severity,
                current_value=current.error_rate,
                baseline_value=baseline_error_rate,
                change_percent=change_percent,
                detected_at=datetime.now().timestamp(),
                details={"threshold_increase": 0.05},
            ))

        # Detect perfection decrease
        if current.avg_perfection < baseline_perfection - 0.1:
            change_percent = ((baseline_perfection - current.avg_perfection) / baseline_perfection) * 100
            severity = min(1.0, change_percent / 100.0)
            regressions.append(RegressionAlert(
                tool_name=tool_name,
                regression_type=RegressionType.PERFECTION_DECREASE,
                severity=severity,
                current_value=current.avg_perfection,
                baseline_value=baseline_perfection,
                change_percent=change_percent,
                detected_at=datetime.now().timestamp(),
                details={"threshold_decrease": 0.1},
            ))

        # Store detected regressions
        for regression in regressions:
            self.store.store_regression(regression)

        return regressions


class ReportGenerator:
    """Generate trend reports."""

    def __init__(self, store: Optional[ReportStore] = None, analyzer: Optional[TrendAnalyzer] = None):
        self.store = store or ReportStore()
        self.analyzer = analyzer or TrendAnalyzer(self.store)

    def generate_daily_report(self, tool_name: str) -> Dict[str, Any]:
        """Generate daily report for tool."""
        history = self.store.get_trend_history(tool_name, days=1)
        if not history:
            return {"error": "No data available"}

        latest = history[0]
        return {
            "report_type": "daily",
            "tool_name": tool_name,
            "date": latest["period"],
            "metrics": latest,
            "status": self._get_status(latest),
        }

    def generate_weekly_report(self, tool_name: str) -> Dict[str, Any]:
        """Generate weekly report for tool."""
        history = self.store.get_trend_history(tool_name, days=7)
        if not history:
            return {"error": "No data available"}

        # Aggregate metrics
        latencies = [m["avg_latency"] for m in history]
        error_rates = [m["error_rate"] for m in history]
        perfection_scores = [m["avg_perfection"] for m in history]

        report = {
            "report_type": "weekly",
            "tool_name": tool_name,
            "period_days": 7,
            "metrics": {
                "avg_latency": statistics.mean(latencies),
                "max_latency": max(latencies),
                "avg_error_rate": statistics.mean(error_rates),
                "max_error_rate": max(error_rates),
                "avg_perfection": statistics.mean(perfection_scores),
                "min_perfection": min(perfection_scores),
                "total_requests": sum(m["request_count"] for m in history),
                "total_anomalies": sum(m["anomaly_count"] for m in history),
            },
            "trend": self._detect_trend(history),
            "regressions": self.store.get_regressions(tool_name, limit=10),
        }
        return report

    def generate_monthly_report(self, tool_name: str) -> Dict[str, Any]:
        """Generate monthly report for tool."""
        history = self.store.get_trend_history(tool_name, days=30)
        if not history:
            return {"error": "No data available"}

        latencies = [m["avg_latency"] for m in history]
        error_rates = [m["error_rate"] for m in history]
        perfection_scores = [m["avg_perfection"] for m in history]

        report = {
            "report_type": "monthly",
            "tool_name": tool_name,
            "period_days": 30,
            "metrics": {
                "avg_latency": statistics.mean(latencies),
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
                "avg_error_rate": statistics.mean(error_rates),
                "avg_perfection": statistics.mean(perfection_scores),
                "total_requests": sum(m["request_count"] for m in history),
                "total_anomalies": sum(m["anomaly_count"] for m in history),
                "critical_regressions": len([r for r in self.store.get_regressions(tool_name) if r["severity"] > 0.7]),
            },
            "health_assessment": self._assess_health(history),
        }
        return report

    def _get_status(self, metrics: Dict) -> str:
        """Determine status from metrics."""
        if metrics["health_score"] > 0.8:
            return "healthy"
        elif metrics["health_score"] > 0.6:
            return "degraded"
        else:
            return "critical"

    def _detect_trend(self, history: List[Dict]) -> str:
        """Detect trend direction."""
        if len(history) < 2:
            return "stable"

        latencies = [m["avg_latency"] for m in history]
        early = statistics.mean(latencies[:len(latencies)//2])
        late = statistics.mean(latencies[len(latencies)//2:])

        if late > early * 1.1:
            return "degrading"
        elif late < early * 0.9:
            return "improving"
        else:
            return "stable"

    def _assess_health(self, history: List[Dict]) -> Dict[str, str]:
        """Assess overall tool health."""
        avg_health = statistics.mean([m["health_score"] for m in history])
        error_rate = statistics.mean([m["error_rate"] for m in history])
        perfection = statistics.mean([m["avg_perfection"] for m in history])

        return {
            "overall": self._get_status({"health_score": avg_health}),
            "latency": "normal" if statistics.mean([m["avg_latency"] for m in history]) < 1.0 else "elevated",
            "reliability": "excellent" if error_rate < 0.01 else "poor" if error_rate > 0.1 else "good",
            "quality": "excellent" if perfection > 0.9 else "poor" if perfection < 0.7 else "good",
        }


# Global instances
report_store = ReportStore()
trend_analyzer = TrendAnalyzer(report_store)
report_generator = ReportGenerator(report_store, trend_analyzer)
