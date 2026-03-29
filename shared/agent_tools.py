import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx


class AgentToolError(RuntimeError):
    pass


class AgentToolRegistry:
    SAFE_SHELL_PREFIXES = (
        "rg ",
        "rg.exe ",
        "pytest",
        "python -m py_compile",
        "python -m unittest",
    )

    def __init__(self, *, root_dir: Path, bridge_base: str, mcp_base: str = ""):
        self.root_dir = root_dir.resolve()
        self.bridge_base = bridge_base.rstrip("/")
        default_mcp = mcp_base or os.getenv("MCP_BASE_URL", "http://127.0.0.1:8081")
        self.mcp_base = default_mcp.rstrip("/")

    async def execute(self, tool_name: str, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        import time
        from webapp.tool_ecosystem import tool_health_monitor, ToolCategory
        from webapp.perfection_index import perfection_tracker
        from webapp.tool_analytics import analytics_store, anomaly_detector, remediation_engine
        from webapp.alerting_system import alert_manager
        from webapp.remediation_callbacks import remediation_callback_registry

        # Register tool if not already registered
        tool_health_monitor.register_tool(tool_name, ToolCategory.EXECUTION, cluster_id="default")
        
        # Check if tool can execute (circuit breaker, rate limiter)
        can_run, reason = tool_health_monitor.can_execute_tool(tool_name)
        if not can_run:
            error_msg = f"Tool {tool_name} is {reason}"
            analytics_store.log_lifecycle(tool_name, "execute_blocked", 0.0, False, error_msg)
            raise AgentToolError(f"Tool {tool_name} is isolated ({reason})")

        start_time = time.time()
        success = False
        error_msg = None
        result = None
        
        try:
            if tool_name == "file_read":
                result = await self._file_read(params, session_context)
            elif tool_name == "code_search":
                result = await self._code_search(params, session_context)
            elif tool_name == "shell_exec":
                result = await self._shell_exec(params, session_context)
            elif tool_name == "test_run":
                result = await self._test_run(params, session_context)
            elif tool_name == "doc_query":
                result = await self._doc_query(params, session_context)
            elif tool_name.startswith("mcp/"):
                result = await self._mcp_query(tool_name, params)
            else:
                raise AgentToolError(f"Unsupported agent tool: {tool_name}")
            success = True
            return result
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # Phase 1: Record health metrics
            tool_health_monitor.record_execution(tool_name, latency_ms, success, error_msg)
            
            # Phase 1: Record perfection index metrics
            perfection_tracker.record_tool_execution(
                tool_name,
                latency_ms,
                success,
                session_id=session_context.get("session_id"),
                agent=session_context.get("agent"),
                error=error_msg,
            )
            
            # Phase 1: Log to analytics store
            analytics_store.log_lifecycle(
                tool_name,
                "execute",
                latency_ms,
                success,
                error_msg,
                metadata={
                    "params": str(params)[:200],
                    "session_id": session_context.get("session_id"),
                    "agent": session_context.get("agent"),
                }
            )
            
            # Phase 1: Record execution for anomaly detection
            anomaly_detector.record_execution(tool_name, latency_ms, error=not success)
            
            # Phase 2: Detect anomalies and trigger remediation
            anomalies = anomaly_detector.detect_anomalies(tool_name)
            if anomalies:
                actions = remediation_engine.trigger_remediation(tool_name, anomalies)
                
                # Phase 2: Execute registered callbacks
                for action in actions:
                    try:
                        if action.action in remediation_callback_registry.callbacks:
                            callback = remediation_callback_registry.callbacks[action.action]
                            callback(tool_name, action.reason)
                    except Exception as e:
                        pass  # Log but don't raise
                
                # Phase 4: Send alerts for high-severity anomalies
                for anomaly in anomalies:
                    if anomaly.severity >= 0.7:  # High severity
                        alert_manager.send_alert(
                            tool_name=tool_name,
                            anomaly_type=anomaly.anomaly_type.value,
                            severity=anomaly.severity,
                            message=f"Critical anomaly in {tool_name}: {anomaly.anomaly_type.value}",
                            details=anomaly.details
                        )


    async def _file_read(self, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        path = self._resolve_path(params.get("path") or "", session_context)
        if not path.exists() or not path.is_file():
            raise AgentToolError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        max_chars = int(params.get("max_chars") or 8000)
        return {
            "tool": "file_read",
            "path": str(path),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    async def _code_search(self, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        pattern = str(params.get("pattern") or "").strip()
        if not pattern:
            raise AgentToolError("Missing search pattern")
        cwd = self._resolve_path(session_context.get("cwd") or ".", session_context)
        proc = await asyncio.create_subprocess_exec(
            "rg",
            "-n",
            pattern,
            str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "tool": "code_search",
            "pattern": pattern,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="ignore")[:12000],
            "stderr": stderr.decode("utf-8", errors="ignore")[:4000],
        }

    async def _shell_exec(self, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        command = str(params.get("command") or "").strip()
        if not command:
            raise AgentToolError("Missing shell command")
        if not command.startswith(self.SAFE_SHELL_PREFIXES):
            raise AgentToolError("Shell command is outside the safe allow-list")
        cwd = self._resolve_path(session_context.get("cwd") or ".", session_context)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "tool": "shell_exec",
            "command": command,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="ignore")[:12000],
            "stderr": stderr.decode("utf-8", errors="ignore")[:4000],
        }

    async def _test_run(self, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        target = str(params.get("target") or "").strip()
        command = f"pytest {target}".strip()
        return await self._shell_exec({"command": command}, session_context)

    async def _doc_query(self, params: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        query = str(params.get("query") or "").strip()
        docs = params.get("docs") or []
        retrieval = params.get("retrieval") or {}
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.bridge_base}/api/v1/retrieve",
                json={"query": query, "docs": docs, "retrieval": retrieval},
            )
            response.raise_for_status()
            result = response.json()
        return {"tool": "doc_query", "query": query, "result": result}

    async def _mcp_query(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        external_tool = tool_name.split("/", 1)[1]
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.mcp_base}/tools/call",
                json={"name": external_tool, "arguments": params},
            )
            response.raise_for_status()
            return {"tool": tool_name, "result": response.json()}

    def _resolve_path(self, raw_path: str, session_context: Dict[str, Any]) -> Path:
        base = Path(session_context.get("cwd") or self.root_dir)
        if not base.is_absolute():
            base = (self.root_dir / base).resolve()
        path = Path(raw_path)
        if not path.is_absolute():
            path = (base / path).resolve()
        if not str(path).lower().startswith(str(self.root_dir).lower()):
            raise AgentToolError(f"Path escapes workspace root: {path}")
        return path
