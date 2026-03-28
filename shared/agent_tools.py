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
        if tool_name == "file_read":
            return await self._file_read(params, session_context)
        if tool_name == "code_search":
            return await self._code_search(params, session_context)
        if tool_name == "shell_exec":
            return await self._shell_exec(params, session_context)
        if tool_name == "test_run":
            return await self._test_run(params, session_context)
        if tool_name == "doc_query":
            return await self._doc_query(params, session_context)
        if tool_name.startswith("mcp/"):
            return await self._mcp_query(tool_name, params)
        raise AgentToolError(f"Unsupported agent tool: {tool_name}")

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
