from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from shared.agent_tools import AgentToolRegistry


class QwenToolNativeHandler:
    TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def __init__(self, tool_registry: AgentToolRegistry):
        self.tool_registry = tool_registry

    def extract_tool_calls(self, text: str, seen_payloads: set[str] | None = None) -> List[Dict[str, Any]]:
        seen_payloads = seen_payloads or set()
        calls: List[Dict[str, Any]] = []
        for match in self.TOOL_BLOCK_RE.finditer(text or ""):
            payload = match.group(1).strip()
            if payload in seen_payloads:
                continue
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if not isinstance(parsed, dict) or not parsed.get("name"):
                continue
            seen_payloads.add(payload)
            calls.append(
                {
                    "name": str(parsed.get("name")),
                    "arguments": parsed.get("arguments") or {},
                    "raw": payload,
                }
            )
        return calls

    async def execute_calls(
        self,
        calls: List[Dict[str, Any]],
        *,
        session_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in calls:
            result = await self.tool_registry.execute(item["name"], item.get("arguments") or {}, session_context)
            results.append({"tool": item["name"], "arguments": item.get("arguments") or {}, "result": result})
        return results
