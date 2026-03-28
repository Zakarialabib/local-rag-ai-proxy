import json
from typing import Any, Dict, List


class ContinueFormatter:
    @staticmethod
    def render(turn_response: Dict[str, Any]) -> Dict[str, Any]:
        turn = turn_response.get("turn_result", {})
        actions = turn.get("actions", []) or []
        content_parts = []
        thinking = turn.get("thinking", "")
        output = turn.get("output", "")
        if thinking:
            content_parts.append(f"<thinking>\n{thinking}\n</thinking>")
        if output:
            content_parts.append(output)
        return {
            "role": "assistant",
            "content": "\n\n".join(part for part in content_parts if part),
            "function_calls": [
                {
                    "name": action.get("tool"),
                    "arguments": json.dumps(action.get("params", {})),
                }
                for action in actions
                if action.get("type") == "tool_call"
            ],
            "metadata": {
                "session_id": turn_response.get("session_id"),
                "agent": turn.get("agent"),
                "agent_chain": turn_response.get("state_update", {}).get("agent_stack", []),
                "checkpoint": turn_response.get("state_update", {}).get("checkpoint_created"),
                "tool_results": turn.get("tool_results", []),
            },
        }


class AiderFormatter:
    @staticmethod
    def render(turn_response: Dict[str, Any]) -> str:
        turn = turn_response.get("turn_result", {})
        body = turn.get("output", "")
        meta = {
            "session_id": turn_response.get("session_id"),
            "agent": turn.get("agent"),
            "actions": turn.get("actions", []),
        }
        return f"{body}\n\n```json\n{json.dumps(meta, indent=2)}\n```"


class ConsoleFormatter:
    @staticmethod
    def render(turn_response: Dict[str, Any]) -> Dict[str, Any]:
        turn = turn_response.get("turn_result", {})
        return {
            "agent": turn.get("agent"),
            "thinking": turn.get("thinking", ""),
            "output": turn.get("output", ""),
            "actions": turn.get("actions", []),
            "tool_results": turn.get("tool_results", []),
            "state_update": turn_response.get("state_update", {}),
        }
