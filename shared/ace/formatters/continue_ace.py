import json
from typing import Any, Dict, List


class ContinueACEFormatter:
    @staticmethod
    def format_stream_event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if event_type == "token":
            return {"type": "text", "content": data.get("token", "")}
        if event_type == "thinking":
            return {"type": "status", "content": data.get("status", "thinking")}
        if event_type == "checklist_question":
            return {
                "type": "function_call",
                "function_call": {
                    "name": "present_options",
                    "arguments": json.dumps(data, ensure_ascii=True),
                },
                "render": {"type": "checklist", "options": data.get("options", [])},
            }
        if event_type == "tool_call":
            return {
                "type": "function_call",
                "function_call": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=True),
                },
            }
        if event_type == "tool_result":
            return {"type": "tool_result", "content": data}
        if event_type == "context_injection":
            return {"type": "system", "content": "", "metadata": data}
        if event_type == "branch_exploration":
            return {"type": "status", "content": data.get("message", "Exploring branches"), "branches": data.get("branches", [])}
        if event_type == "final_output":
            return {"type": "final", "content": data.get("content", ""), "metadata": data}
        return {"type": event_type, "content": data}

    @staticmethod
    def create_final_response(
        *,
        session_id: str,
        content: str,
        reasoning: str,
        interventions: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        branches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "role": "assistant",
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "ace_enhancements": {
                "interventions_made": len(interventions),
                "parallel_branches_explored": len(branches),
                "context_injections": interventions,
            },
        }
