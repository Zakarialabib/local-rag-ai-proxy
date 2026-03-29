import json
import uuid
import structlog
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = structlog.get_logger("mcp_host")

@dataclass
class MCPSession:
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPHost:
    def __init__(self):
        self.sessions: Dict[str, MCPSession] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}

    def create_session(self, initial_context: Optional[Dict[str, Any]] = None) -> str:
        session_id = f"mcp_{uuid.uuid4().hex[:16]}"
        self.sessions[session_id] = MCPSession(id=session_id, context=initial_context or {})
        logger.info("mcp_session_created", session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[MCPSession]:
        return self.sessions.get(session_id)

    async def handle_rpc(self, session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard MCP JSON-RPC requests."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        session = self.get_session(session_id)
        if not session and method != "initialize":
            return self._rpc_error(request_id, -32001, "Session not found")

        try:
            if method == "initialize":
                return await self._handle_initialize(request_id, params)
            elif method == "tools/list":
                return await self._handle_tools_list(request_id, session)
            elif method == "tools/call":
                return await self._handle_tools_call(request_id, session, params)
            else:
                return self._rpc_error(request_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.error("mcp_rpc_failed", method=method, error=str(e))
            return self._rpc_error(request_id, -32603, str(e))

    async def _handle_initialize(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = self.create_session(params.get("client_info"))
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocol_version": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "server_info": {"name": "Qwen-Optimizing-Proxy", "version": "4.1.0"},
                "session_id": session_id
            }
        }

    async def _handle_tools_list(self, request_id: Any, session: MCPSession) -> Dict[str, Any]:
        # Return registered tools + built-in sequential thinking
        tools = [
            {
                "name": "sequential_thinking",
                "description": "Decompose complex tasks into logical steps. Mandatory for reasoning paths.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "The current line of reasoning"},
                        "next_steps": {"type": "array", "items": {"type": "string"}},
                        "requires_tools": {"type": "boolean"}
                    },
                    "required": ["thought", "next_steps"]
                }
            }
            # Additional tools will be registered here
        ]
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }

    async def _handle_tools_call(self, request_id: Any, session: MCPSession, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info("mcp_tool_call", session_id=session.id, tool=tool_name)
        
        # Log to session history
        session.tool_history.append({
            "ts": datetime.now().isoformat(),
            "tool": tool_name,
            "arguments": arguments
        })

        if tool_name == "sequential_thinking":
            result = {
                "status": "thought_recorded",
                "session_state": "processing",
                "plan": arguments.get("next_steps")
            }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
            }
        
        return self._rpc_error(request_id, -32601, f"Tool not implemented in Host: {tool_name}")

    def _rpc_error(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        }
