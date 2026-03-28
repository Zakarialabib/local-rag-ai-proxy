from __future__ import annotations

import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List

import httpx

from shared.ace.context_injector import RealTimeContextInjector
from shared.ace.formatters.continue_ace import ContinueACEFormatter
from shared.ace.pattern_detector import StreamingPatternDetector
from shared.ace.qwen_tool_native import QwenToolNativeHandler
from shared.ace.session_store import ACESession, ACESessionStore, utc_ts
from shared.ace.stream_brancher import StreamBranchingEngine
from shared.agent_tools import AgentToolRegistry


PrepareBodyCallable = Callable[[Dict[str, Any], str], Awaitable[Dict[str, Any]]]


class ACEOrchestrator:
    def __init__(
        self,
        *,
        session_store: ACESessionStore,
        tool_registry: AgentToolRegistry,
        lmstudio_base: str,
        prepare_body: PrepareBodyCallable,
    ):
        self.session_store = session_store
        self.tool_registry = tool_registry
        self.lmstudio_base = lmstudio_base.rstrip("/")
        self.prepare_body = prepare_body
        self.detector = StreamingPatternDetector()
        self.injector = RealTimeContextInjector(tool_registry.bridge_base)
        self.brancher = StreamBranchingEngine()
        self.tool_handler = QwenToolNativeHandler(tool_registry)

    def create_session(self, *, prompt: str, mode: str, model: str, role_map: Dict[str, str], docs: List[str]) -> ACESession:
        return self.session_store.create_session(prompt=prompt, mode=mode, model=model, role_map=role_map, docs=docs)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self.session_store.list_sessions()

    def get_trace(self, session_id: str) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        return {
            "session": {
                "id": session.id,
                "status": session.status,
                "mode": session.mode,
                "model": session.model,
                "updated_at": session.updated_at,
            },
            "trace": session.trace,
            "selections": session.selections,
            "final_text": session.final_text,
            "reasoning_text": session.reasoning_text,
        }

    def record_selection(self, session_id: str, option_id: str, label: str = "") -> Dict[str, Any]:
        selection = self.session_store.record_selection(session_id, option_id, label)
        self.session_store.append_trace(session_id, {"ts": utc_ts(), "type": "selection_recorded", "data": selection})
        return selection

    async def generate(self, request: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("ACE prompt is required")
        role_map = request.get("role_map") or {}
        mode = str(request.get("mode") or "think")
        model = str(request.get("model") or role_map.get("reasoning") or role_map.get("main") or "")
        docs = [str(item) for item in (request.get("docs") or []) if str(item).strip()]
        session_id = str(request.get("session_id") or "")
        session = self.session_store.load_session(session_id) if session_id else self.create_session(
            prompt=prompt,
            mode=mode,
            model=model,
            role_map=role_map,
            docs=docs,
        )
        session.status = "running"
        session.prompt = prompt
        session.docs = docs
        session.metadata["last_request"] = {"mode": mode, "docs": len(docs)}
        self.session_store.save_session(session)

        branches = self.brancher.suggest_branches(prompt, mode)
        branch_event = {"message": f"Exploring {len(branches)} ACE paths", "branches": branches}
        self.session_store.append_trace(session.id, {"ts": utc_ts(), "type": "branch_exploration", "data": branch_event})
        yield self._event("branch_exploration", branch_event, session.id)

        system_prompt = (
            "You are operating inside ACE. Stay concise, minimize tokens, ground claims, and emit Qwen native "
            "tool calls only when needed using <tool_call>{...}</tool_call>."
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": request.get("temperature", 0.3),
            "max_tokens": int(request.get("max_output_tokens") or 1024),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_k", 40),
            "repeat_penalty": request.get("repeat_penalty", 1.05),
            "extra_body": {"mode": mode},
        }
        domain = "code" if any(token in prompt.lower() for token in ("code", "python", "debug", "build")) else "reasoning"
        body = await self.prepare_body(body, domain)

        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        interventions: List[Dict[str, Any]] = []
        seen_patterns: set[str] = set()
        seen_tool_payloads: set[str] = set()
        emitted_checklist = False
        token_position = 0

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.lmstudio_base}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"ACE upstream failed with HTTP {response.status_code}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        payload = json.loads(data_str)
                    except Exception:
                        continue
                    choice = (payload.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    chunk = ""
                    if delta.get("reasoning_content"):
                        chunk = str(delta["reasoning_content"])
                        reasoning_chunks.append(chunk)
                        token_position += max(1, len(chunk.split()))
                        yield self._event("thinking", {"token": chunk, "status": "reasoning"}, session.id)
                    if delta.get("content"):
                        chunk = str(delta["content"])
                        content_chunks.append(chunk)
                        token_position += max(1, len(chunk.split()))
                        yield self._event("token", {"token": chunk}, session.id)

                    combined = "".join(reasoning_chunks + content_chunks)
                    for detection in self.detector.detect(combined, position=token_position, seen=seen_patterns):
                        record = {
                            "ts": utc_ts(),
                            "type": "pattern_detected",
                            "data": {
                                "pattern": detection.type,
                                "confidence": detection.confidence,
                                "position": detection.position,
                                "action": detection.suggested_action,
                            },
                        }
                        self.session_store.append_trace(session.id, record)
                        intervention = await self.injector.engineer_context(
                            detection,
                            prompt=prompt,
                            docs=docs,
                            session_context={"cwd": request.get("cwd", ""), "workflow": request.get("workflow", ""), "retrieval": request.get("retrieval") or {}},
                        )
                        if intervention:
                            payload = {
                                "type": intervention.type,
                                "trigger": intervention.trigger_pattern,
                                "content": intervention.content,
                                "priority": intervention.priority,
                            }
                            interventions.append(payload)
                            self.session_store.append_trace(session.id, {"ts": utc_ts(), "type": "intervention", "data": payload})
                            if intervention.type == "checklist_question" and not emitted_checklist:
                                emitted_checklist = True
                                yield self._event("checklist_question", intervention.content, session.id)
                            else:
                                yield self._event("context_injection", payload, session.id)

                    tool_calls = self.tool_handler.extract_tool_calls(combined, seen_payloads=seen_tool_payloads)
                    if tool_calls:
                        for call in tool_calls:
                            yield self._event("tool_call", {"name": call["name"], "arguments": call["arguments"]}, session.id)
                        results = await self.tool_handler.execute_calls(
                            tool_calls,
                            session_context={"cwd": request.get("cwd", ""), "session_id": session.id, "agent": "ace"},
                        )
                        for result in results:
                            payload = {"tool": result["tool"], "result": result["result"]}
                            self.session_store.append_trace(session.id, {"ts": utc_ts(), "type": "tool_result", "data": payload})
                            yield self._event("tool_result", payload, session.id)

        session.status = "completed"
        session.final_text = "".join(content_chunks)
        session.reasoning_text = "".join(reasoning_chunks)
        session.metadata["interventions"] = len(interventions)
        self.session_store.save_session(session)
        final = ContinueACEFormatter.create_final_response(
            session_id=session.id,
            content=session.final_text,
            reasoning=session.reasoning_text,
            interventions=interventions,
            tool_calls=[],
            branches=branches,
        )
        self.session_store.append_trace(session.id, {"ts": utc_ts(), "type": "final_output", "data": final})
        yield self._event("final_output", final, session.id)

    def _event(self, event_type: str, data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        return {
            "type": event_type,
            "data": data,
            "session_id": session_id,
            "continue": ContinueACEFormatter.format_stream_event(event_type, data),
        }
