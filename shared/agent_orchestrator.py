import json
import re
import uuid
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from shared.agent_formatters import AiderFormatter, ConsoleFormatter, ContinueFormatter
from shared.agent_state import AgentSession, AgentSessionStore, AgentTurnRecord, utc_ts
from shared.agent_tools import AgentToolError, AgentToolRegistry


LLMCallable = Callable[[str, List[Dict[str, str]], Dict[str, Any]], Awaitable[Dict[str, Any]]]


class HierarchicalContextCompressor:
    def __init__(self, max_chars: int = 6500):
        self.max_chars = max_chars

    def compress(self, session: AgentSession, current_input: str) -> Dict[str, Any]:
        recent_turns = session.turns[-3:]
        episodic = session.episodic_memory[-6:]
        tool_outputs = session.tool_outputs[-4:]
        recent_text = "\n".join(
            f"[{item.get('agent')}] {item.get('output', '')[:280]}" for item in recent_turns
        )
        episodic_text = "\n".join(f"- {item}" for item in episodic)
        tool_text = "\n".join(
            f"- {item.get('tool')}: {self._shorten(json.dumps(item.get('result', {}), ensure_ascii=True))}"
            for item in tool_outputs
        )
        working_memory = self._shorten(session.working_memory or "")
        compressed = {
            "recent_turns": recent_turns,
            "episodic_summary": episodic_text[:1600],
            "tool_buffer": tool_text[:1200],
            "working_memory": working_memory[:1200],
            "query_focus": self._shorten(current_input, 800),
        }
        compressed["prompt_memory"] = "\n\n".join(
            part
            for part in (
                f"[RECENT]\n{recent_text}" if recent_text else "",
                f"[EPISODIC]\n{compressed['episodic_summary']}" if compressed["episodic_summary"] else "",
                f"[TOOLS]\n{compressed['tool_buffer']}" if compressed["tool_buffer"] else "",
                f"[WORKING]\n{compressed['working_memory']}" if compressed["working_memory"] else "",
            )
            if part
        )[: self.max_chars]
        compressed["stats"] = {
            "recent_turns": len(recent_turns),
            "episodic_items": len(episodic),
            "tool_items": len(tool_outputs),
            "prompt_chars": len(compressed["prompt_memory"]),
        }
        return compressed

    def _shorten(self, text: str, max_chars: int = 320) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return compact[:max_chars]


class AgentOrchestrator:
    AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
        "planner": {
            "model_role": "reasoning",
            "tools": ["file_read", "code_search", "mcp/list_models"],
            "max_turns": 3,
            "blocking": False,
            "system": "You are the planning agent. Produce concise plans and next actions in valid JSON.",
        },
        "researcher": {
            "model_role": "main",
            "tools": ["code_search", "doc_query", "mcp/model_stats"],
            "max_turns": 5,
            "blocking": False,
            "system": "You are the research agent. Gather facts, compare options, and stay concise in valid JSON.",
        },
        "builder": {
            "model_role": "main",
            "tools": ["file_read", "code_search", "shell_exec", "test_run"],
            "max_turns": 8,
            "blocking": False,
            "system": "You are the builder agent. Propose safe implementation steps and verification commands in valid JSON.",
        },
        "reviewer": {
            "model_role": "reasoning",
            "tools": ["file_read", "test_run", "code_search"],
            "max_turns": 2,
            "blocking": True,
            "system": "You are the reviewer agent. Identify risks, missing tests, and return approve or revise in valid JSON.",
        },
    }

    WORKFLOWS: Dict[str, Dict[str, Any]] = {
        "coding_sprint": {"start_agent": "planner", "max_turns": 10},
        "research_deep": {"start_agent": "researcher", "max_turns": 8},
        "debug_fix": {"start_agent": "planner", "max_turns": 9},
        "architect_review": {"start_agent": "planner", "max_turns": 7},
    }

    def __init__(
        self,
        *,
        session_store: AgentSessionStore,
        tool_registry: AgentToolRegistry,
        llm_generate: LLMCallable,
        workspace_root: Path,
    ):
        self.session_store = session_store
        self.tool_registry = tool_registry
        self.llm_generate = llm_generate
        self.workspace_root = workspace_root
        self.compressor = HierarchicalContextCompressor()
        self.embed_func: Optional[Callable] = None

    def create_session(self, *, workflow: str, role_map: Dict[str, str], cwd: str, tool_budget: int = 6) -> AgentSession:
        cfg = self.WORKFLOWS.get(workflow, self.WORKFLOWS["coding_sprint"])
        session = self.session_store.create_session(
            workflow=workflow,
            cwd=cwd,
            role_map=role_map,
            current_agent=cfg["start_agent"],
            max_turns=cfg["max_turns"],
            tool_budget=tool_budget,
        )
        checkpoint = self.session_store.create_checkpoint(session, checkpoint_id="initial")
        session.last_continue_token = self._continue_token(session)
        self.session_store.save_session(session)
        self.session_store.append_event(session.id, "session_ready", checkpoint_id=checkpoint)
        return session

    async def execute_turn(self, session_id: str, turn: Dict[str, Any]) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        turn_input = turn.get("input", {}) if isinstance(turn.get("input"), dict) else {}
        input_type = str(turn_input.get("type") or "user_request")
        input_content = str(turn_input.get("content") or "").strip()
        if not input_content and input_type != "tool_result":
            raise ValueError("Agent turn input content is required")

        if input_type == "tool_result":
            tool_result = turn.get("tool_result") or {}
            session.tool_outputs.append(tool_result)
            session.working_memory = self._merge_memory(session.working_memory, f"Tool result received: {json.dumps(tool_result)[:400]}")

        current_agent = session.current_agent
        agent_profile = self.AGENT_PROFILES[current_agent]
        compressed = self.compressor.compress(session, input_content)
        messages = self._build_messages(session, input_content, agent_profile, compressed)
        model_id = session.role_map.get(agent_profile["model_role"]) or session.role_map.get("main", "")
        generation = await self.llm_generate(model_id, messages, {"mode": "think" if agent_profile["model_role"] == "reasoning" else "fast"})
        parsed = self._parse_generation(generation, current_agent, session.workflow, input_content)
        tool_results = await self._execute_actions(session, parsed.get("actions", []))
        parsed["tool_results"] = tool_results

        session.turn_count += 1
        session.tool_calls_used += len([a for a in parsed.get("actions", []) if a.get("type") == "tool_call"])
        session.working_memory = self._merge_memory(
            session.working_memory,
            f"{current_agent}: {parsed.get('output', '')[:240]}",
        )
        if parsed.get("output"):
            session.episodic_memory.append(parsed["output"][:320])

        next_agent = self._next_agent(session, current_agent, parsed)
        session.current_agent = next_agent
        session.agent_stack = self._agent_stack(session.agent_stack, current_agent, parsed, next_agent)
        checkpoint_id = self.session_store.create_checkpoint(session)
        session.last_continue_token = self._continue_token(session)

        turn_record = AgentTurnRecord(
            id=f"turn_{uuid.uuid4().hex[:12]}",
            ts=utc_ts(),
            agent=current_agent,
            input_type=input_type,
            input_content=input_content,
            thinking=parsed.get("thinking", ""),
            output=parsed.get("output", ""),
            actions=parsed.get("actions", []),
            tool_results=tool_results,
            prompt_stats=generation.get("prompt_stats", {}),
            model_id=model_id,
            status="completed",
        )
        session.turns.append(asdict(turn_record))
        self.session_store.append_turn(session.id, turn_record)
        self.session_store.save_session(session)
        self.session_store.append_event(session.id, "turn_completed", agent=current_agent, next_agent=next_agent, checkpoint_id=checkpoint_id)

        result = {
            "session_id": session.id,
            "turn_result": {
                "agent": current_agent,
                "thinking": parsed.get("thinking", ""),
                "output": parsed.get("output", ""),
                "actions": parsed.get("actions", []),
                "tool_results": tool_results,
            },
            "state_update": {
                "next_agent": next_agent,
                "checkpoint_created": checkpoint_id,
                "agent_stack": list(session.agent_stack),
                "context_window": {
                    "used": compressed["stats"]["prompt_chars"],
                    "remaining": max(0, 8192 - compressed["stats"]["prompt_chars"]),
                    "compression_triggered": True,
                },
                "turn_count": session.turn_count,
                "tool_calls_used": session.tool_calls_used,
            },
            "for_client": {
                "display_mode": "structured",
                "continue_token": session.last_continue_token,
                "continue": ContinueFormatter.render({"session_id": session.id, "turn_result": {"agent": current_agent, "thinking": parsed.get("thinking", ""), "output": parsed.get("output", ""), "actions": parsed.get("actions", []), "tool_results": tool_results}, "state_update": {"checkpoint_created": checkpoint_id, "agent_stack": list(session.agent_stack)}}),
                "aider": AiderFormatter.render({"session_id": session.id, "turn_result": {"agent": current_agent, "output": parsed.get("output", ""), "actions": parsed.get("actions", [])}}),
                "console": ConsoleFormatter.render({"turn_result": {"agent": current_agent, "thinking": parsed.get("thinking", ""), "output": parsed.get("output", ""), "actions": parsed.get("actions", []), "tool_results": tool_results}, "state_update": {"checkpoint_created": checkpoint_id, "agent_stack": list(session.agent_stack)}}),
            },
        }
        return result

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        return {
            "session": asdict(session),
            "events": self.session_store.read_events(session_id, 200),
            "turns": self.session_store.read_turns(session_id, 50),
        }

    def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> Dict[str, Any]:
        session = self.session_store.restore_checkpoint(session_id, checkpoint_id)
        return {"session_id": session.id, "checkpoint_id": checkpoint_id, "state": asdict(session)}

    def branch_session(self, session_id: str, checkpoint_id: str) -> Dict[str, Any]:
        session = self.session_store.branch_session(session_id, checkpoint_id)
        return {"session_id": session.id, "parent_session_id": session.parent_session_id, "checkpoint_id": checkpoint_id}

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self.session_store.list_sessions()

    async def submit_tool_result(self, session_id: str, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        session.tool_outputs.append(tool_result)
        session.working_memory = self._merge_memory(session.working_memory, f"External tool result: {json.dumps(tool_result)[:400]}")
        self.session_store.save_session(session)
        self.session_store.append_event(session_id, "tool_result_received", tool=tool_result.get("tool"))
        return {"session_id": session_id, "accepted": True}

    async def execute_turn_stream(self, session_id: str, turn: Dict[str, Any]):
        session = self.session_store.load_session(session_id)
        self.session_store.append_event(session.id, "stream_start", agent=session.current_agent)
        yield {"type": "thinking", "data": {"agent": session.current_agent, "status": "starting"}}
        result = await self.execute_turn(session_id, turn)
        for action in result["turn_result"].get("actions", []):
            kind = "tool_call" if action.get("type") == "tool_call" else "agent_handoff"
            yield {"type": kind, "data": action}
        for item in result["turn_result"].get("tool_results", []):
            yield {"type": "tool_result", "data": item}
        yield {"type": "checkpoint", "data": {"checkpoint_id": result["state_update"]["checkpoint_created"]}}
        yield {"type": "final_output", "data": result}

    def _build_messages(self, session: AgentSession, input_content: str, agent_profile: Dict[str, Any], compressed: Dict[str, Any]) -> List[Dict[str, str]]:
        system = "\n\n".join(
            part
            for part in (
                agent_profile["system"],
                "Return valid JSON with keys: thinking, output, actions.",
                'Actions may contain tool_call, agent_spawn, checkpoint, or finish.',
                f"Allowed tools: {', '.join(agent_profile['tools'])}",
                compressed["prompt_memory"],
            )
            if part
        )
        user = (
            f"Workflow: {session.workflow}\n"
            f"Current agent: {session.current_agent}\n"
            f"Input: {input_content}\n"
            "Stay concise. Use tool_call actions only when necessary."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_generation(self, generation: Dict[str, Any], current_agent: str, workflow: str, input_content: str) -> Dict[str, Any]:
        output = generation.get("content", "") or generation.get("output_text", "") or ""
        parsed = self._extract_json(output)
        if parsed:
            parsed.setdefault("thinking", "")
            parsed.setdefault("output", output if not parsed.get("output") else parsed["output"])
            parsed.setdefault("actions", [])
            return parsed
        return self._heuristic_fallback(current_agent, workflow, input_content, output)

    async def _execute_actions(self, session: AgentSession, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for action in actions:
            if action.get("type") != "tool_call":
                continue
            if session.tool_calls_used >= session.tool_budget:
                results.append({"tool": action.get("tool"), "error": "tool_budget_exhausted"})
                continue
            try:
                result = await self.tool_registry.execute(
                    action.get("tool", ""),
                    action.get("params", {}),
                    {"cwd": session.cwd, "session_id": session.id, "agent": session.current_agent},
                )
                
                # RECURSIVE VALIDATION: Check for semantic drift
                if self.embed_func and "content" in result:
                    drift_score = await self._check_drift(action.get("reason", ""), result["content"])
                    result["validation"] = {"score": drift_score, "status": "passed" if drift_score > 0.4 else "drift_detected"}
                    if drift_score < 0.4:
                        self.session_store.append_event(session.id, "validation_warning", tool=action.get("tool"), score=drift_score)

                session.tool_outputs.append({"tool": action.get("tool"), "result": result, "agent": session.current_agent})
                self.session_store.append_event(session.id, "tool_result", tool=action.get("tool"), ok=True)
                results.append(result)
            except (AgentToolError, Exception) as exc:
                error = {"tool": action.get("tool"), "error": str(exc)}
                session.tool_outputs.append({"tool": action.get("tool"), "result": error, "agent": session.current_agent})
                self.session_store.append_event(session.id, "tool_result", tool=action.get("tool"), ok=False, error=str(exc))
                results.append(error)
        return results

    async def _check_drift(self, intent: str, content: str) -> float:
        """Measure semantic similarity between intent and result content."""
        if not intent or not content or not self.embed_func:
            return 1.0
        try:
            # Sliced 512-dim embedding comparison
            embs = await self.embed_func([intent, content[:2000]])
            if len(embs) < 2: return 1.0
            
            vec_a = embs[0] / (np.linalg.norm(embs[0]) + 1e-9)
            vec_b = embs[1] / (np.linalg.norm(embs[1]) + 1e-9)
            return float(np.dot(vec_a, vec_b))
        except Exception:
            return 1.0

    def _heuristic_fallback(self, current_agent: str, workflow: str, input_content: str, raw_output: str) -> Dict[str, Any]:
        actions: List[Dict[str, Any]] = []
        lower = input_content.lower()
        if current_agent == "planner":
            actions.append(
                {
                    "type": "agent_spawn",
                    "agent": "researcher" if workflow == "research_deep" else "builder",
                    "task": "Continue based on the generated plan",
                    "inherit_context": True,
                }
            )
        elif current_agent == "builder" and any(token in lower for token in ("test", "verify", "pytest")):
            actions.append(
                {
                    "type": "tool_call",
                    "tool": "test_run",
                    "params": {"target": ""},
                    "reason": "Run a safe verification command",
                }
            )
        output = raw_output.strip() or f"{current_agent.title()} processed the request."
        return {
            "thinking": f"{current_agent} analyzed the request and generated a concise next step.",
            "output": output,
            "actions": actions,
        }

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return None
        candidates = [stripped]
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def _next_agent(self, session: AgentSession, current_agent: str, parsed: Dict[str, Any]) -> str:
        for action in parsed.get("actions", []):
            if action.get("type") == "agent_spawn" and action.get("agent"):
                return action["agent"]
        profile = self.AGENT_PROFILES[current_agent]
        if session.turn_count + 1 >= profile["max_turns"] and len(session.agent_stack) > 1:
            return session.agent_stack[-2]
        if current_agent == "planner":
            return "builder" if session.workflow != "research_deep" else "researcher"
        if current_agent == "builder":
            return "reviewer"
        if current_agent == "reviewer":
            return "builder" if profile["blocking"] else "planner"
        return current_agent

    def _agent_stack(self, current_stack: List[str], current_agent: str, parsed: Dict[str, Any], next_agent: str) -> List[str]:
        stack = list(current_stack)
        spawned = next((a for a in parsed.get("actions", []) if a.get("type") == "agent_spawn" and a.get("agent")), None)
        if spawned:
            stack.append(spawned["agent"])
            return stack[-6:]
        if stack and stack[-1] != next_agent:
            stack.append(next_agent)
        return stack[-6:]

    def _continue_token(self, session: AgentSession) -> str:
        token = json.dumps({"session_id": session.id, "turn_count": session.turn_count})
        return token.encode("utf-8").hex()

    def _merge_memory(self, working_memory: str, fragment: str) -> str:
        compact = " | ".join(part for part in [working_memory.strip(), fragment.strip()] if part).strip(" |")
        return compact[:1800]
