import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class AgentAction:
    type: str
    tool: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    agent: str = ""
    task: str = ""
    inherit_context: bool = True
    id: str = ""


@dataclass
class AgentTurnRecord:
    id: str
    ts: str
    agent: str
    input_type: str
    input_content: str
    thinking: str = ""
    output: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    prompt_stats: Dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    status: str = "completed"


@dataclass
class AgentSession:
    id: str
    workflow: str
    current_agent: str
    agent_stack: List[str]
    role_map: Dict[str, str]
    cwd: str
    created_at: str
    updated_at: str
    turn_count: int = 0
    max_turns: int = 10
    tool_budget: int = 6
    tool_calls_used: int = 0
    status: str = "ready"
    working_memory: str = ""
    episodic_memory: List[str] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    parent_session_id: Optional[str] = None
    branch_from_checkpoint: Optional[str] = None
    last_continue_token: str = ""


class AgentSessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create_session(
        self,
        *,
        workflow: str,
        cwd: str,
        role_map: Dict[str, str],
        current_agent: str = "planner",
        max_turns: int = 10,
        tool_budget: int = 6,
        parent_session_id: Optional[str] = None,
        branch_from_checkpoint: Optional[str] = None,
    ) -> AgentSession:
        now = utc_ts()
        session = AgentSession(
            id=f"ags_{uuid.uuid4().hex[:16]}",
            workflow=workflow,
            current_agent=current_agent,
            agent_stack=[current_agent],
            role_map=dict(role_map),
            cwd=cwd,
            created_at=now,
            updated_at=now,
            max_turns=max_turns,
            tool_budget=tool_budget,
            parent_session_id=parent_session_id,
            branch_from_checkpoint=branch_from_checkpoint,
        )
        self.save_session(session)
        self.append_event(session.id, "session_created", workflow=workflow, current_agent=current_agent)
        return session

    def load_session(self, session_id: str) -> AgentSession:
        path = self._session_dir(session_id) / "state.json"
        if not path.exists():
            raise FileNotFoundError(f"Agent session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentSession(**data)

    def save_session(self, session: AgentSession) -> None:
        with self._lock:
            session.updated_at = utc_ts()
            session_dir = self._session_dir(session.id)
            session_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(session_dir / "state.json", asdict(session))

    def append_turn(self, session_id: str, turn: AgentTurnRecord) -> None:
        with self._lock:
            session_dir = self._session_dir(session_id)
            turns_path = session_dir / "turns.jsonl"
            with turns_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(turn), ensure_ascii=True) + "\n")

    def read_turns(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        path = self._session_dir(session_id) / "turns.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
        output: List[Dict[str, Any]] = []
        for line in lines:
            try:
                output.append(json.loads(line))
            except Exception:
                continue
        return output

    def append_event(self, session_id: str, event_type: str, **data: Any) -> Dict[str, Any]:
        record = {
            "id": int(time.time() * 1000),
            "ts": utc_ts(),
            "type": event_type,
            "data": data,
        }
        with self._lock:
            events_path = self._session_dir(session_id) / "events.jsonl"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record

    def read_events(self, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        path = self._session_dir(session_id) / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
        output: List[Dict[str, Any]] = []
        for line in lines:
            try:
                output.append(json.loads(line))
            except Exception:
                continue
        return output

    def create_checkpoint(self, session: AgentSession, checkpoint_id: Optional[str] = None) -> str:
        checkpoint = checkpoint_id or f"chk_{uuid.uuid4().hex[:12]}"
        with self._lock:
            session_dir = self._session_dir(session.id)
            checkpoint_dir = session_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(checkpoint_dir / f"{checkpoint}.json", asdict(session))
        session.checkpoints.append(checkpoint)
        self.save_session(session)
        self.append_event(session.id, "checkpoint_created", checkpoint_id=checkpoint)
        return checkpoint

    def restore_checkpoint(self, session_id: str, checkpoint_id: str) -> AgentSession:
        path = self._session_dir(session_id) / "checkpoints" / f"{checkpoint_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        session = AgentSession(**data)
        session.status = "ready"
        self.save_session(session)
        self.append_event(session_id, "checkpoint_restored", checkpoint_id=checkpoint_id)
        return session

    def branch_session(self, session_id: str, checkpoint_id: str) -> AgentSession:
        source = self.restore_checkpoint(session_id, checkpoint_id)
        branch = self.create_session(
            workflow=source.workflow,
            cwd=source.cwd,
            role_map=source.role_map,
            current_agent=source.current_agent,
            max_turns=source.max_turns,
            tool_budget=source.tool_budget,
            parent_session_id=source.id,
            branch_from_checkpoint=checkpoint_id,
        )
        branch.turn_count = source.turn_count
        branch.tool_calls_used = source.tool_calls_used
        branch.working_memory = source.working_memory
        branch.episodic_memory = list(source.episodic_memory)
        branch.tool_outputs = list(source.tool_outputs)
        branch.turns = list(source.turns)
        branch.agent_stack = list(source.agent_stack)
        branch.checkpoints = list(source.checkpoints)
        self.save_session(branch)
        self.append_event(branch.id, "session_branched", parent_session_id=source.id, checkpoint_id=checkpoint_id)
        return branch

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for entry in sorted(self.root.glob("ags_*"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            state_path = entry / "state.json"
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "id": data.get("id"),
                        "workflow": data.get("workflow"),
                        "current_agent": data.get("current_agent"),
                        "status": data.get("status"),
                        "updated_at": data.get("updated_at"),
                        "turn_count": data.get("turn_count"),
                    }
                )
            except Exception:
                continue
        return sessions

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
