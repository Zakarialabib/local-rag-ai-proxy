from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class ACESession:
    id: str
    created_at: str
    updated_at: str
    prompt: str
    mode: str
    model: str
    role_map: Dict[str, str]
    docs: List[str] = field(default_factory=list)
    status: str = "ready"
    final_text: str = ""
    reasoning_text: str = ""
    selections: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ACESessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create_session(self, *, prompt: str, mode: str, model: str, role_map: Dict[str, str], docs: List[str]) -> ACESession:
        now = utc_ts()
        session = ACESession(
            id=f"ace_{uuid.uuid4().hex[:16]}",
            created_at=now,
            updated_at=now,
            prompt=prompt,
            mode=mode,
            model=model,
            role_map=dict(role_map),
            docs=list(docs),
        )
        self.save_session(session)
        self.append_trace(
            session.id,
            {"ts": now, "type": "session_created", "data": {"mode": mode, "model": model, "docs": len(docs)}},
        )
        return session

    def load_session(self, session_id: str) -> ACESession:
        path = self._session_dir(session_id) / "state.json"
        if not path.exists():
            raise FileNotFoundError(f"ACE session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ACESession(**data)

    def save_session(self, session: ACESession) -> None:
        with self._lock:
            session.updated_at = utc_ts()
            session_dir = self._session_dir(session.id)
            session_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(session_dir / "state.json", asdict(session))

    def append_trace(self, session_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            session = self.load_session(session_id)
            session.trace.append(record)
            session.trace = session.trace[-300:]
            self.save_session(session)

    def record_selection(self, session_id: str, option_id: str, label: str = "") -> Dict[str, Any]:
        with self._lock:
            session = self.load_session(session_id)
            selection = {"ts": utc_ts(), "option_id": option_id, "label": label}
            session.selections.append(selection)
            self.save_session(session)
            return selection

    def list_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for entry in sorted(self.root.glob("ace_*"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            state_path = entry / "state.json"
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sessions.append(
                {
                    "id": data.get("id"),
                    "mode": data.get("mode"),
                    "model": data.get("model"),
                    "status": data.get("status"),
                    "updated_at": data.get("updated_at"),
                }
            )
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
