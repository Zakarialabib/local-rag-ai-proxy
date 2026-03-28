import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OperatorState:
    bridge_base: str
    lmstudio_base: str
    selected_model: str = ""
    role_map: Dict[str, str] = field(default_factory=dict)
    runtime_status: str = "Bridge stopped"
    loaded_models: List[str] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)
    request_counts: Dict[str, int] = field(default_factory=dict)
    probe_mode: str = "quick"
    last_response: Dict[str, Any] = field(default_factory=dict)
    last_chat: Dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    last_task_id: str = ""


class OperatorStateStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "web_console_state.json"
        self.events_path = self.root / "web_console_events.jsonl"
        self._lock = threading.RLock()

    def load(self, default: OperatorState) -> OperatorState:
        with self._lock:
            if not self.state_path.exists():
                return default
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                data = asdict(default)
                data.update(payload)
                return OperatorState(**data)
            except Exception:
                return default

    def save(self, state: OperatorState) -> None:
        with self._lock:
            self._atomic_write_json(self.state_path, asdict(state))

    def update(self, state: OperatorState, **changes: Any) -> OperatorState:
        with self._lock:
            for key, value in changes.items():
                setattr(state, key, value)
            self.save(state)
            return state

    def append_event(self, event_type: str, message: str, **data: Any) -> Dict[str, Any]:
        record = {
            "id": int(time.time() * 1000),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": event_type,
            "message": message,
            "data": data,
        }
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record

    def read_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.events_path.exists():
                return []
            lines = self.events_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
        events: List[Dict[str, Any]] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        return events

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
