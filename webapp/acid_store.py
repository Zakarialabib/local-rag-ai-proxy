import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "acid_sessions.db"
SCHEMA_PATH = Path(__file__).parent / "acid_schema.sql"

class ACIDSessionStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            with open(SCHEMA_PATH, "r") as f:
                conn.executescript(f.read())

    def create_session(self, session_id: str, type_: str, workflow: str, meta: Dict[str, Any] = None, parent_id: str = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, type, workflow, meta, parent_id) VALUES (?, ?, ?, ?, ?)",
                (session_id, type_, workflow, json.dumps(meta or {}), parent_id)
            )

    def log_event(self, session_id: str, type_: str, payload: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events (session_id, type, payload) VALUES (?, ?, ?)",
                (session_id, type_, json.dumps(payload))
            )

    def save_checkpoint(self, session_id: str, state: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO checkpoints (session_id, state) VALUES (?, ?)",
                (session_id, json.dumps(state))
            )

    def create_branch(self, session_id: str, parent_checkpoint_id: int, label: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO branches (session_id, parent_checkpoint_id, label) VALUES (?, ?, ?)",
                (session_id, parent_checkpoint_id, label)
            )

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, ts, type, payload FROM events WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
            return [dict(id=row[0], ts=row[1], type=row[2], payload=json.loads(row[3])) for row in rows]

    # Add more methods for timeline, branching, diffing, etc.
