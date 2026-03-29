#!/usr/bin/env python
"""ACID session persistence test."""

from webapp.acid_store import ACIDSessionStore
import json

def test_acid():
    store = ACIDSessionStore()
    
    # Test 1: Create session
    session_id = "test_session_001"
    session_type = "agent"
    workflow = "coding_sprint"
    meta = {"tool_budget": 5, "user": "test"}
    
    try:
        store.create_session(session_id, session_type, workflow, meta=meta)
        print("✓ Created ACID session")
    except Exception as e:
        print(f"✗ Failed to create session: {e}")
        return
    
    # Test 2: Log events
    events = [
        ("tool_call", {"tool": "grep", "args": ["pattern"]}),
        ("tool_success", {"tool": "grep", "duration_ms": 125, "output_len": 42}),
        ("tool_call", {"tool": "file_read", "args": ["path"]}),
        ("tool_error", {"tool": "file_read", "error": "permission denied"}),
    ]
    
    for event_type, data in events:
        try:
            store.log_event(session_id, event_type, data)
        except Exception as e:
            print(f"✗ Failed to log {event_type}: {e}")
    
    print(f"✓ Logged {len(events)} events")
    
    # Test 3: Retrieve events
    try:
        retrieved = store.get_session_events(session_id)
        print(f"✓ Retrieved {len(retrieved)} events")
        if retrieved:
            print(f"  └─ Event types: {[e.get('type') for e in retrieved[:3]]}")
    except Exception as e:
        print(f"✗ Failed to retrieve events: {e}")
    
    # Test 4: Query sessions
    try:
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            rows = conn.execute(
                "SELECT id, type, workflow, status FROM sessions ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        print(f"✓ Found {len(rows)} sessions in database")
        if rows:
            print(f"  └─ Most recent: {rows[0][0]} ({rows[0][1]})")
    except Exception as e:
        print(f"✗ Query failed: {e}")

if __name__ == '__main__':
    print("\n=== ACID PERSISTENCE TEST ===\n")
    test_acid()
    print("\n=== TESTS COMPLETE ===\n")
