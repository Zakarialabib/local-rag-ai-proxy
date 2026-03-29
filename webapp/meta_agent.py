import threading
import queue
import time
from typing import Any, Callable, Dict, List, Optional

class MetaAgentEvent:
    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_type = event_type
        self.payload = payload
        self.timestamp = time.time()

class MetaAgent:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.handlers: Dict[str, Callable[[MetaAgentEvent], None]] = {}

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self.handle_event(event)
            except queue.Empty:
                continue

    def handle_event(self, event: MetaAgentEvent):
        handler = self.handlers.get(event.event_type)
        if handler:
            handler(event)
        # else: ignore or log

    def register_handler(self, event_type: str, handler: Callable[[MetaAgentEvent], None]):
        self.handlers[event_type] = handler

    def emit(self, event_type: str, payload: Dict[str, Any]):
        event = MetaAgentEvent(event_type, payload)
        self.event_queue.put(event)

# Singleton instance for app integration
meta_agent = MetaAgent()
