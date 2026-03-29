# Multi-channel SSE topology for real-time dashboard streaming
# Provides hot/warm/cold event streaming plus alerts and health status channels

import json
import threading
import time
from queue import Queue
from typing import Any, Dict

from flask import Response, stream_with_context


class SSEChannel:
    """Thread-safe SSE channel for broadcasting events."""
    
    def __init__(self, name: str, buffer_size: int = 100):
        self.name = name
        self.listeners = []
        self.queue = Queue(maxsize=buffer_size)
        self.lock = threading.RLock()

    def publish(self, data: Any, event_type: str = "message"):
        """Publish data to all subscribers."""
        try:
            message = {
                "timestamp": time.time(),
                "event": event_type,
                "channel": self.name,
                "data": data,
            }
            with self.lock:
                # Try to queue for future subscribers
                try:
                    self.queue.put_nowait(message)
                except:
                    pass
                # Send to existing subscribers
                for listener in self.listeners[:]:
                    try:
                        listener.put_nowait(message)
                    except:
                        self.listeners.remove(listener)
        except Exception as e:
            pass

    def subscribe(self):
        """Subscribe to channel events."""
        q = Queue()
        with self.lock:
            self.listeners.append(q)
        return q

    def unsubscribe(self, q: Queue):
        """Unsubscribe from channel."""
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)


# Global SSE channels
sse_channels = {
    "hot": SSEChannel("hot"),       # Real-time alerts and anomalies
    "warm": SSEChannel("warm"),     # Health status updates
    "cold": SSEChannel("cold"),     # Periodic reports and metrics
    "alerts": SSEChannel("alerts"), # Alert-specific streaming
    "health": SSEChannel("health"), # Health status streaming
}


def sse_stream(channel_name: str):
    """Create SSE stream response for client."""
    if channel_name not in sse_channels:
        return Response("Channel not found", status=404)
    
    channel = sse_channels[channel_name]
    q = channel.subscribe()
    
    def event_stream():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'channel': channel_name})}\n\n"
            
            # Stream events
            while True:
                try:
                    message = q.get(timeout=30)
                    yield f"data: {json.dumps(message)}\n\n"
                except:
                    # Heartbeat every 30 seconds
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
        except GeneratorExit:
            channel.unsubscribe(q)
    
    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


def broadcast_alert(alert: Dict[str, Any]):
    """Broadcast alert to interested channels."""
    sse_channels["alerts"].publish(alert, event_type="alert")
    sse_channels["hot"].publish(alert, event_type="alert")


def broadcast_health(health_status: Dict[str, Any]):
    """Broadcast health update to interested channels."""
    sse_channels["health"].publish(health_status, event_type="health_update")
    sse_channels["warm"].publish(health_status, event_type="health_update")


def broadcast_metrics(metrics: Dict[str, Any]):
    """Broadcast metrics update."""
    sse_channels["cold"].publish(metrics, event_type="metrics_update")
