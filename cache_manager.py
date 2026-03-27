import structlog
from typing import Dict, Any, Optional
from collections import OrderedDict
import hashlib
import json

logger = structlog.get_logger()

class LRUCache:
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

class CacheManager:
    def __init__(self):
        self.response_cache = LRUCache(max_size=100)
        self.model_list_cache: Optional[Dict[str, Any]] = None
        self.model_list_expiry: float = 0
        
    def build_cache_key(self, body: Dict) -> str:
        """Build a unique cache key from request body."""
        essential = {
            "model": body.get("model", ""),
            "messages": body.get("messages", []),
            "stream": bool(body.get("stream", False)),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_tokens": body.get("max_tokens"),
            "stop": body.get("stop"),
            "presence_penalty": body.get("presence_penalty"),
            "frequency_penalty": body.get("frequency_penalty"),
            "seed": body.get("seed"),
            "response_format": body.get("response_format"),
            "tools": body.get("tools"),
            "tool_choice": body.get("tool_choice"),
        }
        key_str = json.dumps(essential, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
