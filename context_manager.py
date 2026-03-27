import structlog
from typing import List, Dict

logger = structlog.get_logger()

class ContextManager:
    def prune_conversation_history(self, messages: List[Dict], max_turns: int = 10) -> List[Dict]:
        """Keep only the last N turns to reduce context bloat."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        
        # Keep last max_turns pairs
        pruned = non_system[-(max_turns * 2):]
        return system_msgs + pruned

    def extract_key_content(self, text: str, max_length: int = 4000) -> str:
        """Extract meaningful content, removing boilerplate."""
        lines = text.split('\n')
        meaningful = [l for l in lines if not any(x in l.lower() for x in 
            ['thank you', 'please help', 'could you', 'i would appreciate', 
             'write a function', 'create a', 'make a', 'generate'])]
        
        content = '\n'.join(meaningful)
        if len(content) > max_length:
            content = content[:max_length] + "... [content truncated]"
        return content

    def compress_context(self, messages: List[Dict], max_turns: int = 5) -> List[Dict]:
        """Compress conversation history by summarizing older turns."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        
        if len(non_system) <= max_turns * 2:
            return messages
        
        recent = non_system[-(max_turns * 2):]
        older = non_system[:-(max_turns * 2)]
        
        older_summary = self._summarize_turns(older)
        
        return system_msgs + [{
            "role": "system",
            "content": f"[Earlier conversation summarized: {older_summary}]"
        }] + recent

    def _summarize_turns(self, turns: List[Dict]) -> str:
        if not turns:
            return "No earlier context"
        
        topics = []
        for turn in turns:
            content = turn.get("content", "")[:50]
            role = turn.get("role", "")
            topics.append(f"{role}: {content}")
        
        return f"{len(turns)} turns about " + "; ".join(topics[:3])

    def decompress_context(self, messages: List[Dict]) -> List[Dict]:
        """Decompress context by expanding summary markers (RAG hook)."""
        decompressed = []
        for msg in messages:
            content = msg.get("content", "")
            if "[Earlier conversation summarized:" in content:
                # Placeholder for vector DB retrieval logic
                decompressed.append(msg)
            else:
                decompressed.append(msg)
        return decompressed

    def should_compress(self, messages: List[Dict], token_budget: int = 4000) -> bool:
        """Check if context should be compressed based on estimated token count."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        return estimated_tokens > token_budget
