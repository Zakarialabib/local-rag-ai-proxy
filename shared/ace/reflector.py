import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime

logger = structlog.get_logger("ace_reflector")

class ACEReflector:
    def __init__(self, playbook_dir: Path):
        self.playbook_dir = playbook_dir
        self.playbook_dir.mkdir(parents=True, exist_ok=True)

    def load_playbook(self, domain: str) -> List[str]:
        path = self.playbook_dir / f"{domain}_playbook.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except:
                return []
        return []

    def save_playbook(self, domain: str, bullets: List[str]):
        path = self.playbook_dir / f"{domain}_playbook.json"
        path.write_text(json.dumps(bullets, indent=2))

    async def reflect_on_session(self, session_id: str, trace: List[Dict[str, Any]], final_text: str):
        """Analyze a finished session and distill new playbook bullets."""
        # Simple heuristic: If tool results were successful, they are good bullets
        new_bullets = []
        domain = "general"
        
        for event in trace:
            event_type = event.get("type")
            data = event.get("data", {})
            
            if event_type == "cognitive_route_detected":
                domain = data.get("domain", "general")
            
            if event_type == "tool_result":
                # If validation passed, this was a good strategy
                val = data.get("validation", {})
                if val.get("status") == "passed":
                    bullet = f"For tasks involving {data.get('tool')}, effective context includes: {str(data.get('result'))[:200]}..."
                    new_bullets.append(bullet)

        if new_bullets:
            current = self.load_playbook(domain)
            # Add new bullets and keep it tidy (simple deduplication)
            updated = list(set(current + new_bullets))
            self.save_playbook(domain, updated[:50]) # Keep top 50
            logger.info("playbook_updated", domain=domain, new_count=len(new_bullets))
