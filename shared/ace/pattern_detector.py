import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DetectionEvent:
    type: str
    confidence: float
    position: int
    context_window: str
    suggested_action: str


class StreamingPatternDetector:
    PATTERNS: Dict[str, Dict[str, str]] = {
        "explaining": {
            "regex": r"(?i)(here'?s how|let me explain|the approach is|we can solve this)",
            "action": "summarize_reasoning",
        },
        "coding": {
            "regex": r"(?i)(```[\w-]*|def |class |import |function|return )",
            "action": "inject_code_hint",
        },
        "uncertain": {
            "regex": r"(?i)(i think|maybe|possibly|not sure|alternative|might be)",
            "action": "confidence_boost",
        },
        "questioning": {
            "regex": r"(?i)(should i|would it|is it better to|choose between|which approach)",
            "action": "branch_checklist",
        },
        "tool_need": {
            "regex": r"(?i)(need to check|let me look up|search for|find the|inspect the file|open the file)",
            "action": "prefetch_tools",
        },
        "planning": {
            "regex": r"(?i)(step \d|first|then|finally|approach:|plan:)",
            "action": "track_plan",
        },
        "comparing": {
            "regex": r"(?i)(option [abc]|approach [123]| vs\\.? | versus |trade[- ]off)",
            "action": "branch_checklist",
        },
        "hallucination_risk": {
            "regex": r"(?i)(i recall|i remember|typically|usually|generally)",
            "action": "retrieve_context",
        },
    }

    def __init__(self, window_chars: int = 360):
        self.window_chars = window_chars

    def detect(
        self,
        text: str,
        *,
        position: int,
        seen: set[str] | None = None,
    ) -> List[DetectionEvent]:
        current = (text or "")[-self.window_chars :]
        if not current.strip():
            return []
        seen = seen or set()
        events: List[DetectionEvent] = []
        for name, cfg in self.PATTERNS.items():
            match = re.search(cfg["regex"], current)
            if not match:
                continue
            signature = f"{name}:{match.group(0).lower()}"
            if signature in seen:
                continue
            seen.add(signature)
            events.append(
                DetectionEvent(
                    type=name,
                    confidence=self._score(name, match.group(0)),
                    position=position,
                    context_window=current[-220:],
                    suggested_action=cfg["action"],
                )
            )
        return sorted(events, key=lambda item: item.confidence, reverse=True)[:3]

    def _score(self, name: str, matched_text: str) -> float:
        base = {
            "tool_need": 0.93,
            "questioning": 0.89,
            "comparing": 0.88,
            "hallucination_risk": 0.84,
            "uncertain": 0.82,
            "coding": 0.79,
            "planning": 0.74,
            "explaining": 0.71,
        }.get(name, 0.7)
        bonus = min(len(matched_text.strip()) / 100.0, 0.08)
        return round(min(base + bonus, 0.99), 3)
