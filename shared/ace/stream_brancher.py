from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BranchOption:
    id: str
    label: str
    confidence: float
    strategy: str


class StreamBranchingEngine:
    def suggest_branches(self, prompt: str, mode: str) -> List[Dict[str, Any]]:
        base = [
            BranchOption(id="A", label="Direct implementation path", confidence=0.72, strategy="implement"),
            BranchOption(id="B", label="Reasoning and trade-off path", confidence=0.69, strategy="analyze"),
            BranchOption(id="C", label="Compressed hybrid path", confidence=0.63, strategy="hybrid"),
        ]
        lower = (prompt or "").lower()
        if "debug" in lower or "error" in lower:
            base[0] = BranchOption(id="A", label="Root-cause isolate first", confidence=0.78, strategy="debug")
        if mode == "fast":
            base = base[:2]
        return [
            {"id": item.id, "label": item.label, "confidence": item.confidence, "strategy": item.strategy}
            for item in base
        ]
