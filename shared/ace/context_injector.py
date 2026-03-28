from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import httpx

from shared.ace.pattern_detector import DetectionEvent


@dataclass
class ACEIntervention:
    type: str
    trigger_pattern: str
    content: Dict[str, Any]
    priority: str = "medium"
    token_count: int = 0
    impact_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeContextInjector:
    def __init__(self, bridge_base: str):
        self.bridge_base = bridge_base.rstrip("/")

    async def engineer_context(
        self,
        detection: DetectionEvent,
        *,
        prompt: str,
        docs: List[str],
        session_context: Dict[str, Any],
    ) -> ACEIntervention | None:
        if detection.type in {"comparing", "questioning"}:
            options = self._extract_options(prompt)
            if len(options) < 2:
                options = ["Implementation path", "Research-first path", "Hybrid path"]
            payload = {
                "question": "Which path should ACE prioritize?",
                "options": [{"id": chr(65 + idx), "label": label} for idx, label in enumerate(options[:3])],
                "auto_select_if_confident": False,
            }
            return ACEIntervention(
                type="checklist_question",
                trigger_pattern=detection.type,
                content=payload,
                priority="high",
                token_count=len(json.dumps(payload)),
                impact_score=0.78,
            )

        if detection.type in {"tool_need", "hallucination_risk"} and docs:
            retrieval = await self._retrieve(prompt, docs, session_context)
            if retrieval:
                return ACEIntervention(
                    type="context_injection",
                    trigger_pattern=detection.type,
                    content=retrieval,
                    priority="medium",
                    token_count=len(json.dumps(retrieval)),
                    impact_score=0.72,
                )

        if detection.type == "uncertain":
            payload = {
                "hint": "Condense the answer, ground claims in retrieved context, and keep next steps explicit.",
                "focus": session_context.get("workflow") or "general",
            }
            return ACEIntervention(
                type="confidence_boost",
                trigger_pattern=detection.type,
                content=payload,
                priority="low",
                token_count=len(json.dumps(payload)),
                impact_score=0.61,
            )

        if detection.type == "coding":
            payload = {
                "hint": "Prefer short runnable code blocks and name files/functions explicitly.",
                "cwd": session_context.get("cwd", ""),
            }
            return ACEIntervention(
                type="code_hint",
                trigger_pattern=detection.type,
                content=payload,
                priority="low",
                token_count=len(json.dumps(payload)),
                impact_score=0.55,
            )
        return None

    async def _retrieve(self, prompt: str, docs: List[str], session_context: Dict[str, Any]) -> Dict[str, Any] | None:
        normalized_docs = self._normalize_docs(docs, session_context)
        payload = {
            "query": prompt[-600:],
            "docs": normalized_docs[:12],
            "retrieval": session_context.get("retrieval") or {},
        }
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(f"{self.bridge_base}/api/v1/retrieve", json=payload)
                response.raise_for_status()
                result = response.json()
        except Exception:
            return None
        chunks = ((result or {}).get("retrieval") or {}).get("chunks") or []
        if not chunks:
            return None
        snippets = []
        for item in chunks[:2]:
            text = str(item.get("text") or "")[:320]
            source = item.get("source") or "context"
            snippets.append({"source": source, "text": text})
        return {"snippets": snippets, "query": payload["query"]}

    def _normalize_docs(self, docs: List[str], session_context: Dict[str, Any]) -> List[str]:
        cwd = Path(session_context.get("cwd") or ".")
        normalized: List[str] = []
        for item in docs:
            raw = str(item).strip()
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (cwd / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8", errors="ignore")
                    normalized.append(f"[SOURCE {candidate.name}]\n{text[:10000]}")
                    continue
                except Exception:
                    pass
            normalized.append(raw)
        return normalized

    def _extract_options(self, prompt: str) -> List[str]:
        lines = [line.strip("-* \t") for line in (prompt or "").splitlines() if line.strip()]
        options = [line for line in lines if len(line) < 120][:3]
        return options
