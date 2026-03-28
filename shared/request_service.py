import json
import re
from typing import Any, Dict, List, Tuple

import httpx


class RequestService:
    def __init__(self, bridge_base: str, hardware: Dict[str, Any] | None = None):
        self.bridge_base = bridge_base.rstrip("/")
        self.hardware = hardware or {}

    def build_responses_payload(self, form: Dict[str, Any], role_map: Dict[str, str], profile: Dict[str, Any]) -> Dict[str, Any]:
        retrieval = self._retrieval_config(form, profile)
        raw_instructions = str(form.get("instructions") or profile.get("system_prompt") or "").strip()
        raw_input = str(form.get("input") or "").strip()
        instructions = self._condense_text(raw_instructions, kind="system")
        user_input = self._condense_text(raw_input, kind="user")
        payload: Dict[str, Any] = {
            "model": str(form.get("model") or role_map.get("main") or profile.get("model_id") or "").strip(),
            "instructions": instructions,
            "input": user_input,
            "reasoning": {"effort": str(form.get("reasoning_effort") or "medium").strip()},
            "max_output_tokens": self._safe_int(form.get("max_output_tokens"), profile.get("max_tokens", 1024)),
            "temperature": self._safe_float(form.get("temperature"), profile.get("temperature", 0.3)),
            "top_p": self._safe_float(form.get("top_p"), profile.get("top_p", 0.95)),
            "top_k": self._safe_int(form.get("top_k"), profile.get("top_k", 40)),
            "repeat_penalty": self._safe_float(form.get("repeat_penalty"), profile.get("repeat_penalty", 1.1)),
            "mode": str(form.get("mode") or profile.get("mode") or "fast").strip(),
            "retrieval": retrieval,
            "stream": False,
            "_prompt_source": {
                "instructions": raw_instructions,
                "input": raw_input,
            },
        }
        if form.get("tools"):
            payload["tools"] = form["tools"]
        if form.get("tool_choice"):
            payload["tool_choice"] = form["tool_choice"]
        docs = form.get("context_docs") or []
        if isinstance(docs, list):
            payload["context_docs"] = docs
        return payload

    def build_chat_payload(self, form: Dict[str, Any], role_map: Dict[str, str], profile: Dict[str, Any]) -> Dict[str, Any]:
        raw_system_prompt = str(form.get("system_prompt") or profile.get("system_prompt") or "").strip()
        raw_user_input = str(form.get("input") or "").strip()
        system_prompt = self._condense_text(raw_system_prompt, kind="system")
        user_input = self._condense_text(raw_user_input, kind="user")
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_input:
            messages.append({"role": "user", "content": user_input})
        payload: Dict[str, Any] = {
            "model": str(form.get("model") or role_map.get("main") or profile.get("model_id") or "").strip(),
            "messages": messages,
            "max_tokens": self._safe_int(form.get("max_tokens"), profile.get("max_tokens", 1024)),
            "temperature": self._safe_float(form.get("temperature"), profile.get("temperature", 0.3)),
            "top_p": self._safe_float(form.get("top_p"), profile.get("top_p", 0.95)),
            "top_k": self._safe_int(form.get("top_k"), profile.get("top_k", 40)),
            "repeat_penalty": self._safe_float(form.get("repeat_penalty"), profile.get("repeat_penalty", 1.1)),
            "mode": str(form.get("mode") or profile.get("mode") or "fast").strip(),
            "stream": bool(form.get("stream", False)),
            "_prompt_source": {
                "system_prompt": raw_system_prompt,
                "input": raw_user_input,
            },
        }
        return payload

    def recommend_workspace(
        self,
        *,
        model_id: str,
        use_case: str,
        profile: Dict[str, Any],
        role_map: Dict[str, str],
        loaded_models: List[str] | None = None,
    ) -> Dict[str, Any]:
        loaded = {str(item).strip() for item in (loaded_models or []) if str(item).strip()}
        model_text = str(model_id or profile.get("model_id") or role_map.get("main") or "").lower()
        low_resource = self._is_low_resource()
        think_capable = bool(profile.get("thinking_recommended")) or any(
            hint in model_text for hint in ("reasoning", "deepseek-r1", "claude", "opus", "r1")
        )
        vision_capable = any(hint in model_text for hint in ("vision", "vl", "llava", "pixtral", "molmo"))
        mode = "think" if think_capable and use_case in {"coding", "logic"} else str(profile.get("mode") or "fast")
        if low_resource and mode == "architect":
            mode = "fast"

        reasoning_effort = "high" if mode == "think" and not low_resource else ("medium" if think_capable else "low")
        temperature = self._recommended_temperature(use_case, think_capable)
        top_p = 0.9 if use_case in {"coding", "logic"} else 0.95
        top_k = 20 if use_case in {"coding", "logic"} else 40
        repeat_penalty = 1.05 if think_capable else 1.1
        max_tokens = self._recommended_output_tokens(profile, use_case, low_resource)
        retrieval = {
            "top_k": min(self._safe_int(profile.get("retrieval_top_k"), 4), 3 if low_resource else 5),
            "chunk_size": min(self._safe_int(profile.get("chunk_size"), 900), 700 if low_resource else 1100),
            "chunk_overlap": min(self._safe_int(profile.get("chunk_overlap"), 150), 100 if low_resource else 180),
            "max_context_chars": min(self._safe_int(profile.get("max_context_chars"), 6000), 2400 if low_resource else 7000),
            "include_sources": True,
        }

        prompt_packs = {
            "balanced": (
                "Answer with a direct result first, then only the minimal supporting detail needed.",
                "Keep the answer reliable, practical, and stable."
            ),
            "coding": (
                "Act like a local coding agent. Prefer deterministic solutions, explicit tradeoffs, and safe edits.",
                "When relevant, return implementation steps, code, and verification notes in a compact structure."
            ),
            "creative": (
                "Be vivid and polished, but keep structure coherent and avoid drifting from the request.",
                "Use retrieval context when it is available and relevant."
            ),
            "logic": (
                "Break the task into ordered reasoning steps and state the conclusion clearly.",
                "Prefer precise, falsifiable claims over stylistic flourish."
            ),
        }
        base_prompt = str(profile.get("system_prompt") or "").strip()
        extra_a, extra_b = prompt_packs.get(use_case, prompt_packs["balanced"])
        instructions = "\n\n".join(part for part in (base_prompt, extra_a, extra_b) if part)
        notes = [
            f"Mode auto-selected as `{mode}` from hardware + model profile.",
            f"Reasoning effort set to `{reasoning_effort}`.",
            "Temperature lowered for deterministic output." if temperature <= 0.3 else "Temperature kept flexible for breadth.",
            "Retrieval budget clamped for current hardware." if low_resource else "Retrieval budget kept broad for richer context.",
            f"Embedding role loaded: {'yes' if role_map.get('embed') in loaded else 'no'}",
            f"Rerank role loaded: {'yes' if role_map.get('rerank') in loaded else 'no'}",
        ]

        return {
            "responses": {
                "model": model_id or role_map.get("main") or profile.get("model_id", ""),
                "instructions": instructions,
                "mode": mode,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "retrieval_top_k": retrieval["top_k"],
                "chunk_size": retrieval["chunk_size"],
                "chunk_overlap": retrieval["chunk_overlap"],
                "max_context_chars": retrieval["max_context_chars"],
            },
            "chat": {
                "model": model_id or role_map.get("main") or profile.get("model_id", ""),
                "system_prompt": instructions,
                "mode": mode,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "stream": False,
            },
            "capabilities": {
                "thinking": think_capable,
                "vision": vision_capable,
                "tool_ready": think_capable,
                "embed_loaded": role_map.get("embed") in loaded,
                "rerank_loaded": role_map.get("rerank") in loaded,
            },
            "notes": notes,
        }

    def preview_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        guarded, warnings, timeout = self.apply_guardrails(payload)
        prompt_stats = self._prompt_stats(payload, guarded)
        return {
            "payload": guarded,
            "warnings": warnings,
            "timeout_policy": timeout,
            "prompt_stats": prompt_stats,
        }

    def run_responses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        guarded, warnings, timeout = self.apply_guardrails(payload)
        self._ensure_bridge_ready()
        request_body = {k: v for k, v in guarded.items() if k != "_prompt_source"}
        with httpx.Client(timeout=timeout["seconds"]) as client:
            response = client.post(f"{self.bridge_base}/v1/responses", json=request_body)
            response.raise_for_status()
            data = response.json()
        return {
            "request": request_body,
            "response": data,
            "warnings": warnings,
            "timeout_policy": timeout,
            "prompt_stats": self._prompt_stats(payload, guarded),
        }

    def run_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        guarded, warnings, timeout = self.apply_guardrails(payload)
        self._ensure_bridge_ready()
        request_body = {k: v for k, v in guarded.items() if k != "_prompt_source"}
        if request_body.get("stream"):
            return self._run_chat_stream(request_body, warnings, timeout, original_payload=guarded)
        with httpx.Client(timeout=timeout["seconds"]) as client:
            response = client.post(f"{self.bridge_base}/v1/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
        return {
            "request": request_body,
            "response": data,
            "warnings": warnings,
            "timeout_policy": timeout,
            "prompt_stats": self._prompt_stats(payload, guarded),
        }

    def apply_guardrails(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
        body = json.loads(json.dumps(payload))
        warnings: List[str] = []
        prompt_source = body.pop("_prompt_source", None)
        low_resource = bool(
            ((self.hardware.get("gpu_vram_gb") or 0) <= 8.5)
            or ((self.hardware.get("system_ram_gb") or 0) <= 16.5)
        )
        mode = str(body.get("mode") or "fast").strip()
        timeout = self._timeout_policy(mode, bool(body.get("stream")))

        retrieval = body.get("retrieval")
        if isinstance(retrieval, dict):
            if low_resource:
                original = dict(retrieval)
                retrieval["top_k"] = min(self._safe_int(retrieval.get("top_k"), 4), 3)
                retrieval["max_context_chars"] = min(self._safe_int(retrieval.get("max_context_chars"), 3200), 2400)
                retrieval["chunk_size"] = min(self._safe_int(retrieval.get("chunk_size"), 900), 700)
                retrieval["chunk_overlap"] = min(self._safe_int(retrieval.get("chunk_overlap"), 150), 100)
                if retrieval != original:
                    warnings.append("Low-resource guardrails reduced retrieval budget for this request.")

        max_output_tokens = self._safe_int(body.get("max_output_tokens", body.get("max_tokens")), 1024)
        if low_resource and max_output_tokens > 1024:
            if "max_output_tokens" in body:
                body["max_output_tokens"] = 1024
            if "max_tokens" in body:
                body["max_tokens"] = 1024
            warnings.append("Output tokens were capped to 1024 for the current hardware profile.")
        if "temperature" in body:
            body["temperature"] = max(0.0, min(self._safe_float(body.get("temperature"), 0.3), 1.2))
        if "top_p" in body:
            body["top_p"] = max(0.1, min(self._safe_float(body.get("top_p"), 0.95), 1.0))
        if "top_k" in body:
            body["top_k"] = max(1, min(self._safe_int(body.get("top_k"), 40), 200))
        if "repeat_penalty" in body:
            body["repeat_penalty"] = max(1.0, min(self._safe_float(body.get("repeat_penalty"), 1.1), 1.3))

        estimated_chars = self._estimate_prompt_chars(body)
        context_length = self._safe_int(body.get("context_length"), self.hardware.get("context_length", 8192) or 8192)
        context_budget_chars = max(context_length, 2048) * 4
        if estimated_chars > int(context_budget_chars * 0.9):
            warnings.append("Prompt and retrieval payload are close to the configured context limit.")
        if prompt_source is not None:
            body["_prompt_source"] = prompt_source
        return body, warnings, timeout

    def _run_chat_stream(
        self,
        payload: Dict[str, Any],
        warnings: List[str],
        timeout: Dict[str, Any],
        *,
        original_payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        transcript: List[str] = []
        reasoning: List[str] = []
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", f"{self.bridge_base}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
                    if line.strip() == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if not chunk or chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    if not isinstance(delta, dict):
                        continue
                    if delta.get("content"):
                        transcript.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
        return {
            "request": payload,
            "response": {
                "streamed": True,
                "content": "".join(transcript),
                "reasoning_content": "".join(reasoning),
            },
            "warnings": warnings,
            "timeout_policy": timeout,
            "prompt_stats": self._prompt_stats(original_payload or payload, original_payload or payload),
        }

    def _retrieval_config(self, form: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "top_k": self._safe_int(form.get("retrieval_top_k"), profile.get("retrieval_top_k", 4)),
            "chunk_size": self._safe_int(form.get("chunk_size"), profile.get("chunk_size", 900)),
            "chunk_overlap": self._safe_int(form.get("chunk_overlap"), profile.get("chunk_overlap", 150)),
            "max_context_chars": self._safe_int(form.get("max_context_chars"), profile.get("max_context_chars", 6000)),
            "include_sources": bool(form.get("include_sources", True)),
        }

    def _estimate_prompt_chars(self, payload: Dict[str, Any]) -> int:
        base = 0
        if payload.get("instructions"):
            base += len(str(payload["instructions"]))
        if payload.get("input"):
            base += len(str(payload["input"]))
        for message in payload.get("messages", []):
            if isinstance(message, dict):
                base += len(str(message.get("content", "")))
        retrieval = payload.get("retrieval")
        if isinstance(retrieval, dict):
            base += self._safe_int(retrieval.get("max_context_chars"), 0)
        max_output_tokens = self._safe_int(payload.get("max_output_tokens", payload.get("max_tokens")), 0)
        base += max_output_tokens * 4
        return base

    def _prompt_stats(self, original: Dict[str, Any], condensed: Dict[str, Any]) -> Dict[str, Any]:
        source = condensed.get("_prompt_source") or original.get("_prompt_source") or {}
        original_chars = self._estimate_prompt_chars(self._payload_from_prompt_source(source, condensed))
        condensed_chars = self._estimate_prompt_chars({k: v for k, v in condensed.items() if k != "_prompt_source"})
        saved = max(original_chars - condensed_chars, 0)
        return {
            "original_chars": original_chars,
            "condensed_chars": condensed_chars,
            "saved_chars": saved,
            "saved_percent": round((saved / original_chars) * 100, 2) if original_chars else 0.0,
        }

    def _payload_from_prompt_source(self, source: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        rebuilt = {k: v for k, v in payload.items() if k != "_prompt_source"}
        if "instructions" in rebuilt and source.get("instructions") is not None:
            rebuilt["instructions"] = source.get("instructions", rebuilt["instructions"])
        if "input" in rebuilt and source.get("input") is not None:
            rebuilt["input"] = source.get("input", rebuilt["input"])
        if "messages" in rebuilt and isinstance(rebuilt["messages"], list):
            messages = json.loads(json.dumps(rebuilt["messages"]))
            for item in messages:
                if item.get("role") == "system" and source.get("system_prompt") is not None:
                    item["content"] = source.get("system_prompt", item.get("content", ""))
                elif item.get("role") == "user" and source.get("input") is not None:
                    item["content"] = source.get("input", item.get("content", ""))
                    break
            rebuilt["messages"] = messages
        return rebuilt

    def _ensure_bridge_ready(self) -> None:
        try:
            with httpx.Client(timeout=4) as client:
                response = client.get(f"{self.bridge_base}/api/v1/models")
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Bridge unavailable at {self.bridge_base}: {exc}") from exc

    def _timeout_policy(self, mode: str, stream: bool) -> Dict[str, Any]:
        if stream:
            return {"label": "streaming/no hard timeout", "seconds": None}
        if mode == "think":
            return {"label": "think/300s", "seconds": 300}
        if mode == "architect":
            return {"label": "architect/480s", "seconds": 480}
        return {"label": "fast/90s", "seconds": 90}

    def _is_low_resource(self) -> bool:
        return bool(
            ((self.hardware.get("gpu_vram_gb") or 0) <= 8.5)
            or ((self.hardware.get("system_ram_gb") or 0) <= 16.5)
        )

    def _recommended_temperature(self, use_case: str, think_capable: bool) -> float:
        if use_case == "coding":
            return 0.15
        if use_case == "logic":
            return 0.2 if think_capable else 0.25
        if use_case == "creative":
            return 0.8
        return 0.3 if think_capable else 0.45

    def _recommended_output_tokens(self, profile: Dict[str, Any], use_case: str, low_resource: bool) -> int:
        base = self._safe_int(profile.get("max_tokens"), 1024)
        if low_resource:
            return min(base, 1024)
        if use_case == "coding":
            return min(max(base, 2048), 4096)
        if use_case == "logic":
            return min(max(base, 1536), 3072)
        return min(base, 2048)

    def _condense_text(self, text: str, *, kind: str) -> str:
        value = re.sub(r"\s+", " ", text or "").strip()
        if not value:
            return ""

        if kind == "system":
            lowered = value.lower()
            if "always start your response with a <thinking>" in lowered or "specialized reasoning model" in lowered:
                return (
                    "Reason carefully and keep hidden reasoning concise. "
                    "Return a short final answer first, then compact steps, code, or checks only when useful."
                )
            replacements = [
                (r"Act like a local coding agent\.\s*", "Act as a precise local coding agent. "),
                (r"When relevant, return implementation steps, code, and verification notes in a compact structure\.\s*", "Prefer concise steps, code, and verification notes. "),
                (r"Break every complex problem into clear, numbered steps\.\s*", "Use clear numbered steps. "),
                (r"ALWAYS start your response with a <thinking> or <reasoning> block\.\s*", ""),
                (r"Inside this block, perform a deep, multi-step analysis of the prompt\.\s*", ""),
                (r"Format:\s*<thinking>.*?\[Your final, polished answer here\]", "",),
            ]
            compact = value
            for pattern, repl in replacements:
                compact = re.sub(pattern, repl, compact, flags=re.IGNORECASE | re.DOTALL)
            compact = re.sub(r"\s+", " ", compact).strip()
            return compact[:420]

        value = re.sub(r"\n{3,}", "\n\n", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        return value[:2400]

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default
