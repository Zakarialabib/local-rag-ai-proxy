import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import httpx

from model_discovery import get_local_models


def clean_model_id(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"model_key='([^']+)'", text)
    if match:
        return match.group(1)
    return text


class RuntimeService:
    def __init__(self, bridge_base: str, lmstudio_base: str):
        self.bridge_base = bridge_base.rstrip("/")
        self.lmstudio_base = lmstudio_base.rstrip("/")

    def list_local_models(self) -> List[Dict[str, Any]]:
        return get_local_models()

    def build_role_choices(self, models: List[Dict[str, Any]], role_key: str, show_all: bool = False) -> List[str]:
        if show_all:
            return [item["id"] for item in models]
        wanted = "llm" if role_key in {"main", "reasoning"} else ("embedding" if role_key == "embed" else "rerank")
        filtered = [item["id"] for item in models if item.get("type") == wanted]
        if wanted == "llm" and not filtered:
            filtered = [item["id"] for item in models]
        return filtered

    def refresh_runtime_status(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        with httpx.Client(timeout=10) as client:
            try:
                report["bridge_models"] = client.get(f"{self.bridge_base}/api/v1/models").json()
            except Exception as exc:
                report["bridge_error"] = str(exc)
            try:
                report["bridge_hardware"] = client.get(f"{self.bridge_base}/api/v1/hardware").json()
            except Exception:
                pass
            try:
                report["lmstudio_models"] = client.get(f"{self.lmstudio_base}/api/v1/models").json()
            except Exception as exc:
                report["lmstudio_error"] = str(exc)
        return report

    def list_loaded_models(self, report: Dict[str, Any], local_models: List[Dict[str, Any]] | None = None) -> List[str]:
        loaded: List[str] = []
        for payload_key in ("bridge_models", "lmstudio_models"):
            payload = report.get(payload_key)
            for item in self._runtime_model_list(payload):
                if self._runtime_model_is_loaded(item):
                    loaded.append(item.get("id") or item.get("key") or item.get("display_name"))
            if loaded:
                return loaded
        if local_models:
            loaded = [item["id"] for item in local_models if item.get("loaded")]
        return loaded

    def load_role_model(self, role_key: str, model_id: str, context_length: int | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": clean_model_id(model_id)}
        if context_length:
            payload["context_length"] = int(context_length)
        with httpx.Client(timeout=90) as client:
            response = client.post(f"{self.bridge_base}/api/v1/models/load", json=payload)
            response.raise_for_status()
            return {"role": role_key, "model": payload["model"], "result": response.json()}

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        cleaned = clean_model_id(model_id)
        result = subprocess.run(
            ["lms", "unload", cleaned],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "LM Studio unload failed").strip()
            raise RuntimeError(message)
        return {"model": cleaned, "output": (result.stdout or "").strip()}

    def save_role_mapping(self, role_map: Dict[str, str], env_path: Path) -> None:
        existing: List[str] = []
        if env_path.exists():
            existing = env_path.read_text(encoding="utf-8").splitlines()
        keys = {"MAIN_MODEL", "REASONING_MODEL", "EMBED_MODEL", "RERANK_MODEL"}
        kept = [line for line in existing if not any(line.startswith(f"{key}=") for key in keys)]
        kept.extend(
            [
                f"MAIN_MODEL={clean_model_id(role_map.get('main', ''))}",
                f"REASONING_MODEL={clean_model_id(role_map.get('reasoning', ''))}",
                f"EMBED_MODEL={clean_model_id(role_map.get('embed', ''))}",
                f"RERANK_MODEL={clean_model_id(role_map.get('rerank', ''))}",
            ]
        )
        env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def build_inventory_rows(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in models:
            rows.append(
                {
                    "id": item.get("id", ""),
                    "type": item.get("type", "llm"),
                    "arch": item.get("arch", "?"),
                    "params": item.get("params", "?"),
                    "size": item.get("size", "?"),
                    "loaded": bool(item.get("loaded")),
                }
            )
        return rows

    def _runtime_model_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return data
            models = payload.get("models")
            if isinstance(models, list):
                return models
        if isinstance(payload, list):
            return payload
        return []

    def _runtime_model_is_loaded(self, item: Dict[str, Any]) -> bool:
        if item.get("state") == "loaded":
            return True
        loaded_instances = item.get("loaded_instances")
        return isinstance(loaded_instances, list) and len(loaded_instances) > 0
