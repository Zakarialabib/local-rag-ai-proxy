import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from exporters import ConfigExporter


class PresetService:
    def __init__(self):
        home = Path.home()
        self.preset_dirs = [
            home / ".cache" / "lm-studio" / "config-presets",
            home / ".cache" / "lm-studio" / "presets",
        ]

    def list_presets(self) -> List[Path]:
        presets: List[Path] = []
        for directory in self.preset_dirs:
            if not directory.exists():
                continue
            presets.extend(directory.glob("*.json"))
            presets.extend(directory.glob("*.preset.json"))
        unique = {path.resolve(): path for path in presets}
        return sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)

    def read_preset(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def apply_to_profile(
        self,
        preset: Dict[str, Any],
        current_profile: Dict[str, Any],
        role_map: Dict[str, str],
        selected_model: str,
    ) -> Dict[str, Any]:
        profile = dict(current_profile)
        profile.update(
            {
                "status": "preset-loaded",
                "model_id": selected_model or current_profile.get("model_id") or role_map.get("main", ""),
                "embed_model": role_map.get("embed", current_profile.get("embed_model", "")),
                "rerank_model": role_map.get("rerank", current_profile.get("rerank_model", "")),
            }
        )

        def get_field(section: str, key: str):
            for field in preset.get(section, {}).get("fields", []):
                if field.get("key") == key:
                    return field.get("value")
            return None

        def get_legacy(section: str, key: str, default=None):
            return preset.get(section, {}).get(key, default)

        profile["temperature"] = get_field("operation", "llm.prediction.temperature") or get_legacy(
            "inference_params", "temp", profile.get("temperature", 0.3)
        )
        top_p_val = get_field("operation", "llm.prediction.topPSampling")
        profile["top_p"] = (
            top_p_val.get("value", profile.get("top_p", 0.95))
            if isinstance(top_p_val, dict)
            else get_legacy("inference_params", "top_p", profile.get("top_p", 0.95))
        )
        top_k_val = get_field("operation", "llm.prediction.topKSampling")
        profile["top_k"] = (
            top_k_val.get("value", profile.get("top_k", 40))
            if isinstance(top_k_val, dict)
            else get_legacy("inference_params", "top_k", profile.get("top_k", 40))
        )
        repeat_penalty = get_field("operation", "llm.prediction.repeatPenalty")
        profile["repeat_penalty"] = (
            repeat_penalty.get("value", profile.get("repeat_penalty", 1.1))
            if isinstance(repeat_penalty, dict)
            else get_legacy("inference_params", "repeat_penalty", profile.get("repeat_penalty", 1.1))
        )
        profile["max_tokens"] = (
            get_field("operation", "llm.prediction.maxTokens")
            or get_legacy("inference_params", "n_predict")
            or profile.get("max_tokens", 1024)
        )
        profile["system_prompt"] = (
            get_field("operation", "llm.prediction.systemPrompt")
            or get_legacy("inference_params", "pre_prompt")
            or profile.get("system_prompt", "")
        )
        profile["context_length"] = (
            get_field("load", "llm.load.contextLength")
            or get_legacy("load_params", "n_ctx")
            or profile.get("context_length", 2048)
        )

        bridge_profile = preset.get("_bridge_profile", {})
        if isinstance(bridge_profile, dict):
            profile["mode"] = bridge_profile.get("mode", profile.get("mode", "fast"))
            profile["embed_model"] = bridge_profile.get("embed_model", profile["embed_model"])
            profile["rerank_model"] = bridge_profile.get("rerank_model", profile["rerank_model"])
            retrieval = bridge_profile.get("retrieval", {})
            if isinstance(retrieval, dict):
                profile["chunk_size"] = retrieval.get("chunk_size", profile.get("chunk_size", 900))
                profile["chunk_overlap"] = retrieval.get("chunk_overlap", profile.get("chunk_overlap", 150))
                profile["retrieval_top_k"] = retrieval.get("top_k", profile.get("retrieval_top_k", 4))
                profile["max_context_chars"] = retrieval.get(
                    "max_context_chars", profile.get("max_context_chars", 6000)
                )
        return profile

    def export_preset(self, rec: Any, path: Path, profile: Dict[str, Any], hardware: Dict[str, Any]) -> Path:
        ConfigExporter.export_preset(
            rec,
            path,
            profile=profile,
            hardware=hardware,
            name=f"{rec.model_id} - Agent Console Web",
            identifier=f"@local:{rec.model_id.replace('/', '-').lower()}-web-console",
        )
        return path
