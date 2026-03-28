import re
from typing import Any, Dict, List, Optional

from engine import RecommendationEngine, USE_CASE_PROFILES
from hardware_detector import HardwareDetector
from model_discovery import extract_model_specs, get_model_path


DEFAULT_RETRIEVAL = {
    "chunk_size": 900,
    "chunk_overlap": 150,
    "top_k": 4,
    "max_context_chars": 6000,
}


class ProfileService:
    def __init__(self):
        self.detector = HardwareDetector()
        self.hardware = self.detector.detect()
        self.engine = RecommendationEngine(self.hardware)
        self.spec_cache: Dict[str, Dict[str, Any]] = {}

    def get_hardware_dict(self) -> Dict[str, Any]:
        return self.hardware.model_dump()

    def compute_recommendations(
        self,
        model_id: str,
        params_b: float,
        use_case: str = "balanced",
        backend: str = "cuda",
        flash_attention: bool = True,
    ) -> List[Any]:
        specs = self._get_model_specs(model_id)
        recs = self.engine.recommend(
            model_id=model_id,
            params_b=params_b,
            num_layers=specs.get("num_layers", 32),
            hidden_size=specs.get("hidden_size", 4096),
            num_heads=specs.get("num_heads", 32),
            kv_heads=specs.get("kv_heads"),
            use_case=use_case,
            backend=backend,
            flash_attention=flash_attention,
        )
        if recs:
            return recs
        if backend != "cpu":
            return self.engine.recommend(
                model_id=model_id,
                params_b=params_b,
                num_layers=specs.get("num_layers", 32),
                hidden_size=specs.get("hidden_size", 4096),
                num_heads=specs.get("num_heads", 32),
                kv_heads=specs.get("kv_heads"),
                use_case=use_case,
                backend="cpu",
                flash_attention=False,
                min_quality=0.0,
            )
        return self.engine.recommend(
            model_id=model_id,
            params_b=params_b,
            num_layers=specs.get("num_layers", 32),
            hidden_size=specs.get("hidden_size", 4096),
            num_heads=specs.get("num_heads", 32),
            kv_heads=specs.get("kv_heads"),
            use_case=use_case,
            backend=backend,
            flash_attention=flash_attention,
            min_quality=0.0,
        )

    def estimate_params_b(self, model_entry: Dict[str, Any], default: float = 4.0) -> float:
        params_text = str(model_entry.get("params") or "").strip().lower()
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*b", params_text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return default
        return default

    def build_profile(self, rec: Any, role_map: Dict[str, str]) -> Dict[str, Any]:
        low_resource = bool(
            ((self.hardware.gpu_vram_gb or 0) <= 8.5) or ((self.hardware.system_ram_gb or 0) <= 16.5)
        )
        mode = "think" if rec.enable_thinking else ("architect" if rec.context_length >= 16384 else "fast")
        if low_resource and mode == "architect":
            mode = "fast"
        return {
            "status": "predicted",
            "model_id": rec.model_id,
            "system_prompt": rec.system_prompt or "You are a helpful local AI assistant.",
            "temperature": rec.temperature,
            "top_p": rec.top_p,
            "top_k": rec.top_k,
            "repeat_penalty": rec.repeat_penalty,
            "max_tokens": min(rec.max_tokens, 1024) if low_resource else rec.max_tokens,
            "context_length": min(rec.context_length, 4096) if low_resource else rec.context_length,
            "mode": mode,
            "chunk_size": 700 if low_resource else DEFAULT_RETRIEVAL["chunk_size"],
            "chunk_overlap": 100 if low_resource else DEFAULT_RETRIEVAL["chunk_overlap"],
            "retrieval_top_k": 3 if low_resource else DEFAULT_RETRIEVAL["top_k"],
            "max_context_chars": 2400 if low_resource else DEFAULT_RETRIEVAL["max_context_chars"],
            "embed_model": role_map.get("embed", ""),
            "rerank_model": role_map.get("rerank", ""),
            "thinking_recommended": rec.enable_thinking,
            "quality_score": rec.quality_score,
            "backend": rec.inference_backend.value,
            "quantization": rec.quantization.value,
        }

    def _get_model_specs(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.spec_cache:
            path = get_model_path(model_id)
            self.spec_cache[model_id] = extract_model_specs(path) or {}
        return self.spec_cache[model_id]
