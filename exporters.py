import json
import time
import yaml
from pathlib import Path
from models import ModelRecommendation

class ConfigExporter:
    """Exports optimized configuration to LM Studio formats."""
    
    @staticmethod
    def build_preset_dict(
        rec: ModelRecommendation,
        *,
        profile: dict | None = None,
        hardware: dict | None = None,
        name: str | None = None,
        identifier: str | None = None,
    ) -> dict:
        """Build an LM Studio preset with bridge-aware metadata and legacy mirrors."""
        profile = profile or {}
        
        op_fields = [
            {"key": "llm.prediction.temperature", "value": rec.temperature},
            {"key": "llm.prediction.topPSampling", "value": {"checked": True, "value": rec.top_p}},
            {"key": "llm.prediction.topKSampling", "value": {"checked": True, "value": rec.top_k}},
            {"key": "llm.prediction.repeatPenalty", "value": {"checked": True, "value": rec.repeat_penalty}},
            {"key": "llm.prediction.maxTokens", "value": rec.max_tokens},
        ]
        
        if rec.system_prompt:
            op_fields.append({"key": "llm.prediction.systemPrompt", "value": rec.system_prompt})
        
        custom_fields = []
        if rec.enable_thinking:
            custom_fields.append({"key": "enableThinking", "value": True})

        preset = {
            "identifier": identifier or f"@local:{rec.model_id.replace('/', '-').lower()}-{rec.quantization.value.lower()}",
            "name": name or f"{rec.model_id} - Optimized ({rec.quantization.value})",
            "changed": True,
            "importedTimeStamp": int(time.time() * 1000),
            "operation": {
                "fields": op_fields,
            },
            "load": {
                "fields": [
                    {"key": "llm.load.contextLength", "value": rec.context_length},
                    {"key": "llm.load.gpuOffload", "value": rec.gpu_layers},
                    {"key": "llm.load.flashAttention", "value": rec.flash_attention},
                    {"key": "llm.load.tryMmap", "value": rec.use_mmap},
                    {"key": "llm.load.threads", "value": rec.threads},
                    {"key": "llm.load.batchSize", "value": rec.batch_size},
                    {"key": "llm.load.kvCacheQuantization", "value": rec.kv_cache_quant},
                    {"key": "llm.load.numa", "value": rec.numa_support},
                ]
            },
            "inference_params": {
                "n_threads": rec.threads,
                "n_predict": rec.max_tokens,
                "top_k": rec.top_k,
                "top_p": rec.top_p,
                "temp": rec.temperature,
                "repeat_penalty": rec.repeat_penalty,
                "pre_prompt": rec.system_prompt or "",
                "memory_f16": rec.kv_cache_quant == "f16",
            },
            "load_params": {
                "n_ctx": rec.context_length,
                "n_batch": rec.batch_size,
                "n_gpu_layers": rec.gpu_layers,
                "f16_kv": rec.kv_cache_quant == "f16",
                "use_mmap": rec.use_mmap,
            },
            "_wizard_meta": {
                "backend": rec.inference_backend.value,
                "engine": rec.inference_engine,
                "quality_score": rec.quality_score,
                "generated_by": "lmstudio-agent-console-v7",
            },
            "_bridge_profile": {
                "mode": profile.get("mode", "fast"),
                "embed_model": profile.get("embed_model"),
                "rerank_model": profile.get("rerank_model"),
                "retrieval": {
                    "chunk_size": profile.get("chunk_size"),
                    "chunk_overlap": profile.get("chunk_overlap"),
                    "top_k": profile.get("retrieval_top_k"),
                    "max_context_chars": profile.get("max_context_chars"),
                },
            },
        }
        if hardware:
            preset["_hardware_profile"] = hardware
        
        if custom_fields:
            preset["customFields"] = custom_fields

        return preset

    @staticmethod
    def export_preset(
        rec: ModelRecommendation,
        path: Path,
        *,
        profile: dict | None = None,
        hardware: dict | None = None,
        name: str | None = None,
        identifier: str | None = None,
    ):
        """Write LM Studio User Preset JSON to disk."""
        preset = ConfigExporter.build_preset_dict(
            rec,
            profile=profile,
            hardware=hardware,
            name=name,
            identifier=identifier,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preset, indent=2))
    
    @staticmethod
    def export_yaml(rec: ModelRecommendation, path: Path):
        """Standard model.yaml format for generic hub uploads."""
        config = {
            "config": {
                "load": {
                    "fields": [
                        {"key": "llm.load.contextLength", "value": rec.context_length},
                        {"key": "llm.load.gpuOffload", "value": rec.gpu_layers},
                        {"key": "llm.load.flashAttention", "value": rec.flash_attention},
                    ]
                },
                "operation": {
                    "fields": [
                        {"key": "llm.prediction.temperature", "value": rec.temperature},
                        {"key": "llm.prediction.topPSampling", "value": {"checked": True, "value": rec.top_p}},
                        {"key": "llm.prediction.maxTokens", "value": rec.max_tokens},
                    ]
                }
            },
        }
        if rec.enable_thinking:
            config["customFields"] = [
                {"key": "enableThinking", "displayName": "Enable Thinking", "type": "boolean", "defaultValue": True}
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(config, default_flow_style=False))
