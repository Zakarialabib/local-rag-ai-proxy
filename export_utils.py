import json
import os
import yaml
from datetime import datetime

def export_to_yaml(config, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True)

def export_to_preset_json(config, model_profile, output_path):
    """
    Exports configuration to LM Studio User Preset JSON format.
    """
    preset = {
        "name": f"{model_profile.get('model_name', 'Default')} - Optimized",
        "load": {
            "fields": [
                {"key": "llm.load.contextLength", "value": config.get("context_length")},
                {"key": "llm.load.gpuOffload", "value": config.get("gpu_offload")},
                {"key": "llm.load.cpuThreads", "value": config.get("cpu_threads")},
                {"key": "llm.load.evaluationBatchSize", "value": config.get("batch_size")},
                {"key": "llm.load.flashAttention", "value": config.get("flash_attention")},
                {"key": "llm.load.tryMmap", "value": config.get("try_mmap")}
            ]
        },
        "operation": {
            "fields": [
                {"key": "llm.prediction.temperature", "value": config.get("temperature")},
                {"key": "llm.prediction.topPSampling", "value": {"checked": True, "value": config.get("top_p")}},
                {"key": "llm.prediction.topKSampling", "value": {"checked": True, "value": config.get("top_k")}},
                {"key": "llm.prediction.maxTokens", "value": config.get("max_tokens")},
                {"key": "llm.prediction.repeatPenalty", "value": {"checked": True, "value": config.get("repeat_penalty")}}
            ]
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(preset, f, indent=2)

def get_preset_save_path(model_name):
    """
    Finds the LM Studio user presets directory on Windows.
    """
    user_home = os.path.expanduser("~")
    presets_dir = os.path.join(user_home, ".cache", "lm-studio", "config-presets")
    filename = f"{model_name.replace(' ', '_')}_optimized.json"
    return os.path.join(presets_dir, filename)
