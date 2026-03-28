import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional


def _normalize_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"embedding", "embed"}:
        return "embedding"
    if text in {"rerank", "reranker"}:
        return "rerank"
    return "llm"


def _sdk_item_data(item: Any) -> Dict[str, Any]:
    data = getattr(item, "_data", None)
    if data is not None:
        return {
            "model_key": getattr(data, "model_key", None),
            "display_name": getattr(data, "display_name", None),
            "architecture": getattr(data, "architecture", None),
            "size_bytes": getattr(data, "size_bytes", None),
            "params_string": getattr(data, "params_string", None),
            "path": getattr(data, "path", None),
        }
    info = getattr(item, "info", None)
    if callable(info):
        try:
            resolved = info()
            return {
                "model_key": getattr(resolved, "model_key", None),
                "display_name": getattr(resolved, "display_name", None),
                "architecture": getattr(resolved, "architecture", None),
                "size_bytes": getattr(resolved, "size_bytes", None),
                "params_string": getattr(resolved, "params_string", None),
                "path": getattr(resolved, "path", None),
            }
        except Exception:
            return {}
    return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_size(size_bytes: Any) -> str:
    size = _safe_int(size_bytes, 0)
    if size <= 0:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return "?"


def _model_id_from_entry(entry: Dict[str, Any]) -> str:
    return (
        str(entry.get("key") or "")
        or str(entry.get("model_key") or "")
        or str(entry.get("id") or "")
        or str(entry.get("display_name") or "")
    )


def _sdk_model_type(item: Any, raw: Dict[str, Any]) -> str:
    class_name = item.__class__.__name__.lower()
    if "embedding" in class_name:
        return "embedding"
    if "rerank" in class_name:
        return "rerank"
    model_key = str(raw.get("model_key") or "").lower()
    if "rerank" in model_key:
        return "rerank"
    if "embed" in model_key:
        return "embedding"
    return "llm"


def _sdk_loaded_ids(lms_module: Any) -> set:
    loaded_ids = set()
    try:
        loaded_items = lms_module.list_loaded_models()
    except Exception:
        loaded_items = []
    for item in loaded_items or []:
        raw = _sdk_item_data(item)
        model_id = _model_id_from_entry(raw) or str(item)
        if model_id:
            loaded_ids.add(model_id)
    return loaded_ids


def _from_lmstudio_sdk() -> Optional[List[Dict[str, Any]]]:
    try:
        import lmstudio as lms
    except Exception:
        return None

    try:
        downloaded = lms.list_downloaded_models()
    except Exception:
        return None

    loaded_ids = _sdk_loaded_ids(lms)
    models: List[Dict[str, Any]] = []
    for item in downloaded or []:
        raw = _sdk_item_data(item)
        model_id = _model_id_from_entry(raw) or str(item)
        if not model_id:
            continue
        model_type = _sdk_model_type(item, raw)
        architecture = str(raw.get("architecture") or raw.get("arch") or "?")
        params = str(raw.get("params_string") or raw.get("params") or "?")
        size = _format_size(raw.get("size_bytes"))
        models.append(
            {
                "id": model_id,
                "params": params,
                "arch": architecture,
                "size": size,
                "type": model_type,
                "loaded": model_id in loaded_ids,
            }
        )
    return models


def _from_lms_cli() -> List[Dict[str, Any]]:
    output = subprocess.check_output(["lms", "ls"], stderr=subprocess.DEVNULL).decode()
    models: List[Dict[str, Any]] = []
    current_type: Optional[str] = None

    for raw_line in output.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if "LLM" in upper and "PARAMS" in upper:
            current_type = "llm"
            continue
        if "EMBEDDING" in upper and "PARAMS" in upper:
            current_type = "embedding"
            continue
        if "RERANK" in upper and "PARAMS" in upper:
            current_type = "rerank"
            continue
        if not current_type:
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 4:
            continue

        model_id = parts[0]
        model_type = current_type
        lower_id = model_id.lower()
        if "rerank" in lower_id:
            model_type = "rerank"
        elif "embed" in lower_id and model_type == "llm":
            model_type = "embedding"

        models.append(
            {
                "id": model_id,
                "params": parts[1],
                "arch": parts[2],
                "size": parts[3],
                "type": model_type,
                "loaded": ("✓" in line) or ("âœ“" in line) or ("Ã¢Å“â€œ" in line),
            }
        )
    return models


def get_local_models() -> List[Dict[str, Any]]:
    """
    Return local models with type/arch/size metadata.
    Prefers lmstudio-python SDK and falls back to `lms ls`.
    """
    models = _from_lmstudio_sdk()
    if models is not None:
        return models
    try:
        return _from_lms_cli()
    except Exception:
        return []


def get_model_path(model_id):
    """
    Attempts to find the physical path to a model folder or .gguf file.
    """
    user_home = os.path.expanduser("~")
    base_cache_path = os.path.join(user_home, ".cache", "lm-studio", "models")

    for root, _dirs, files in os.walk(base_cache_path):
        parts = model_id.split("/")
        if len(parts) >= 2:
            publisher, repo = parts[0], parts[1]
            if publisher.lower() in root.lower() and repo.lower() in root.lower():
                if len(parts) == 3:
                    filename = parts[2]
                    full_path = os.path.join(root, filename)
                    if os.path.exists(full_path):
                        return full_path
                return root
        elif model_id.lower() in root.lower():
            return root

    return None


def extract_gguf_metadata(gguf_path):
    """
    Extracts deep architectural specs from a .gguf file using the gguf library.
    """
    path_to_check = gguf_path
    if not path_to_check.lower().endswith(".gguf"):
        if os.path.isdir(path_to_check):
            files = [f for f in os.listdir(path_to_check) if f.lower().endswith(".gguf")]
            if files:
                path_to_check = os.path.join(path_to_check, files[0])
            else:
                return None
        else:
            return None

    try:
        from gguf import GGUFReader

        reader = GGUFReader(path_to_check)

        def get_field(key, default=None):
            try:
                for field_key in reader.fields.keys():
                    if key in field_key:
                        value = reader.fields[field_key].parts[-1]
                        if isinstance(value, list):
                            return value[0]
                        return value
                return default
            except Exception:
                return default

        return {
            "hidden_size": int(get_field("embedding_length", 4096)),
            "num_layers": int(get_field("block_count", 32)),
            "num_heads": int(get_field("attention.head_count", 32)),
            "kv_heads": int(get_field("attention.head_count_kv", get_field("attention.head_count", 32))),
            "vocab_size": int(get_field("vocab_size", 32000)),
            "model_type": str(get_field("architecture", "llama")),
            "max_position": int(get_field("context_length", 32768)),
            "is_gguf": True,
        }
    except Exception:
        return None


def extract_model_specs(model_path):
    """
    Attempts to load architectural specs from GGUF first, then config.json.
    """
    if not model_path:
        return None

    specs = extract_gguf_metadata(model_path)
    if specs:
        return specs

    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                text_config = config.get("text_config", config)
                return {
                    "hidden_size": text_config.get("hidden_size"),
                    "num_layers": text_config.get("num_hidden_layers"),
                    "num_heads": text_config.get("num_attention_heads"),
                    "kv_heads": text_config.get("num_key_value_heads", text_config.get("num_attention_heads")),
                    "vocab_size": text_config.get("vocab_size"),
                    "model_type": text_config.get("model_type"),
                    "max_position": text_config.get("max_position_embeddings"),
                }
            except Exception:
                pass
    return None


if __name__ == "__main__":
    print("Listing local models...")
    for model in get_local_models():
        print(f"- {model['id']} ({model['type']}, {model['params']}, {model['arch']}, {model['size']})")
        path = get_model_path(model["id"])
        if path:
            print(f"  Path: {path}")
            specs = extract_model_specs(path)
            if specs:
                print(f"  Specs: {specs}")
