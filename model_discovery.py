import subprocess
import json
import os
import re

def get_local_models():
    """
    Get a list of local models using the 'lms ls' CLI tool.
    Returns a list of dictionaries with model info.
    """
    try:
        output = subprocess.check_output(["lms", "ls"], stderr=subprocess.DEVNULL).decode()
        models = []
        lines = output.strip().splitlines()
        
        # Skip header lines until we find LLM or EMBEDDING
        start_index = -1
        for i, line in enumerate(lines):
            if "LLM" in line and "PARAMS" in line:
                start_index = i + 1
                break
        
        if start_index != -1:
            for line in lines[start_index:]:
                if not line.strip() or "EMBEDDING" in line:
                    break
                
                # Split by multiple spaces
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 4:
                    models.append({
                        "id": parts[0],
                        "params": parts[1],
                        "arch": parts[2],
                        "size": parts[3],
                        "loaded": "✓" in line
                    })
        return models
    except Exception:
        return []

def get_model_path(model_id):
    """
    Attempts to find the physical path to a model folder or .gguf file.
    """
    user_home = os.path.expanduser("~")
    base_cache_path = os.path.join(user_home, ".cache", "lm-studio", "models")
    
    # 1. First, look for a directory matching the model_id
    for root, dirs, files in os.walk(base_cache_path):
        # The model_id in lms ls is often "Publisher/ModelRepo/File.gguf"
        # We need to handle this slash-based structure.
        parts = model_id.split('/')
        if len(parts) >= 2:
            publisher, repo = parts[0], parts[1]
            if publisher.lower() in root.lower() and repo.lower() in root.lower():
                # If there's a specific file mentioned in model_id, look for it.
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
    if not path_to_check.lower().endswith('.gguf'):
        # If it's a directory, look for .gguf files inside
        if os.path.isdir(path_to_check):
            files = [f for f in os.listdir(path_to_check) if f.lower().endswith('.gguf')]
            if files:
                path_to_check = os.path.join(path_to_check, files[0])
            else:
                return None
        else:
            return None

    try:
        from gguf import GGUFReader
        reader = GGUFReader(path_to_check)
        
        # Helper to get field with fallback
        def get_field(key, default=None):
            try:
                # GGUF keys often follow llama.XXX or general.XXX
                for k in reader.fields.keys():
                    if key in k:
                        val = reader.fields[k].parts[-1]
                        if isinstance(val, list):
                            return val[0]
                        return val
                return default
            except:
                return default

        return {
            "hidden_size": int(get_field("embedding_length", 4096)),
            "num_layers": int(get_field("block_count", 32)),
            "num_heads": int(get_field("attention.head_count", 32)),
            "kv_heads": int(get_field("attention.head_count_kv", get_field("attention.head_count", 32))),
            "vocab_size": int(get_field("vocab_size", 32000)),
            "model_type": str(get_field("architecture", "llama")),
            "max_position": int(get_field("context_length", 32768)),
            "is_gguf": True
        }
    except Exception as e:
        return None

def extract_model_specs(model_path):
    """
    Attempts to load architectural specs from GGUF first, then config.json.
    """
    if not model_path: return None
    
    # Try GGUF first (it's what we actually run)
    specs = extract_gguf_metadata(model_path)
    if specs: return specs
    
    # Fallback to config.json if directory
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                text_config = config.get("text_config", config)
                return {
                    "hidden_size": text_config.get("hidden_size"),
                    "num_layers": text_config.get("num_hidden_layers"),
                    "num_heads": text_config.get("num_attention_heads"),
                    "kv_heads": text_config.get("num_key_value_heads", text_config.get("num_attention_heads")),
                    "vocab_size": text_config.get("vocab_size"),
                    "model_type": text_config.get("model_type"),
                    "max_position": text_config.get("max_position_embeddings")
                }
            except:
                pass
    return None

if __name__ == "__main__":
    print("Listing local models...")
    for m in get_local_models():
        print(f"- {m['id']} ({m['params']}, {m['arch']}, {m['size']})")
        path = get_model_path(m['id'])
        if path:
            print(f"  Path: {path}")
            specs = extract_model_specs(path)
            if specs:
                print(f"  Specs: {specs}")
