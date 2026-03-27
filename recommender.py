import math

def calculate_vram_estimate(model_profile, context_length, quantization):
    """
    Estimate VRAM usage based on architecture or general heuristics.
    """
    # Base quantization bits
    q_bits = 4.5 if "Q4_K_M" in quantization else (8.0 if "Q8_0" in quantization else 5.0)
    
    # If we have detailed specs (from config.json)
    specs = model_profile.get("specs")
    if specs:
        layers = specs.get("num_layers", 32)
        hidden_size = specs.get("hidden_size", 4096)
        heads = specs.get("num_heads", 32)
        kv_heads = specs.get("kv_heads", heads)
        head_dim = hidden_size // heads
        
        # 1. Weights memory (Approximate)
        # Using params if available, else estimate from size
        params_b = float(model_profile.get("params", "7").replace("B", ""))
        weight_mem_mb = (params_b * q_bits * 1024) / 8
        
        # 2. KV Cache memory (FP16 by default, or 0.5 for Q4_0)
        # Formula: 2 * layers * context * heads * head_dim * precision
        kv_precision = 2 # bytes
        kv_cache_mb = (2 * layers * context_length * kv_heads * head_dim * kv_precision) / (1024 * 1024)
        
        # 3. Activations + Overhead
        overhead_mb = 1000 # Standard overhead
        
        return weight_mem_mb + kv_cache_mb + overhead_mb
    else:
        # Heuristic fallback
        size_gb = 4.0 # Default
        size_str = model_profile.get("model_size", "4-8 GB")
        if "Under 2" in size_str: size_gb = 1.5
        elif "2-4" in size_str: size_gb = 3.0
        elif "4-8" in size_str: size_gb = 6.0
        elif "8-13" in size_str: size_gb = 10.0
        elif "More than 13" in size_str: size_gb = 20.0
        
        # Weights + KV Cache (estimate 20% of weights for KV cache at 4k context)
        return (size_gb * 1024) * 1.25 + 500

def recommend_settings(hardware_info: dict, model_profile: dict) -> dict:
    ram_gb = hardware_info.get("ram_gb", 8.0)
    gpu_mem_mb = hardware_info.get("gpu_memory_mb", 0)
    gpu_name = hardware_info.get("gpu", "")
    
    # Target context based on RAM but adjusted for VRAM
    context_length = 4096
    if ram_gb >= 32: context_length = 32768
    elif ram_gb >= 16: context_length = 8192
    
    quant = model_profile.get("quantization", "Q4_K_M")
    vram_req = calculate_vram_estimate(model_profile, context_length, quant)
    
    # GPU Offload strategy
    gpu_offload = 0
    if gpu_mem_mb > 0:
        if gpu_mem_mb > vram_req:
            gpu_offload = 999 # Full offload
        elif gpu_mem_mb > (vram_req * 0.5):
            gpu_offload = 16 # Partial (heuristic)
            
    # Optimization flags
    flash_attn = "Apple" in gpu_name or "NVIDIA" in gpu_name or gpu_mem_mb >= 8192
    
    # KV Cache Quantization (Suggestion if VRAM is tight)
    kv_quant = "none"
    if gpu_mem_mb > 0 and gpu_mem_mb < vram_req and gpu_mem_mb > (vram_req * 0.7):
        kv_quant = "q8_0" # Suggestions for future config export
    
    # Generation Settings
    goal = model_profile.get("goal", "Balanced")
    temp = 0.7; top_p = 0.9; top_k = 40
    if "Accuracy" in goal or "Coding" in model_profile.get("use_case", ""):
        temp = 0.3; top_p = 0.85; top_k = 50
    elif "Creativity" in goal:
        temp = 1.1; top_p = 0.95; top_k = 100

    return {
        "context_length": context_length,
        "gpu_offload": gpu_offload,
        "cpu_threads": min(hardware_info.get("logical_cores", 4), 8),
        "batch_size": 8 if gpu_mem_mb > 4000 else 4,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": 1.1,
        "max_tokens": 2048 if "Coding" in model_profile.get("use_case", "") else 1024,
        "keep_model_in_memory": ram_gb >= 24,
        "flash_attention": flash_attn,
        "try_mmap": True,
        "kv_cache_quant": kv_quant,
        "vram_estimated_mb": round(vram_req)
    }

