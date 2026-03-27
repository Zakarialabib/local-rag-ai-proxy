from typing import Dict, Optional
from models import QuantizationType, InferenceBackend

# Optional Numba JIT compiler for high-performance math operations
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba is not installed
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def calculate_weights_vram(params_b: float, bytes_per_param: float) -> float:
    """Calculate VRAM required for model weights in GB"""
    return params_b * bytes_per_param

@jit(nopython=True)
def calculate_kv_cache_vram(
    num_layers: int, 
    context_length: int, 
    num_heads: int, 
    head_dim: int, 
    bytes_per_token: float
) -> float:
    """Calculate VRAM required for KV cache in GB"""
    # 2 (Key + Value) * layers * ctx * heads * head_dim * bytes
    total_bytes = 2 * num_layers * context_length * num_heads * head_dim * bytes_per_token
    return total_bytes / (1024**3)

# Backend-specific runtime overhead (GB)
# These account for driver memory, CUDA context, Vulkan instance, etc.
BACKEND_OVERHEAD = {
    InferenceBackend.CUDA:    {"base": 1.2, "ctx_scale": 0.5, "maxwell_penalty": 0.0},
    InferenceBackend.VULKAN:  {"base": 1.5, "ctx_scale": 0.6},  # Vulkan has slightly higher driver overhead
    InferenceBackend.CLBLAST: {"base": 1.5, "ctx_scale": 0.6},  # Same as Vulkan
    InferenceBackend.METAL:   {"base": 0.5, "ctx_scale": 0.3},  # Unified memory = lower overhead
    InferenceBackend.CPU:     {"base": 0.3, "ctx_scale": 0.2},  # Minimal runtime overhead
    InferenceBackend.OPENCL:  {"base": 1.4, "ctx_scale": 0.5},
}

class VRAMCalculator:
    """Accurate VRAM estimation based on llama.cpp formulas."""
    
    @staticmethod
    def calculate(
        params_b: float,         
        quantization: QuantizationType,
        context_length: int,
        batch_size: int = 512,
        num_layers: int = 32,      
        hidden_size: int = 4096,
        num_heads: int = 32,
        kv_heads: Optional[int] = None,
        backend: InferenceBackend = InferenceBackend.CUDA,
        cuda_compute: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Formula breakdown:
        - Weights: params * bits_per_param / 8
        - KV Cache: 2 * layers * n_ctx * n_kv_heads * head_dim * precision
        - Activations: space for intermediate tensors
        - Overhead: backend-specific runtime memory
        """
        if kv_heads is None:
            kv_heads = num_heads
            
        head_dim = hidden_size // max(num_heads, 1)
        
        bits_map = {
            QuantizationType.Q4_0: 4.0,
            QuantizationType.Q4_K_M: 4.5, 
            QuantizationType.Q4_K_S: 4.2,
            QuantizationType.Q5_K_M: 5.5,
            QuantizationType.Q6_K: 6.6,
            QuantizationType.Q8_0: 8.5, 
            QuantizationType.FP16: 16.0,
        }
        
        bits = bits_map.get(quantization, 4.5)
        bytes_per_param = bits / 8
        
        # 1. Model weights
        weights_gb = calculate_weights_vram(params_b, bytes_per_param)
        
        # 2. KV cache (2 for K and V)
        kv_precision = 2.0  # FP16 KV cache
        kv_cache_gb = calculate_kv_cache_vram(
            num_layers, context_length, kv_heads, head_dim, kv_precision
        )
        
        # 3. Activations (rough estimate)
        activations_gb = (hidden_size * context_length * 4) / (1024**3)
        
        # 4. Backend-aware runtime overhead
        overhead_cfg = BACKEND_OVERHEAD.get(backend, BACKEND_OVERHEAD[InferenceBackend.CUDA])
        overhead_gb = overhead_cfg["base"] + (context_length / 32768) * overhead_cfg["ctx_scale"]
        
        # ADD THIS: Maxwell penalty (higher driver overhead)
        if cuda_compute and cuda_compute < 6.0:
            overhead_gb += 0.5  # Maxwell needs extra 500MB for driver overhead
            logger.debug("maxwell_vram_penalty_applied", extra_overhead=0.5)
        
        total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb
    
        return {
            "weights_gb": round(weights_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 2),
            "activations_gb": round(activations_gb, 2),
            "overhead_gb": round(overhead_gb, 2),
            "total_gb": round(total_gb, 2),
            "backend": backend.value,
            "is_maxwell": cuda_compute is not None and cuda_compute < 6.0, 
        }
