from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict
from enum import Enum

class QuantizationType(str, Enum):
    Q4_0 = "Q4_0"      # Fastest, lowest quality
    Q4_K_M = "Q4_K_M"  # Balanced
    Q4_K_S = "Q4_K_S"  # Small, good quality
    Q5_K_M = "Q5_K_M"  # Better quality
    Q6_K = "Q6_K"      # High quality
    Q8_0 = "Q8_0"      # Best quality
    FP16 = "FP16"      # Full precision

class InferenceBackend(str, Enum):
    CUDA = "CUDA"           # NVIDIA GPUs (default for discrete NVIDIA)
    VULKAN = "Vulkan"       # Cross-platform GPU compute (AMD, Intel, NVIDIA)
    CLBLAST = "CLBlast"     # OpenCL-based compute (alternative to Vulkan)
    METAL = "Metal"         # Apple Silicon / macOS
    CPU = "CPU"             # Pure CPU inference (no GPU)
    OPENCL = "OpenCL"       # Legacy GPU compute

class HardwareProfile(BaseModel):
    model_config = {"frozen": True}  # Immutable 
    
    platform: Literal["windows", "linux", "macos"]
    cpu_cores: int = Field(gt=0)
    logical_cores: int = Field(gt=0)
    system_ram_gb: float = Field(gt=0)
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    cuda_compute: Optional[float] = None  # Compute capability
    cuda_version: Optional[str] = None
    is_apple_silicon: bool = False
    
    @field_validator("gpu_vram_gb")
    @classmethod
    def validate_vram(cls, v):
        if v and v < 0.5:
             # Allowing low VRAM but warning should be handled in recommendation
            return v
        return v

class ModelRecommendation(BaseModel):
    model_id: str
    quantization: QuantizationType
    context_length: int = Field(ge=512, le=262144)
    gpu_layers: int = Field(ge=0)
    estimated_vram_gb: float
    quality_score: float = Field(ge=0, le=10)
    
    # Inference / Context Engineering (V3)
    system_prompt: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    enable_thinking: bool = False
    
    # Backend & Performance (V3.2)
    inference_backend: InferenceBackend = InferenceBackend.CUDA
    inference_engine: str = "llama.cpp"
    flash_attention: bool = True
    threads: int = 4
    batch_size: int = 512
    kv_cache_quant: Literal["f16", "q4_0", "q8_0"] = "f16"
    use_mmap: bool = True
    numa_support: bool = False
    
    @property
    def is_offload_full(self) -> bool:
        """Check if model fits entirely in GPU VRAM"""
        return self.gpu_layers >= 999 or self.gpu_layers >= 32

