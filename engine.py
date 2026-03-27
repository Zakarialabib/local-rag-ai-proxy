from typing import List, Optional
from itertools import product
from models import HardwareProfile, ModelRecommendation, QuantizationType, InferenceBackend
from vram_calculator import VRAMCalculator
import structlog

logger = structlog.get_logger()

USE_CASE_PROFILES = {
    "balanced": "⚖️ Balanced / General",
    "coding": "💻 Coding & Technical",
    "creative": "🎨 Creative & Storytelling",
    "logic": "🧠 Logic & Reasoning",
}

BACKEND_LABELS = {
    "cuda":    InferenceBackend.CUDA,
    "vulkan":  InferenceBackend.VULKAN,
    "clblast": InferenceBackend.CLBLAST,
    "metal":   InferenceBackend.METAL,
    "cpu":     InferenceBackend.CPU,
    "opencl":  InferenceBackend.OPENCL,
}

BACKEND_PROFILES = {
    InferenceBackend.CUDA:    {"quality_boost": 0.0,  "engine_name": "llama.cpp (CUDA)"},
    InferenceBackend.VULKAN:  {"quality_boost": -0.2, "engine_name": "llama.cpp (Vulkan)"},
    InferenceBackend.CLBLAST: {"quality_boost": -0.3, "engine_name": "llama.cpp (CLBlast)"},
    InferenceBackend.METAL:   {"quality_boost": 0.1,  "engine_name": "Apple Metal (Unified)"},
    InferenceBackend.CPU:     {"quality_boost": -0.5, "engine_name": "llama.cpp (CPU)"},
    InferenceBackend.OPENCL:  {"quality_boost": -0.4, "engine_name": "llama.cpp (OpenCL)"},
}


class RecommendationEngine:
    def __init__(self, hardware: HardwareProfile):
        self.hardware = hardware
        self.calculator = VRAMCalculator()
        
    def recommend(
        self, 
        model_id: str, 
        params_b: float,
        num_layers: int = 32,
        hidden_size: int = 4096,
        num_heads: int = 32,
        kv_heads: Optional[int] = None,
        min_quality: float = 5.0,
        use_case: str = "balanced",
        backend: str = "cuda",
        flash_attention: Optional[bool] = None,
    ) -> List[ModelRecommendation]:
        """Find optimal configurations given hardware constraints."""

        inf_backend = BACKEND_LABELS.get(backend, InferenceBackend.CUDA)
        
        # 1. Define base lists FIRST
        contexts = [2048, 4096, 8192, 16384, 32768, 65536]
        quants = list(QuantizationType)
        
        # 2. Detect Maxwell AFTER base lists exist
        is_maxwell_or_older = False
        if self.hardware.cuda_compute and self.hardware.cuda_compute < 6.0:
            is_maxwell_or_older = True
            logger.info("maxwell_gpu_detected", compute=self.hardware.cuda_compute)
            
            # 3. Apply constraints (now variables exist)
            max_context_for_gpu = 2048
            recommended_quants = [QuantizationType.Q4_0, QuantizationType.Q4_K_M]
            
            contexts = [c for c in contexts if c <= max_context_for_gpu]
            quants = [q for q in quants if q in recommended_quants]
        else:
            max_context_for_gpu = 32768 if (self.hardware.gpu_vram_gb or 0) > 8 else 8192
        
        # Force non-streaming for Maxwell (your benchmark proves 8.5s overhead)
        force_non_streaming = is_maxwell_or_older
        
        candidates = []
        
        if self.hardware.gpu_vram_gb and self.hardware.gpu_vram_gb > 0:
            gpu_vram = self.hardware.gpu_vram_gb
        else:
            gpu_vram = 0
            
        system_ram = self.hardware.system_ram_gb
        
        # Performance tuning based on CPU
        threads = max(1, self.hardware.cpu_cores - 1)
        if self.hardware.platform == "macos":
            # Apple Silicon performance cores are better for threads
            threads = self.hardware.cpu_cores
        
        # 4. Pass cuda_compute to calculator
        for quant, ctx in product(quants, contexts):
            vram_req = self.calculator.calculate(
                params_b=params_b,
                quantization=quant,
                context_length=ctx,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                kv_heads=kv_heads,
                backend=inf_backend,
                cuda_compute=self.hardware.cuda_compute,
            )
            
            total_gb = vram_req["total_gb"]
            
            if total_gb > (gpu_vram + system_ram * 0.8):
                continue
                
            # GPU Offload layers
            if gpu_vram > 0 and inf_backend != InferenceBackend.CPU:
                if total_gb <= gpu_vram * 0.95:
                    gpu_layers = 999  # Full offload
                elif vram_req["weights_gb"] <= gpu_vram * 0.9:
                    gpu_layers = int(num_layers * 0.8)
                else:
                    ratio = (gpu_vram * 0.8) / max(vram_req["weights_gb"], 0.1)
                    gpu_layers = max(1, int(num_layers * ratio))
            else:
                gpu_layers = 0
            
            quality = self._quality_score(quant, ctx)
            # Add backend-specific quality boost
            backend_cfg = BACKEND_PROFILES.get(inf_backend, {"quality_boost": 0.0, "engine_name": "llama.cpp"})
            quality += backend_cfg["quality_boost"]
            
            inf = self._get_inference_settings(model_id, use_case)
            
            # Determine flash attention support (Override if provided by GUI)
            if flash_attention is not None:
                flash_attn = flash_attention
            else:
                flash_attn = inf_backend in (InferenceBackend.CUDA, InferenceBackend.METAL)
            
            # KV Cache Quantization for Low VRAM or High Context
            kv_quant = "f16"
            if (gpu_vram < 8 and ctx > 8192) or ctx > 32768:
                kv_quant = "q4_0"
            elif gpu_vram < 12 and ctx > 16384:
                kv_quant = "q8_0"

            if quality >= min_quality:
                candidates.append(ModelRecommendation(
                    model_id=model_id,
                    quantization=quant,
                    context_length=ctx,
                    gpu_layers=gpu_layers,
                    estimated_vram_gb=total_gb,
                    quality_score=round(max(0.0, min(10.0, quality)), 2),
                    inference_backend=inf_backend,
                    inference_engine=backend_cfg["engine_name"],
                    flash_attention=flash_attn,
                    threads=threads,
                    batch_size=inf.pop("batch_size", 512),
                    kv_cache_quant=kv_quant,
                    use_mmap=True,
                    numa_support=self.hardware.cpu_cores > 16, # Rough heuristic for multi-socket
                    **inf,
                ))
        
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        logger.info("recommendations_generated", 
                    model=model_id,
                    use_case=use_case,
                    backend=inf_backend.value,
                    count=len(candidates),
                    top_score=candidates[0].quality_score if candidates else 0)
        
        return candidates[:5]
    def _get_inference_settings(self, model_id: str, use_case: str = "balanced") -> dict:
        """Context Engineering — model-aware and use-case-aware inference tuning."""
        
        settings = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "enable_thinking": False,
            "batch_size": 512,
            "system_prompt": "You are a helpful, accurate, and precise AI assistant.",
        }
        
        # 1. Use-Case Overrides
        if use_case == "coding":
            settings.update({
                "temperature": 0.2,
                "top_k": 30,
                "max_tokens": 4096,
                "batch_size": 256, # Smaller batch for low latency coding
                "system_prompt": (
                    "You are an expert software engineer and code reviewer. "
                    "Write clean, efficient, well-documented code. "
                    "Always consider edge cases, security implications, and performance. "
                    "When explaining, be concise and precise."
                ),
            })
        elif use_case == "creative":
            settings.update({
                "temperature": 1.0,
                "top_p": 0.99,
                "top_k": 100,
                "repeat_penalty": 1.0,
                "batch_size": 1024, # Larger batch for throughput in long stories
                "system_prompt": (
                    "You are a creative storyteller and world-class writer. "
                    "Craft rich narratives with vivid imagery, compelling characters, "
                    "and unexpected twists. Use varied sentence structure and engaging prose."
                ),
            })
        elif use_case == "logic":
            settings.update({
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 30,
                "enable_thinking": True,
                "batch_size": 512,
                "system_prompt": (
                    "You are a logical reasoning and analytical expert. "
                    "Break every complex problem into clear, numbered steps. "
                    "Identify assumptions, evaluate evidence, and present conclusions "
                    "with confidence levels. Use structured reasoning throughout."
                ),
            })

        # 2. Model-Specific Overrides (Higher Priority — these refine the use-case)
        m = model_id.lower()
        
        if "opus-reasoning" in m or "claude" in m or "deepseek-r1" in m:
            # Distilled reasoning model — always enable thinking
            settings["enable_thinking"] = True
            settings["repeat_penalty"] = 1.05
            settings["system_prompt"] = (
                "You are a specialized reasoning model. "
                "ALWAYS start your response with a <thinking> or <reasoning> block. "
                "Inside this block, perform a deep, multi-step analysis of the prompt. "
                "1. Break the problem into atomic components.\n"
                "2. Evaluate multiple solution paths.\n"
                "3. Verify logic for consistency.\n"
                "4. Finalize the optimal response structure.\n\n"
                "Format:\n"
                "<thinking>\n"
                "[Your internal chain-of-thought here]\n"
                "</thinking>\n\n"
                "[Your final, polished answer here]"
            )
            # Adjust temp based on use-case even for this model
            if use_case == "coding":
                settings["temperature"] = 0.2
                settings["max_tokens"] = 8192
            elif use_case == "creative":
                settings["temperature"] = 0.8
            elif use_case == "logic":
                settings["temperature"] = 0.3
            else:
                settings["temperature"] = 0.5
                
        elif "nemotron" in m:
            settings["top_p"] = 1.0  # NVIDIA recommends Top-P 1.0 for Nemotron
            settings["repeat_penalty"] = 1.0
            settings["system_prompt"] = (
                "You are a helpful AI assistant created by NVIDIA. "
                "Provide accurate, well-structured, and helpful answers. "
                "When solving problems, show your work clearly. "
                "Prefer factual precision over speculation."
            )
            if use_case == "coding":
                settings["temperature"] = 0.2
            elif use_case == "creative":
                settings["temperature"] = 0.8
            else:
                settings["temperature"] = 0.5
                
        elif "qwen" in m:
            # General Qwen models (non-opus)
            if use_case == "coding":
                settings["system_prompt"] = (
                    "You are Qwen, an expert AI coding assistant. "
                    "Write production-quality code with proper error handling, "
                    "type hints, and documentation. Follow language-specific best practices."
                )
            elif use_case == "logic":
                settings["enable_thinking"] = True
                settings["system_prompt"] = (
                    "You are Qwen, optimized for complex reasoning. "
                    "Use a structured <reasoning> approach to decompose the user request "
                    "before providing a definitive answer."
                )
            
        return settings

    def _quality_score(self, quant: QuantizationType, ctx: int) -> float:
        quant_scores = {
            QuantizationType.Q4_0: 6.0,
            QuantizationType.Q4_K_M: 7.5,
            QuantizationType.Q4_K_S: 7.0,
            QuantizationType.Q5_K_M: 8.5,
            QuantizationType.Q6_K: 9.0,
            QuantizationType.Q8_0: 9.5,
            QuantizationType.FP16: 10.0,
        }
        quant_score = quant_scores.get(quant, 5.0)
        ctx_score = 10.0 if ctx >= 32768 else (ctx / 32768) * 10
        return (quant_score * 0.6) + (ctx_score * 0.4)
