import platform
import psutil
import subprocess
import sys
import re
import warnings
import structlog
from typing import Dict, Optional

from models import HardwareProfile

logger = structlog.get_logger()

class HardwareDetector:
    """Modern hardware detection using native APIs with nvidia-ml-py."""
    
    def detect(self) -> HardwareProfile:
        plat = platform.system().lower()
        is_apple = False
        
        if plat == "darwin":
            plat = "macos"
            is_apple = platform.machine() == "arm64"
        
        cpu_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or cpu_cores
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        gpu_info = self._detect_gpu()
        cuda_ver = self._detect_cuda()
        
        profile = HardwareProfile(
            platform=plat,
            cpu_cores=cpu_cores,
            logical_cores=logical_cores,
            system_ram_gb=ram_gb,
            gpu_name=gpu_info.get("name"),
            gpu_vram_gb=gpu_info.get("vram_gb"),
            cuda_version=cuda_ver,
            is_apple_silicon=is_apple,
            cuda_compute=gpu_info.get("compute_capability")
        )
        
        logger.info("hardware_detected",
                     cpu_cores=cpu_cores,
                     ram_gb=ram_gb,
                     gpu=gpu_info.get("name", "None"),
                     vram_gb=gpu_info.get("vram_gb"),
                     compute=gpu_info.get("compute_capability"),
                     cuda=cuda_ver,
                     platform=plat)
        return profile
    
    def _detect_gpu(self) -> Dict[str, Optional[float]]:
        """Try nvidia-ml-py first, then nvidia-smi, then WMIC fallback."""
        gpu_info = {"name": None, "vram_gb": None, "compute_capability": None}
        
        # 1. nvidia-ml-py (successor to pynvml, no deprecation warnings)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex
                from pynvml import nvmlDeviceGetName, nvmlDeviceGetMemoryInfo
                from pynvml import nvmlDeviceGetCudaComputeCapability
            
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            
            # Get name
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            
            # Get VRAM
            mem = nvmlDeviceGetMemoryInfo(handle)
            vram_gb = round(mem.total / (1024**3), 2)
            
            # Get compute capability (Major.Minor)
            major, minor = nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = float(f"{major}.{minor}")
            
            nvmlShutdown()
            
            gpu_info = {
                "name": name,
                "vram_gb": vram_gb,
                "compute_capability": compute_capability
            }
            logger.debug("gpu_detected_via_nvidia_ml_py", 
                        name=name, vram_gb=vram_gb, compute=compute_capability)
            return gpu_info
            
        except ImportError:
            logger.debug("nvidia_ml_py_not_installed_trying_pynvml")
            # Fallback to pynvml if nvidia-ml-py not available
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                
                gpu_info = {
                    "name": name,
                    "vram_gb": round(mem.total / (1024**3), 2),
                    "compute_capability": float(f"{major}.{minor}")
                }
                pynvml.nvmlShutdown()
                return gpu_info
            except Exception as e:
                logger.debug("pynvml_fallback_failed", error=str(e))
        except Exception as e:
            logger.debug("nvidia_ml_py_failed", error=str(e))
        
        # 2. nvidia-smi fallback (ADD compute capability detection)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                if len(parts) >= 3:
                    return {
                        "name": parts[0], 
                        "vram_gb": round(int(parts[1]) / 1024, 2),
                        "compute_capability": float(parts[2])
                    }
        except Exception:
            pass
        
        # 3. WMIC fallback (Windows only)
        if platform.system() == "Windows":
            try:
                out = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "AdapterRAM,Name"],
                    stderr=subprocess.DEVNULL
                ).decode()
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                if len(lines) > 1:
                    parts = lines[1].split(maxsplit=1)
                    if len(parts) == 2:
                        vram_gb = round(int(parts[0]) / (1024**3), 2)
                        return {"name": parts[1], "vram_gb": vram_gb}
            except Exception:
                pass
        
        # 4. macOS
        if platform.system() == "Darwin":
            return {"name": "Apple Silicon (Unified Memory)", "vram_gb": None}
        
        return {}
    
    def _detect_cuda(self) -> Optional[str]:
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
            match = re.search(r"CUDA Version: (\d+\.\d+)", out)
            if match:
                return match.group(1)
        except Exception:
            pass
        return None


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
    )
    detector = HardwareDetector()
    profile = detector.detect()
    print(profile.model_dump_json(indent=2))
