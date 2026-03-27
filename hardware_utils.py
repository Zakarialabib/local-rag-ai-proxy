import platform
import psutil
import subprocess
import sys
from rich.console import Console
from rich.table import Table

# Fix for Windows terminal emoji rendering
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
import re

def get_gpu_info():
    """
    Get detailed GPU info with multi-source fallback.
    """
    system = platform.system()
    gpus = []

    # Try nvidia-smi directly
    try:
        nv_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        if nv_out:
            for line in nv_out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": int(parts[1]),
                        "memory_used": int(parts[2]),
                        "driver": parts[3],
                        "load": f"{parts[4]}%",
                        "type": "NVIDIA"
                    })
    except Exception:
        pass

    if not gpus:
        if system == "Darwin":  # macOS (Metal/Unified Memory)
            try:
                output = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    stderr=subprocess.DEVNULL
                ).decode()
                
                name = "Apple Silicon"
                vram = "Shared"
                for line in output.splitlines():
                    if "Chipset Model" in line:
                        name = line.split(":")[1].strip()
                    if "VRAM" in line:
                        vram = line.split(":")[1].strip()
                
                gpus.append({"name": name, "memory_total": vram, "type": "Apple"})
            except Exception:
                pass

        elif system == "Windows":
            try:
                # Fallback to WMIC for non-NVIDIA or when GPUtil fails
                output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "AdapterRAM,Name"],
                    stderr=subprocess.DEVNULL
                ).decode()
                lines = [line.strip() for line in output.splitlines() if line.strip()]
                for line in lines[1:]:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        mem_mb = int(parts[0]) // (1024 * 1024)
                        gpus.append({"name": parts[1], "memory_total": mem_mb, "type": "Generic"})
            except Exception:
                pass

    # Detect CUDA version if available
    cuda_ver = "Not Found"
    try:
        nv_out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
        match = re.search(r"CUDA Version: (\d+\.\d+)", nv_out)
        if match:
            cuda_ver = match.group(1)
    except Exception:
        pass

    return gpus, cuda_ver

def get_system_info():
    cpu = platform.processor() or platform.machine()
    mem = psutil.virtual_memory()
    gpus, cuda_ver = get_gpu_info()
    
    gpu_names = [g["name"] for g in gpus] or ["None"]
    gpu_mem = [f"{g['memory_total']} MB" for g in gpus] or ["Unknown"]

    return {
        "CPU": cpu,
        "Physical Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
        "RAM Total (GB)": round(mem.total / (1024**3), 1),
        "GPU(s)": ", ".join(gpu_names),
        "GPU Memory": ", ".join(gpu_mem),
        "CUDA Version": cuda_ver,
        "OS": f"{platform.system()} {platform.release()}"
    }

def display_hardware_report():
    console = Console()
    info = get_system_info()
    
    table = Table(title="🚀 LM Studio Hardware Profile", title_style="bold cyan")
    table.add_column("Property", style="bold magenta")
    table.add_column("Value", style="green")
    
    for key, val in info.items():
        table.add_row(key, str(val))
    
    console.print(table)

def get_hardware_profile():
    """Return a simplified dict for the recommender."""
    info = get_system_info()
    
    # Use first GPU for primary metrics
    gpu_mem_mb = 0
    try:
        gpus, _ = get_gpu_info()
        if gpus:
             # Extract numeric memory if possible
             mem_str = str(gpus[0].get("memory_total", "0"))
             gpu_mem_mb = int(re.search(r"\d+", mem_str).group()) if re.search(r"\d+", mem_str) else 0
    except Exception:
        pass

    return {
        "cpu": info["CPU"],
        "logical_cores": info["Logical Cores"],
        "ram_gb": info["RAM Total (GB)"],
        "gpu": info["GPU(s)"],
        "gpu_memory_mb": gpu_mem_mb,
        "cuda_version": info["CUDA Version"]
    }

if __name__ == "__main__":
    display_hardware_report()

