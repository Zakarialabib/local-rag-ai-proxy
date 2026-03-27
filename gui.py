"""
LM Studio Config Wizard — V3.1 MODERN GUI
CustomTkinter-based GUI with hardware caching, backend selection,
copyable VRAM reports, and full Context Engineering previews.
"""

import os
import json
import time
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from hardware_detector import HardwareDetector
from engine import RecommendationEngine, USE_CASE_PROFILES, BACKEND_LABELS
from model_discovery import get_local_models, get_model_path, extract_model_specs
from exporters import ConfigExporter
from models import ModelRecommendation, InferenceBackend

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ── Instruction Info Content ──────────────────────────────────────────────────
INSTRUCTION_INFO_TEXT = """\
═══════════════════════════════════════════════════════════
  LM STUDIO CONFIG WIZARD — QUICK START GUIDE
═══════════════════════════════════════════════════════════

HOW TO APPLY YOUR PRESET
─────────────────────────
1. Export your preferred recommendation using the button below.
2. The .json preset is saved to:
   ~/.cache/lm-studio/config-presets/
3. Open LM Studio → Click the model dropdown → Presets.
4. Your "Optimized" preset will appear in the list.

WHAT THE SETTINGS MEAN
──────────────────────
• GPU Offload (gpu_layers):
  Number of model layers loaded onto your GPU.
  "Full Offload" = entire model in VRAM = fastest inference.
  "Partial" = some layers on CPU RAM = slower but fits larger models.

• Flash Attention:
  Hardware-accelerated attention mechanism.
  Supported on NVIDIA (Compute ≥ 7.0) and Apple Metal.
  Reduces VRAM usage and speeds up long-context inference.

• Context Length:
  Maximum number of tokens the model can "see" at once.
  Higher = more memory. 8192 is a good default for most tasks.
  32768+ recommended only if VRAM allows or for deep research.

• KV Cache Quantization (V3.2):
  Reduces memory footprint of the context.
  "f16" (Best quality) | "q8_0" (Balanced) | "q4_0" (Best performance).
  Automatically set based on your VRAM and context length.

• Threads & Batch Size:
  Optimized for your CPU.
  Coding = Small batch (low latency).
  Creative = Large batch (high throughput).

• Temperature:
  Controls randomness. Low (0.1-0.3) = deterministic/coding.
  Medium (0.5-0.7) = balanced. High (0.8-1.2) = creative.

• Top-P / Top-K:
  Sampling strategies. Top-P limits cumulative probability,
  Top-K limits the number of candidate tokens.

• Enable Thinking:
  Activates <reasoning> tags for chain-of-thought models.
  Used by Opus-Reasoning distilled models.

INFERENCE BACKENDS
──────────────────
• CUDA    — Best for NVIDIA GPUs. Lowest overhead, Flash Attention support.
• Vulkan  — Cross-platform (AMD, Intel, NVIDIA). Slightly higher overhead.
• CLBlast — OpenCL-based compute (alternative to Vulkan).
• Metal   — Apple Silicon. Unified memory = very efficient.
• CPU     — Pure CPU inference. No VRAM needed, but much slower.
• OpenCL  — Legacy GPU compute. Use Vulkan if available.

TIPS FOR LOW-VRAM SYSTEMS (< 6 GB)
───────────────────────────────────
• Use Q4_0 or Q4_K_S quantization for smallest memory footprint.
• Keep context length at 4096 or below.
• Enable Flash Attention to reduce KV cache memory.
• Consider partial GPU offload instead of forcing full offload.
"""


class LMStudioConfigGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LM Studio Config Wizard v3.1")
        self.geometry("1140x760")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── State ─────────────────────────────────────────────────────────────
        self.detector = HardwareDetector()
        self._cached_hardware = None       # Hardware cache
        self._model_specs_cache = {}       # {model_id: specs_dict}
        self.engine = None
        self.models = []
        self.recommendations = []
        self.selected_rec = None
        self._active_backend_key = "cuda"
        
        # Server State
        self.proxy_process = None
        self.model_roles = {
            "main": "qwen3.5-4b",
            "reasoning": "qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2",
            "embed": "text-embedding-nomic-embed-text-v1.5",
            "rerank": "text-embedding-bge-reranker-v2-m3"
        }

        # ── Sidebar ───────────────────────────────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="⚡ LMStudio Wizard",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 5))

        ctk.CTkLabel(
            self.sidebar, text="v3.1 — Best Configuration Engine",
            font=ctk.CTkFont(size=11), text_color="#8b949e",
        ).grid(row=1, column=0, padx=20, pady=(0, 15))

        ctk.CTkLabel(
            self.sidebar, text="SYSTEM PROFILE",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff",
        ).grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.hw_info_box = ctk.CTkTextbox(
            self.sidebar, width=240, height=200, corner_radius=10,
            font=("Consolas", 12),
        )
        self.hw_info_box.grid(row=3, column=0, padx=20, pady=10)
        self.hw_info_box.configure(state="disabled")

        self.refresh_btn = ctk.CTkButton(
            self.sidebar, text="🔄 Recheck Hardware",
            command=self._force_recheck_hardware,
            fg_color="transparent", border_width=1,
        )
        self.refresh_btn.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="n")

        # Appearance mode at bottom
        ctk.CTkLabel(
            self.sidebar, text="Appearance:", anchor="w",
        ).grid(row=5, column=0, padx=20, pady=(10, 0), sticky="sw")

        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["Dark", "Light", "System"],
            command=lambda m: ctk.set_appearance_mode(m),
        )
        self.appearance_menu.grid(row=6, column=0, padx=20, pady=(5, 20), sticky="sw")
        self.appearance_menu.set("Dark")

        # ── Main Content ──────────────────────────────────────────────────────
        self.main_frame = ctk.CTkScrollableFrame(self, bg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Selection Row (Model + Use-Case + Backend)
        sel = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        sel.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        sel.grid_columnconfigure(0, weight=3)
        sel.grid_columnconfigure(1, weight=1)
        sel.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(sel, text="Model", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, sticky="w", padx=5)
        self.model_menu = ctk.CTkOptionMenu(sel, values=["Scanning..."], width=420, dynamic_resizing=False, command=self._on_model_select)
        self.model_menu.grid(row=1, column=0, sticky="ew", padx=5, pady=3)

        ctk.CTkLabel(sel, text="Use-Case", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=1, sticky="w", padx=5)
        self.usecase_menu = ctk.CTkOptionMenu(sel, values=list(USE_CASE_PROFILES.values()), width=220, command=self._on_change)
        self.usecase_menu.grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        self.usecase_menu.set(USE_CASE_PROFILES["balanced"])

        ctk.CTkLabel(sel, text="Backend", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=2, sticky="w", padx=5)
        backend_labels = ["CUDA", "Vulkan", "CLBlast", "Metal", "CPU", "OpenCL"]
        self.backend_menu = ctk.CTkOptionMenu(sel, values=backend_labels, width=140, command=self._on_backend_change)
        self.backend_menu.grid(row=1, column=2, sticky="ew", padx=5, pady=3)
        self.backend_menu.set("CUDA")

        # Results Cards
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=1, column=0, sticky="ew", pady=10)
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.results_title = ctk.CTkLabel(
            self.results_frame, text="GOLD STANDARD RECOMMENDATIONS",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#7ee787",
        )
        self.results_title.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        self.recs_scroll = ctk.CTkScrollableFrame(self.results_frame, height=250, fg_color="transparent")
        self.recs_scroll.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.recs_scroll.grid_columnconfigure(0, weight=1)

        # Tabs
        self.tabview = ctk.CTkTabview(self.main_frame, height=300)
        self.tabview.grid(row=2, column=0, sticky="ew", pady=15)
        self.tabview.add("� Server & Models")
        self.tabview.add("� VRAM Breakdown")
        self.tabview.add("🧠 Context Engineering")
        self.tabview.add("📋 Instructions & Guide")

        # 4. Action Banner
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(self.action_frame, text="Starting up...", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=20)

        self.export_btn = ctk.CTkButton(
            self.action_frame, text="💾 EXPORT PRESET TO LM STUDIO",
            state="disabled", fg_color="#2ea44f", hover_color="#3fb950",
            font=ctk.CTkFont(size=14, weight="bold"), command=self._export_selected,
        )
        self.export_btn.pack(side="right", padx=20, pady=10)

        # Flash Attention Toggle (V3.1)
        self.fa_var = tk.BooleanVar(value=True)
        self.fa_checkbox = ctk.CTkCheckBox(self.action_frame, text="Force Flash Attention", variable=self.fa_var, command=self._on_change)
        self.fa_checkbox.pack(side="right", padx=20)

        # Populate static tab
        self._populate_instruction_tab()
        self._populate_server_tab()

        # Deferred init
        self.after(300, self._initial_load)

    # ══════════════════════════════════════════════════════════════════════════
    #   UI HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _populate_instruction_tab(self):
        tab = self.tabview.tab("📋 Instructions & Guide")
        txt = ctk.CTkTextbox(tab, font=("Consolas", 12), wrap="word")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", INSTRUCTION_INFO_TEXT)
        txt.configure(state="disabled")

    def _populate_server_tab(self):
        tab = self.tabview.tab("🚀 Server & Models")
        
        # Server Control Frame
        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        self.btn_start_server = ctk.CTkButton(ctrl_frame, text="▶ Start Proxy Server", 
                                             fg_color="#2ea44f", hover_color="#3fb950",
                                             command=self._toggle_server)
        self.btn_start_server.pack(side="left", padx=5)
        
        self.server_status_lbl = ctk.CTkLabel(ctrl_frame, text="Status: Stopped", text_color="#8b949e")
        self.server_status_lbl.pack(side="left", padx=15)
        
        # Model Mapping Frame
        map_frame = ctk.CTkFrame(tab)
        map_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(map_frame, text="Model Routing Map (Saved to .env)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        self.role_vars = {}
        roles = [
            ("Main LLM", "main", self.model_roles["main"]),
            ("Reasoning", "reasoning", self.model_roles["reasoning"]),
            ("Embedder", "embed", self.model_roles["embed"]),
            ("Reranker", "rerank", self.model_roles["rerank"]),
        ]
        
        for i, (label_text, key, default_val) in enumerate(roles):
            ctk.CTkLabel(map_frame, text=label_text).grid(row=i+1, column=0, sticky="w", padx=20, pady=5)
            var = tk.StringVar(value=default_val)
            self.role_vars[key] = var
            # We'll update the values of these menus after scanning models
            menu = ctk.CTkOptionMenu(map_frame, variable=var, values=[default_val], width=300)
            menu.grid(row=i+1, column=1, sticky="ew", padx=20, pady=5)
            
        save_btn = ctk.CTkButton(map_frame, text="💾 Save Configuration", command=self._save_model_config)
        save_btn.grid(row=len(roles)+1, column=0, columnspan=2, pady=15)

    def _save_model_config(self):
        env_path = Path(".env")
        lines = []
        if env_path.exists():
            with open(env_path, "r") as f:
                lines = f.readlines()
                
        # Filter out old keys
        keys_to_update = ["MAIN_MODEL", "REASONING_MODEL", "EMBED_MODEL", "RERANK_MODEL"]
        lines = [l for l in lines if not any(l.startswith(k) for k in keys_to_update)]
        
        # Add new keys
        lines.append(f"MAIN_MODEL={self.role_vars['main'].get()}\n")
        lines.append(f"REASONING_MODEL={self.role_vars['reasoning'].get()}\n")
        lines.append(f"EMBED_MODEL={self.role_vars['embed'].get()}\n")
        lines.append(f"RERANK_MODEL={self.role_vars['rerank'].get()}\n")
        
        with open(env_path, "w") as f:
            f.writelines(lines)
            
        messagebox.showinfo("Saved", "Configuration saved to .env\nRestart the server to apply changes.")

    def _toggle_server(self):
        if self.proxy_process is None:
            # Start
            try:
                env = os.environ.copy()
                env["MAIN_MODEL"] = self.role_vars["main"].get()
                env["REASONING_MODEL"] = self.role_vars["reasoning"].get()
                env["EMBED_MODEL"] = self.role_vars["embed"].get()
                env["RERANK_MODEL"] = self.role_vars["rerank"].get()
                
                # Use subprocess to start proxy.py
                self.proxy_process = subprocess.Popen(
                    ["python", "proxy.py"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                self.btn_start_server.configure(text="⏹ Stop Proxy Server", fg_color="#d73a49", hover_color="#cb2431")
                self.server_status_lbl.configure(text="Status: Running (Port 8080)", text_color="#2ea44f")
                
                # Thread to consume output so it doesn't block
                def monitor_output(proc):
                    for line in iter(proc.stdout.readline, ''):
                        pass # Could pipe to a GUI log window later
                threading.Thread(target=monitor_output, args=(self.proxy_process,), daemon=True).start()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start server: {e}")
                self.proxy_process = None
        else:
            # Stop
            self.proxy_process.terminate()
            self.proxy_process = None
            self.btn_start_server.configure(text="▶ Start Proxy Server", fg_color="#2ea44f", hover_color="#3fb950")
            self.server_status_lbl.configure(text="Status: Stopped", text_color="#8b949e")

    def _copy_to_clipboard(self, text, source_name):
        self.clipboard_clear()
        self.clipboard_append(text)
        old_status = self.status_label.cget("text")
        self.status_label.configure(text=f"Copied {source_name} to clipboard!")
        self.after(2000, lambda: self.status_label.configure(text=old_status))

    # ══════════════════════════════════════════════════════════════════════════
    #   LIFECYCLE & CACHING
    # ══════════════════════════════════════════════════════════════════════════

    def _initial_load(self):
        self._detect_hardware_cached()
        self._load_models()

    def _detect_hardware_cached(self):
        if self._cached_hardware is None:
            t0 = time.time()
            self._cached_hardware = self.detector.detect()
            dt = time.time() - t0
            
            # Auto-detect best backend
            hw = self._cached_hardware
            if hw.platform == "macos" and hw.is_apple_silicon:
                self._active_backend_key = "metal"
            elif hw.gpu_name and ("nvidia" in hw.gpu_name.lower() or hw.cuda_version):
                self._active_backend_key = "cuda"
            elif hw.gpu_name:
                self._active_backend_key = "vulkan"
            else:
                self._active_backend_key = "cpu"
                
            self.backend_menu.set(self._active_backend_key.capitalize())
            self.status_label.configure(text=f"Hardware detected in {dt:.1f}s")
            
        self.engine = RecommendationEngine(self._cached_hardware)
        self._render_hw_sidebar()

    def _force_recheck_hardware(self):
        self._cached_hardware = None
        self.status_label.configure(text="Re-scanning hardware...")
        self.update_idletasks()
        self._detect_hardware_cached()
        if self.models:
            self._run_optimization()

    def _render_hw_sidebar(self):
        hw = self._cached_hardware
        lines = [
            f"Platform : {hw.platform.upper()}",
            f"CPU      : {hw.cpu_cores}P / {hw.logical_cores}L cores",
            f"RAM      : {hw.system_ram_gb:.1f} GB",
            f"GPU      : {hw.gpu_name or 'Not Found'}",
        ]
        if hw.gpu_vram_gb:
            lines.append(f"VRAM     : {hw.gpu_vram_gb:.1f} GB")
        if hw.cuda_version:
            lines.append(f"CUDA     : {hw.cuda_version}")
        if hw.is_apple_silicon:
            lines.append("Apple M  : Unified Memory")
        
        self.hw_info_box.configure(state="normal")
        self.hw_info_box.delete("1.0", tk.END)
        self.hw_info_box.insert(tk.END, "\n".join(lines))
        self.hw_info_box.configure(state="disabled")

    def _load_models(self):
        self.models = get_local_models()
        if not self.models:
            self.model_menu.configure(values=["No models found — launch LM Studio"])
            return

        names = [m["id"] for m in self.models]
        self.model_menu.configure(values=names)
        self.model_menu.set(names[0])
        self._on_model_select(names[0])
        
        # Update server map dropdowns
        tab = self.tabview.tab("🚀 Server & Models")
        for child in tab.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ctk.CTkOptionMenu):
                        subchild.configure(values=names)

    def _get_specs_cached(self, model_id: str) -> dict:
        if model_id not in self._model_specs_cache:
            m_path = get_model_path(model_id)
            self._model_specs_cache[model_id] = extract_model_specs(m_path) or {}
        return self._model_specs_cache[model_id]

    # ══════════════════════════════════════════════════════════════════════════
    #   HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _on_model_select(self, model_id):
        self._run_optimization()

    def _on_change(self, _val=None):
        self._run_optimization()

    def _on_backend_change(self, label):
        self._active_backend_key = label.lower()
        self._run_optimization()

    # ══════════════════════════════════════════════════════════════════════════
    #   OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _run_optimization(self):
        model_id = self.model_menu.get()
        if not model_id or model_id.startswith("No models"): return

        uc_label = self.usecase_menu.get()
        uc_key = next((k for k, v in USE_CASE_PROFILES.items() if v == uc_label), "balanced")

        m_info = next((m for m in self.models if m["id"] == model_id), None)
        if not m_info: return

        try:
            params_b = float("".join(c for c in m_info.get("params", "7") if c.isdigit() or c == "."))
        except:
            params_b = 7.0

        specs = self._get_specs_cached(model_id)

        t0 = time.time()
        self.recommendations = self.engine.recommend(
            model_id=model_id,
            params_b=params_b,
            num_layers=specs.get("num_layers", 32),
            hidden_size=specs.get("hidden_size", 4096),
            num_heads=specs.get("num_heads", 32),
            kv_heads=specs.get("kv_heads"),
            use_case=uc_key,
            backend=self._active_backend_key,
            flash_attention=self.fa_var.get()
        )
        dt = time.time() - t0

        self._render_cards()

        if self.recommendations:
            self.selected_rec = self.recommendations[0]
            self._render_vram_tab()
            self._render_context_tab()
            self.export_btn.configure(state="normal")
            self.status_label.configure(text=f"✅ {len(self.recommendations)} configs ({dt:.2f}s)")
        else:
            self.export_btn.configure(state="disabled")
            self.status_label.configure(text="⚠ No fits found.")

    def _render_cards(self):
        for w in self.recs_scroll.winfo_children(): w.destroy()
        if not self.recommendations:
            ctk.CTkLabel(self.recs_scroll, text="No fits found.").pack(pady=20)
            return

        for idx, rec in enumerate(self.recommendations):
            is_top = idx == 0
            card = ctk.CTkFrame(self.recs_scroll, corner_radius=10, 
                               fg_color="#1e293b" if is_top else "transparent",
                               border_width=2 if is_top else 1,
                               border_color="#58a6ff" if is_top else "#30363d")
            card.pack(fill="x", padx=5, pady=4)
            card.bind("<Button-1>", lambda e, r=rec: self._select_rec(r))

            rank = f"{'⭐ #1' if is_top else f'#{idx+1}'}"
            info = f"{rank} | {rec.quantization.value} | {rec.context_length:,} ctx | {rec.estimated_vram_gb:.1f} GB"
            lbl = ctk.CTkLabel(card, text=info, font=ctk.CTkFont(size=14, weight="bold" if is_top else "normal"))
            lbl.pack(side="left", padx=15, pady=12)
            lbl.bind("<Button-1>", lambda e, r=rec: self._select_rec(r))

    def _select_rec(self, rec):
        self.selected_rec = rec
        self._render_vram_tab()
        self._render_context_tab()

    def _render_vram_tab(self):
        tab = self.tabview.tab("📊 VRAM Breakdown")
        for w in tab.winfo_children(): w.destroy()
        rec = self.selected_rec
        if not rec: return

        from vram_calculator import VRAMCalculator
        # Note: In a real app we'd fetch params_b again, here we assume it's stored or we use a fallback
        m_info = next((m for m in self.models if m["id"] == rec.model_id), {})
        params_b = float("".join(c for c in m_info.get("params", "7") if c.isdigit() or c == "."))
        
        breakdown = VRAMCalculator.calculate(params_b, rec.quantization, rec.context_length, backend=rec.inference_backend)
        total = breakdown["total_gb"]
        
        lines = [
            f"VRAM REPORT — {rec.model_id}",
            f"Backend: {rec.inference_backend.value} | Total: {total:.2f} GB",
            "─" * 40,
            f"Weights:     {breakdown['weights_gb']:.2f} GB",
            f"KV Cache:    {breakdown['kv_cache_gb']:.2f} GB",
            f"Activations: {breakdown['activations_gb']:.2f} GB",
            f"Overhead:    {breakdown['overhead_gb']:.2f} GB",
            "─" * 40,
        ]
        report = "\n".join(lines)

        btn_container = ctk.CTkFrame(tab, fg_color="transparent")
        btn_container.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(btn_container, text="📋 Copy Report", width=120, height=24,
                     command=lambda: self._copy_to_clipboard(report, "Report")).pack(side="right")

        txt = ctk.CTkTextbox(tab, font=("Consolas", 12))
        txt.pack(fill="both", expand=True, padx=10, pady=5)
        txt.insert("1.0", report)
        txt.configure(state="disabled")

    def _render_context_tab(self):
        tab = self.tabview.tab("🧠 Context Engineering")
        for w in tab.winfo_children(): w.destroy()
        rec = self.selected_rec
        if not rec: return

        # Upper row: Parameter chips
        params_frame = ctk.CTkFrame(tab, fg_color="transparent")
        params_frame.pack(fill="x", padx=20, pady=(10, 5))
        
        # Helper to create small parameter labels
        def add_chip(label, value):
            chip = ctk.CTkFrame(params_frame, fg_color="#30363d", corner_radius=12)
            chip.pack(side="left", padx=5)
            ctk.CTkLabel(chip, text=f"{label}: {value}", font=ctk.CTkFont(size=11, weight="bold"), height=24).pack(padx=10)

        add_chip("Temp", rec.temperature)
        add_chip("Top-P", rec.top_p)
        add_chip("Top-K", rec.top_k)
        add_chip("Penalty", rec.repeat_penalty)
        add_chip("Ctx", rec.context_length)
        add_chip("Threads", rec.threads)
        add_chip("Batch", rec.batch_size)
        add_chip("KV Cache", rec.kv_cache_quant.upper())
        
        if rec.enable_thinking:
            chip = ctk.CTkFrame(params_frame, fg_color="#2ea44f", corner_radius=12)
            chip.pack(side="left", padx=5)
            ctk.CTkLabel(chip, text="🧠 Reasoning Enabled", font=ctk.CTkFont(size=11, weight="bold"), height=24).pack(padx=10)
        
        if rec.use_mmap:
            add_chip("MMAP", "ON")

        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(header, text="SYSTEM PROMPT", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="📋 Copy Prompt", width=120, height=24,
                     command=lambda: self._copy_to_clipboard(rec.system_prompt, "Prompt")).pack(side="right")

        txt = ctk.CTkTextbox(tab, height=180, font=("Consolas", 12), wrap="word")
        txt.pack(fill="both", expand=True, padx=20, pady=(5, 15))
        txt.insert("1.0", rec.system_prompt or "")
        txt.configure(state="disabled")

    def _export_selected(self):
        if not self.selected_rec: return
        try:
            target_dir = Path.home() / ".cache" / "lm-studio" / "config-presets"
            safe_id = self.selected_rec.model_id.replace("/", "_")
            target_path = target_dir / f"{safe_id}_v3.json"
            ConfigExporter.export_preset(self.selected_rec, target_path)
            messagebox.showinfo("Success", f"Preset exported to:\n{target_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = LMStudioConfigGUI()
    app.mainloop()
