"""
LM Studio Agent Console V6
"""

import asyncio
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from urllib.parse import urlparse

import customtkinter as ctk
import httpx

from benchmark import StreamingBenchmark
from engine import RecommendationEngine, USE_CASE_PROFILES
from exporters import ConfigExporter
from hardware_detector import HardwareDetector
from model_discovery import extract_model_specs, get_local_models, get_model_path
from vram_calculator import VRAMCalculator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

DEFAULT_LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE_URL", "http://").rstrip("/")
DEFAULT_BRIDGE_HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
DEFAULT_BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8080"))
DEFAULT_BRIDGE_BASE = f"http://{DEFAULT_BRIDGE_HOST}:{DEFAULT_BRIDGE_PORT}"
DEFAULT_RETRIEVAL = {
    "chunk_size": 900,
    "chunk_overlap": 150,
    "top_k": 4,
    "max_chunks": 64,
    "max_chunk_chars": 1600,
    "max_context_chars": 6000,
    "include_sources": True,
}

GUIDE_TEXT = """LM Studio Agent Console

Use this app to manage local models, preview retrieval context, run responses or chat
requests, validate profiles with benchmarks, and export LM Studio presets.
"""


class LMStudioAgentConsole(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LM Studio Agent Console V6")
        self.geometry("1540x980")
        self.minsize(1380, 860)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.detector = HardwareDetector()
        self.hardware = None
        self.engine = None
        self.models = []
        self.spec_cache = {}
        self.recommendations = []
        self.selected_rec = None
        self.validated_probe = None
        self.validated_tuned_profile = None
        self.proxy_process = None
        self.runtime_log_lines = []
        self.retrieval_sources = []
        self.selected_source_index = None
        self.current_profile = {}
        self.active_backend_key = "cuda"
        self.task_in_flight = False
        self.available_presets = []
        self.state_dir = Path(".gui_state")
        self.request_counts = {}

        self.model_roles = {
            "main": self._clean_model_id(os.getenv("MAIN_MODEL", "qwen3.5-4b")),
            "reasoning": self._clean_model_id(os.getenv("REASONING_MODEL", "qwen3.5-4b")),
            "embed": self._clean_model_id(os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-4b")),
            "rerank": self._clean_model_id(os.getenv("RERANK_MODEL", "qwen.qwen3-reranker-4b")),
        }
        self.role_vars = {}

        self.bridge_base_var = tk.StringVar(value=DEFAULT_BRIDGE_BASE)
        self.lmstudio_base_var = tk.StringVar(value=DEFAULT_LMSTUDIO_BASE)
        self.status_var = tk.StringVar(value="Starting up...")
        self.runtime_status_var = tk.StringVar(value="Bridge stopped")
        self.loaded_models_var = tk.StringVar(value="No loaded models yet")
        self.profile_status_var = tk.StringVar(value="No profile generated yet")
        self.busy_var = tk.StringVar(value="Idle")
        self.reasoning_effort_var = tk.StringVar(value="medium")
        self.chat_stream_var = tk.BooleanVar(value=False)
        self.include_sources_var = tk.BooleanVar(value=True)
        self.force_flash_var = tk.BooleanVar(value=True)
        self.selected_preset_var = tk.StringVar(value="No presets found")
        self.show_all_models_var = tk.BooleanVar(value=False)
        self.probe_mode_var = tk.StringVar(value="quick")

        self._build_layout()
        self.after(250, self._initial_load)

    def _build_layout(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="LM Studio Agent Console",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(18, 4), sticky="w")
        ctk.CTkLabel(
            self.sidebar,
            text="Responses-first local operator console",
            text_color="#8b949e",
        ).grid(row=1, column=0, padx=20, pady=(0, 14), sticky="w")

        self.hw_box = ctk.CTkTextbox(self.sidebar, height=160, font=("Consolas", 12))
        self.hw_box.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.hw_box.configure(state="disabled")

        control_frame = ctk.CTkFrame(self.sidebar)
        control_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(control_frame, text="Primary Model", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 2), sticky="w"
        )
        self.model_menu = ctk.CTkOptionMenu(control_frame, values=["Scanning..."], command=self._on_model_select)
        self.model_menu.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(control_frame, text="Use Case", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=10, pady=(2, 2), sticky="w"
        )
        self.usecase_menu = ctk.CTkOptionMenu(
            control_frame,
            values=list(USE_CASE_PROFILES.values()),
            command=lambda _v: self._recompute_profile(),
        )
        self.usecase_menu.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="ew")
        self.usecase_menu.set(USE_CASE_PROFILES["balanced"])

        ctk.CTkLabel(control_frame, text="Backend", font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, padx=10, pady=(2, 2), sticky="w"
        )
        self.backend_menu = ctk.CTkOptionMenu(
            control_frame,
            values=["CUDA", "Vulkan", "CLBlast", "Metal", "CPU", "OpenCL"],
            command=self._on_backend_change,
        )
        self.backend_menu.grid(row=5, column=0, padx=10, pady=(0, 8), sticky="ew")
        self.backend_menu.set("CUDA")

        self.validate_btn = ctk.CTkButton(control_frame, text="Validate / Probe Model", command=self._probe_selected_model)
        self.validate_btn.grid(row=6, column=0, padx=10, pady=(6, 6), sticky="ew")
        self.refresh_hw_btn = ctk.CTkButton(
            control_frame,
            text="Refresh Hardware + Profile",
            command=self._refresh_hardware_and_profile,
            fg_color="transparent",
            border_width=1,
        )
        self.refresh_hw_btn.grid(row=7, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.recommendation_frame = ctk.CTkFrame(self.sidebar)
        self.recommendation_frame.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.recommendation_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.recommendation_frame,
            text="Recommendations",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 6), sticky="w")
        self.recommendation_scroll = ctk.CTkScrollableFrame(self.recommendation_frame, height=260, fg_color="transparent")
        self.recommendation_scroll.grid(row=1, column=0, padx=8, pady=(0, 10), sticky="nsew")
        self.recommendation_scroll.grid_columnconfigure(0, weight=1)

        self.sidebar_status = ctk.CTkLabel(self.sidebar, textvariable=self.profile_status_var, wraplength=300, justify="left")
        self.sidebar_status.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="w")
        ctk.CTkLabel(self.sidebar, textvariable=self.status_var, text_color="#8b949e", wraplength=300, justify="left").grid(
            row=6, column=0, padx=20, pady=(0, 8), sticky="w"
        )

        self.export_preset_btn = ctk.CTkButton(
            self.sidebar,
            text="Export LM Studio Preset",
            state="disabled",
            command=self._export_selected_preset,
        )
        self.export_preset_btn.grid(row=7, column=0, padx=20, pady=(4, 6), sticky="ew")
        self.export_env_btn = ctk.CTkButton(
            self.sidebar,
            text="Copy Bridge Env Profile",
            state="disabled",
            command=self._copy_env_profile,
        )
        self.export_env_btn.grid(row=8, column=0, padx=20, pady=(0, 12), sticky="sew")

        self.main = ctk.CTkFrame(self)
        self.main.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(self.main)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_bar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(top_bar, text="Runtime + Workspaces", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, padx=14, pady=12, sticky="w"
        )
        self.busy_label = ctk.CTkLabel(top_bar, textvariable=self.busy_var, text_color="#f2cc60")
        self.busy_label.grid(row=0, column=1, padx=8, pady=12, sticky="e")
        self.busy_bar = ctk.CTkProgressBar(top_bar, mode="indeterminate", width=140)
        self.busy_bar.grid(row=0, column=2, padx=8, pady=12, sticky="e")
        self.busy_bar.set(0)
        ctk.CTkLabel(top_bar, textvariable=self.runtime_status_var, text_color="#58a6ff").grid(
            row=0, column=3, padx=14, pady=12, sticky="e"
        )

        self.tabs = ctk.CTkTabview(self.main)
        self.tabs.grid(row=1, column=0, sticky="nsew")
        for name in ("Runtime", "Responses", "Chat", "Retrieval", "Benchmark", "Profile", "VRAM", "Guide"):
            self.tabs.add(name)

        self._build_runtime_tab()
        self._build_responses_tab()
        self._build_chat_tab()
        self._build_retrieval_tab()
        self._build_benchmark_tab()
        self._build_profile_tab()
        self._build_vram_tab()
        self._build_guide_tab()

    def _build_runtime_tab(self):
        tab = self.tabs.tab("Runtime")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(2, weight=1)

        conn = ctk.CTkFrame(body)
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        conn.grid_columnconfigure(1, weight=1)
        conn.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(conn, text="Bridge Base", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.bridge_entry = ctk.CTkEntry(conn, textvariable=self.bridge_base_var)
        self.bridge_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(conn, text="LM Studio Base", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.lmstudio_entry = ctk.CTkEntry(conn, textvariable=self.lmstudio_base_var)
        self.lmstudio_entry.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        self.start_bridge_btn = ctk.CTkButton(conn, text="Start Bridge", command=self._toggle_bridge)
        self.start_bridge_btn.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkButton(conn, text="Refresh Health", command=self._refresh_runtime_status).grid(
            row=1, column=1, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(conn, text="Refresh Models", command=self._load_models).grid(
            row=1, column=2, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(conn, text="Refresh Loaded", command=self._refresh_runtime_status).grid(
            row=1, column=3, padx=10, pady=(0, 10), sticky="ew"
        )

        roles = ctk.CTkFrame(body)
        roles.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(0, 12))
        roles.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(roles, text="Runtime Role Map", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=12, pady=(12, 8), sticky="w"
        )
        ctk.CTkCheckBox(
            roles,
            text="Show all models",
            variable=self.show_all_models_var,
            command=self._refresh_role_menus,
        ).grid(row=0, column=2, padx=12, pady=(12, 8), sticky="e")
        role_names = [("Main", "main"), ("Reasoning", "reasoning"), ("Embed", "embed"), ("Rerank", "rerank")]
        for idx, (label, key) in enumerate(role_names, start=1):
            ctk.CTkLabel(roles, text=label).grid(row=idx, column=0, padx=12, pady=8, sticky="w")
            var = tk.StringVar(value=self.model_roles[key])
            self.role_vars[key] = var
            menu = ctk.CTkOptionMenu(roles, variable=var, values=[self.model_roles[key]])
            menu.grid(row=idx, column=1, padx=12, pady=8, sticky="ew")
            setattr(self, f"role_menu_{key}", menu)
            ctk.CTkButton(roles, text="Load", width=70, command=lambda k=key: self._load_role_model(k)).grid(
                row=idx, column=2, padx=(0, 12), pady=8
            )
        ctk.CTkButton(roles, text="Save Role Mapping", command=self._save_role_mapping).grid(
            row=len(role_names) + 1, column=0, columnspan=3, padx=12, pady=(8, 12), sticky="ew"
        )

        health = ctk.CTkFrame(body)
        health.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(0, 12))
        health.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(health, text="Server Health", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 8), sticky="w"
        )
        ctk.CTkLabel(health, text="Loaded Models", font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, padx=12, pady=(4, 2), sticky="w"
        )
        self.loaded_models_label = ctk.CTkLabel(health, textvariable=self.loaded_models_var, justify="left", wraplength=420)
        self.loaded_models_label.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="w")
        self.runtime_health_box = ctk.CTkTextbox(health, height=180, font=("Consolas", 11))
        self.runtime_health_box.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="nsew")
        self.runtime_health_box.configure(state="disabled")
        ctk.CTkLabel(health, text="Local Model Inventory", font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, padx=12, pady=(0, 2), sticky="w"
        )
        self.model_inventory_box = ctk.CTkTextbox(health, height=200, font=("Consolas", 11))
        self.model_inventory_box.grid(row=5, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.model_inventory_box.configure(state="disabled")

        log_frame = ctk.CTkFrame(body)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_frame, text="Runtime Log", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 8), sticky="w"
        )
        self.runtime_log_box = ctk.CTkTextbox(log_frame, font=("Consolas", 11))
        self.runtime_log_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.runtime_log_box.configure(state="disabled")

    def _build_responses_tab(self):
        tab = self.tabs.tab("Responses")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(2, weight=1)

        form = ctk.CTkFrame(body)
        form.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        form.grid_columnconfigure(1, weight=1)
        form.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(form, text="Instructions").grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        self.responses_instructions_box = ctk.CTkTextbox(form, height=90)
        self.responses_instructions_box.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkLabel(form, text="Input").grid(row=0, column=2, padx=10, pady=(10, 4), sticky="w")
        self.responses_input_box = ctk.CTkTextbox(form, height=90)
        self.responses_input_box.grid(row=1, column=2, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(form, text="Reasoning Effort").grid(row=2, column=0, padx=10, pady=(0, 4), sticky="w")
        self.responses_reasoning_menu = ctk.CTkOptionMenu(
            form, values=["low", "medium", "high"], variable=self.reasoning_effort_var
        )
        self.responses_reasoning_menu.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkButton(form, text="Build Request Preview", command=self._update_responses_preview).grid(
            row=3, column=1, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(form, text="Run Responses Request", command=self._run_responses_request).grid(
            row=3, column=2, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(form, text="Clear Response", command=self._clear_responses_output).grid(
            row=3, column=3, padx=10, pady=(0, 10), sticky="ew"
        )

        self.responses_preview_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.responses_preview_box.grid(row=1, column=0, padx=(12, 6), pady=(0, 12), sticky="nsew")
        self.responses_output_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.responses_output_box.grid(row=1, column=1, padx=(6, 12), pady=(0, 12), sticky="nsew")

        lower = ctk.CTkFrame(body)
        lower.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        lower.grid_columnconfigure(0, weight=1)
        lower.grid_columnconfigure(1, weight=1)
        lower.grid_columnconfigure(2, weight=1)
        lower.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(lower, text="Assistant Text", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(lower, text="Reasoning / Tools", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=1, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(lower, text="Raw Metadata", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=2, padx=10, pady=(10, 4), sticky="w"
        )
        self.responses_text_box = ctk.CTkTextbox(lower)
        self.responses_text_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.responses_reasoning_box = ctk.CTkTextbox(lower)
        self.responses_reasoning_box.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")
        self.responses_meta_box = ctk.CTkTextbox(lower, font=("Consolas", 11))
        self.responses_meta_box.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="nsew")

    def _build_chat_tab(self):
        tab = self.tabs.tab("Chat")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(2, weight=1)

        form = ctk.CTkFrame(body)
        form.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="System Message").grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        self.chat_system_box = ctk.CTkTextbox(form, height=90)
        self.chat_system_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkLabel(form, text="User Message").grid(row=0, column=1, padx=10, pady=(10, 4), sticky="w")
        self.chat_user_box = ctk.CTkTextbox(form, height=90)
        self.chat_user_box.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkCheckBox(form, text="Stream flag in payload", variable=self.chat_stream_var).grid(
            row=2, column=0, padx=10, pady=(0, 10), sticky="w"
        )
        ctk.CTkButton(form, text="Build Chat Preview", command=self._update_chat_preview).grid(
            row=2, column=1, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(form, text="Run Chat Request", command=self._run_chat_request).grid(
            row=2, column=2, padx=10, pady=(0, 10), sticky="ew"
        )

        self.chat_preview_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.chat_preview_box.grid(row=1, column=0, padx=(12, 6), pady=(0, 12), sticky="nsew")
        self.chat_output_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.chat_output_box.grid(row=1, column=1, padx=(6, 12), pady=(0, 12), sticky="nsew")

        lower = ctk.CTkFrame(body)
        lower.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        lower.grid_columnconfigure(0, weight=1)
        lower.grid_columnconfigure(1, weight=1)
        lower.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(lower, text="Response Text", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(lower, text="Raw Result", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=1, padx=10, pady=(10, 4), sticky="w"
        )
        self.chat_text_box = ctk.CTkTextbox(lower)
        self.chat_text_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.chat_raw_box = ctk.CTkTextbox(lower, font=("Consolas", 11))
        self.chat_raw_box.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")

    def _build_retrieval_tab(self):
        tab = self.tabs.tab("Retrieval")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        body.grid_rowconfigure(2, weight=1)

        controls = ctk.CTkFrame(body)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        for idx in range(6):
            controls.grid_columnconfigure(idx, weight=1 if idx else 0)

        ctk.CTkButton(controls, text="Add File", command=self._add_file_source).grid(row=0, column=0, padx=8, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Add Folder", command=self._add_folder_source).grid(row=0, column=1, padx=8, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Add URL", command=self._add_url_source).grid(row=0, column=2, padx=8, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Remove Selected", command=self._remove_selected_source).grid(row=0, column=3, padx=8, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Preview Retrieval", command=self._preview_retrieval).grid(row=0, column=4, padx=8, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Sync Profile Defaults", command=self._sync_profile_to_retrieval).grid(row=0, column=5, padx=8, pady=10, sticky="ew")

        left = ctk.CTkFrame(body)
        left.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(12, 6), pady=(0, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        left.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(left, text="Workspace Sources", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 8), sticky="w"
        )
        self.sources_scroll = ctk.CTkScrollableFrame(left, fg_color="transparent", height=260)
        self.sources_scroll.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="nsew")
        self.sources_scroll.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(left, text="Source Preview", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=12, pady=(0, 4), sticky="w"
        )
        self.source_preview_box = ctk.CTkTextbox(left, font=("Consolas", 11))
        self.source_preview_box.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="nsew")

        right_top = ctk.CTkFrame(body)
        right_top.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(0, 6))
        for idx in range(4):
            right_top.grid_columnconfigure(idx, weight=1)
        ctk.CTkLabel(right_top, text="Retrieval Settings", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, columnspan=4, padx=12, pady=(12, 8), sticky="w"
        )
        self.retrieval_var_map = {
            "chunk_size": tk.StringVar(value=str(DEFAULT_RETRIEVAL["chunk_size"])),
            "chunk_overlap": tk.StringVar(value=str(DEFAULT_RETRIEVAL["chunk_overlap"])),
            "top_k": tk.StringVar(value=str(DEFAULT_RETRIEVAL["top_k"])),
            "max_chunks": tk.StringVar(value=str(DEFAULT_RETRIEVAL["max_chunks"])),
            "max_chunk_chars": tk.StringVar(value=str(DEFAULT_RETRIEVAL["max_chunk_chars"])),
            "max_context_chars": tk.StringVar(value=str(DEFAULT_RETRIEVAL["max_context_chars"])),
        }
        row_defs = [
            ("Chunk Size", "chunk_size"),
            ("Overlap", "chunk_overlap"),
            ("Top-K", "top_k"),
            ("Max Chunks", "max_chunks"),
            ("Max Chunk Chars", "max_chunk_chars"),
            ("Max Context Chars", "max_context_chars"),
        ]
        for idx, (label, key) in enumerate(row_defs, start=1):
            col = 0 if idx <= 3 else 2
            row = idx if idx <= 3 else idx - 3
            ctk.CTkLabel(right_top, text=label).grid(row=row, column=col, padx=10, pady=6, sticky="w")
            ctk.CTkEntry(right_top, textvariable=self.retrieval_var_map[key]).grid(row=row, column=col + 1, padx=10, pady=6, sticky="ew")
        ctk.CTkCheckBox(right_top, text="Include Sources", variable=self.include_sources_var).grid(
            row=4, column=0, padx=10, pady=(4, 10), sticky="w"
        )
        ctk.CTkLabel(right_top, text="Preview Query").grid(row=4, column=2, padx=10, pady=(4, 4), sticky="w")
        self.retrieval_query_entry = ctk.CTkEntry(right_top)
        self.retrieval_query_entry.grid(row=4, column=3, padx=10, pady=(4, 10), sticky="ew")

        right_bottom = ctk.CTkFrame(body)
        right_bottom.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=(6, 12))
        right_bottom.grid_columnconfigure(0, weight=1)
        right_bottom.grid_columnconfigure(1, weight=1)
        right_bottom.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(right_bottom, text="Chunk / Rerank Preview", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(right_bottom, text="Injected Context Preview", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=1, padx=10, pady=(10, 4), sticky="w"
        )
        self.retrieval_chunks_box = ctk.CTkTextbox(right_bottom, font=("Consolas", 11))
        self.retrieval_chunks_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.retrieval_context_box = ctk.CTkTextbox(right_bottom, font=("Consolas", 11))
        self.retrieval_context_box.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="nsew")

    def _build_benchmark_tab(self):
        tab = self.tabs.tab("Benchmark")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)
        body.grid_rowconfigure(2, weight=1)

        actions = ctk.CTkFrame(body)
        actions.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        for idx in range(7):
            actions.grid_columnconfigure(idx, weight=1)
        ctk.CTkButton(actions, text="Connectivity Check", command=self._run_connectivity_check).grid(
            row=0, column=0, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="List Models", command=self._run_model_list_check).grid(
            row=0, column=1, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Probe Capabilities", command=self._probe_selected_model).grid(
            row=0, column=2, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Full Streaming Compare", command=self._compare_streaming_modes).grid(
            row=0, column=3, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Apply Tuned Settings", command=self._apply_tuned_profile).grid(
            row=0, column=4, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Chat Smoke Test", command=self._run_chat_smoke_test).grid(
            row=0, column=5, padx=8, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Agent Smoke Test", command=self._run_agent_smoke_test).grid(
            row=0, column=6, padx=8, pady=10, sticky="ew"
        )

        self.benchmark_summary_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.benchmark_summary_box.grid(row=1, column=0, padx=(12, 6), pady=(0, 12), sticky="nsew")
        self.benchmark_raw_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.benchmark_raw_box.grid(row=1, column=1, padx=(6, 12), pady=(0, 12), sticky="nsew")
        self.benchmark_warnings_box = ctk.CTkTextbox(body, font=("Consolas", 11))
        self.benchmark_warnings_box.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="nsew")

    def _build_profile_tab(self):
        tab = self.tabs.tab("Profile")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(body)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        for idx in range(6):
            top.grid_columnconfigure(idx, weight=1)
        self.profile_header_label = ctk.CTkLabel(top, text="No active profile", font=ctk.CTkFont(size=15, weight="bold"))
        self.profile_header_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        ctk.CTkCheckBox(top, text="Force Flash Attention", variable=self.force_flash_var, command=self._recompute_profile).grid(
            row=0, column=2, padx=10, pady=10, sticky="w"
        )
        self.preset_menu = ctk.CTkOptionMenu(top, variable=self.selected_preset_var, values=["No presets found"])
        self.preset_menu.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(top, text="Refresh Presets", command=self._load_available_presets).grid(
            row=0, column=4, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(top, text="Apply Preset", command=self._apply_selected_preset).grid(
            row=0, column=5, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(top, text="Reset To Predicted", command=self._reset_profile_to_predicted).grid(
            row=1, column=5, padx=10, pady=(0, 10), sticky="ew"
        )

        left = ctk.CTkFrame(body)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(0, 12))
        left.grid_columnconfigure(1, weight=1)
        self.profile_vars = {
            "temperature": tk.StringVar(),
            "top_p": tk.StringVar(),
            "top_k": tk.StringVar(),
            "repeat_penalty": tk.StringVar(),
            "max_tokens": tk.StringVar(),
            "context_length": tk.StringVar(),
            "mode": tk.StringVar(),
            "chunk_size": tk.StringVar(),
            "chunk_overlap": tk.StringVar(),
            "retrieval_top_k": tk.StringVar(),
            "max_context_chars": tk.StringVar(),
            "embed_model": tk.StringVar(),
            "rerank_model": tk.StringVar(),
        }
        labels = [
            ("Temperature", "temperature"),
            ("Top P", "top_p"),
            ("Top K", "top_k"),
            ("Repeat Penalty", "repeat_penalty"),
            ("Max Output Tokens", "max_tokens"),
            ("Context Target", "context_length"),
            ("Mode", "mode"),
            ("Chunk Size", "chunk_size"),
            ("Chunk Overlap", "chunk_overlap"),
            ("Retrieval Top-K", "retrieval_top_k"),
            ("Max Context Chars", "max_context_chars"),
            ("Embed Role", "embed_model"),
            ("Rerank Role", "rerank_model"),
        ]
        for row, (label, key) in enumerate(labels):
            ctk.CTkLabel(left, text=label).grid(row=row, column=0, padx=10, pady=6, sticky="w")
            ctk.CTkEntry(left, textvariable=self.profile_vars[key]).grid(row=row, column=1, padx=10, pady=6, sticky="ew")

        right = ctk.CTkFrame(body)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(0, 12))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(right, text="System Prompt", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        self.system_prompt_box = ctk.CTkTextbox(right, height=220)
        self.system_prompt_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(right, text="Bridge Env Preview", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=10, pady=(0, 4), sticky="w"
        )
        self.env_preview_box = ctk.CTkTextbox(right, font=("Consolas", 11))
        self.env_preview_box.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def _build_vram_tab(self):
        tab = self.tabs.tab("VRAM")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.vram_box = ctk.CTkTextbox(tab, font=("Consolas", 12))
        self.vram_box.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

    def _build_guide_tab(self):
        tab = self.tabs.tab("Guide")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        guide = ctk.CTkTextbox(tab, font=("Consolas", 12))
        guide.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        guide.insert("1.0", GUIDE_TEXT)
        guide.configure(state="disabled")

    def _initial_load(self):
        self._detect_hardware()
        self._load_models()
        self._load_available_presets()
        self._refresh_runtime_status()
        self.responses_instructions_box.insert("1.0", "Act like a careful local coding agent. Use provided context when relevant.")
        self.chat_system_box.insert("1.0", "You are a helpful local assistant.")

    def _run_background(self, label, worker, on_success=None, on_error=None):
        if self.task_in_flight:
            self.status_var.set(f"Busy with another task: {self.busy_var.get()}")
            return
        self.task_in_flight = True
        self.busy_var.set(label)
        self.busy_bar.start()
        self.status_var.set(label)
        self._append_runtime_log(f"[action:start] {label}")

        def task():
            try:
                result = worker()
                if on_success:
                    self.after(0, lambda result=result: self._finish_background(label, on_success, result))
                else:
                    self.after(0, lambda: self._finish_background(label, None, None))
            except Exception as exc:
                if on_error:
                    self.after(0, lambda exc=exc: self._fail_background(label, exc, on_error))
                else:
                    self.after(0, lambda exc=exc: self._fail_background(label, exc, None))

        threading.Thread(target=task, daemon=True).start()

    def _finish_background(self, label, on_success, result):
        self.task_in_flight = False
        self.busy_bar.stop()
        self.busy_var.set("Idle")
        self._append_runtime_log(f"[action:done] {label}")
        if on_success:
            on_success(result)
        else:
            self.status_var.set(f"{label} finished")

    def _fail_background(self, label, exc, on_error):
        self.task_in_flight = False
        self.busy_bar.stop()
        self.busy_var.set("Idle")
        self._append_runtime_log(f"[action:failed] {label}: {exc}")
        if on_error:
            on_error(exc)
        else:
            self._show_error(f"{label} failed", exc)

    def _append_runtime_log(self, line):
        if not line:
            return
        self.runtime_log_lines.append(line.rstrip())
        self.runtime_log_lines = self.runtime_log_lines[-400:]
        self.runtime_log_box.configure(state="normal")
        self.runtime_log_box.delete("1.0", tk.END)
        self.runtime_log_box.insert("1.0", "\n".join(self.runtime_log_lines))
        self.runtime_log_box.configure(state="disabled")
        self.runtime_log_box.see(tk.END)

    def _monitor_proxy_output(self, proc):
        for line in iter(proc.stdout.readline, ""):
            self.after(0, lambda value=line: self._append_runtime_log(value))
        rc = proc.poll()
        if rc is not None:
            self.after(0, lambda: self._set_bridge_stopped(f"Bridge exited with code {rc}"))

    def _detect_hardware(self):
        self.hardware = self.detector.detect()
        self.engine = RecommendationEngine(self.hardware)
        if self.hardware.platform == "macos" and self.hardware.is_apple_silicon:
            self.active_backend_key = "metal"
        elif self.hardware.gpu_name and ("nvidia" in self.hardware.gpu_name.lower() or self.hardware.cuda_version):
            self.active_backend_key = "cuda"
        elif self.hardware.gpu_name:
            self.active_backend_key = "vulkan"
        else:
            self.active_backend_key = "cpu"
        self.backend_menu.set(self.active_backend_key.capitalize() if self.active_backend_key != "opencl" else "OpenCL")
        lines = [
            f"Platform : {self.hardware.platform}",
            f"CPU      : {self.hardware.cpu_cores} physical / {self.hardware.logical_cores} logical",
            f"RAM      : {self.hardware.system_ram_gb:.1f} GB",
            f"GPU      : {self.hardware.gpu_name or 'Not detected'}",
            f"VRAM     : {self.hardware.gpu_vram_gb or 0:.1f} GB",
            f"CUDA     : {self.hardware.cuda_version or 'n/a'}",
        ]
        self.hw_box.configure(state="normal")
        self.hw_box.delete("1.0", tk.END)
        self.hw_box.insert("1.0", "\n".join(lines))
        self.hw_box.configure(state="disabled")
        self.status_var.set("Hardware detected")

    def _refresh_hardware(self):
        self._detect_hardware()
        if self.models:
            self._recompute_profile()

    def _refresh_hardware_and_profile(self):
        self._detect_hardware()
        self._load_models()
        self._load_available_presets()
        self._refresh_runtime_status()

    def _load_models(self):
        self._bump_request_count("models_refresh")
        self.models = get_local_models()
        if not self.models:
            empty = ["No local models found - open LM Studio or install with lms"]
            self.model_menu.configure(values=empty)
            self.model_menu.set(empty[0])
            for key in self.role_vars:
                getattr(self, f"role_menu_{key}").configure(values=empty)
            self._update_model_inventory_box()
            self.status_var.set("No local models discovered")
            self._persist_state_snapshot("models_empty")
            return
        llm_names = self._model_choices("llm")
        self.model_menu.configure(values=llm_names)
        current = self.model_menu.get()
        if current not in llm_names:
            self.model_menu.set(llm_names[0])
        self._refresh_role_menus()
        self._update_model_inventory_box()
        self.status_var.set(
            "Loaded "
            f"{len(self._models_for_type('llm'))} llm / "
            f"{len(self._models_for_type('embedding'))} embedding / "
            f"{len(self._models_for_type('rerank'))} rerank models"
        )
        self._persist_state_snapshot("models_refreshed")
        self._recompute_profile()

    def _models_for_type(self, model_type):
        if model_type == "llm":
            models = [model for model in self.models if model.get("type", "llm") == "llm"]
            return models or list(self.models)
        models = [model for model in self.models if model.get("type") == model_type]
        return models or list(self.models)

    def _model_choices(self, model_type):
        if self.show_all_models_var.get():
            return [model["id"] for model in self.models] or ["No local models found - open LM Studio or install with lms"]
        models = self._models_for_type(model_type)
        return [model["id"] for model in models] or ["No local models found - open LM Studio or install with lms"]

    def _refresh_role_menus(self):
        if not self.role_vars:
            return
        for key, var in self.role_vars.items():
            target_type = "llm" if key in ("main", "reasoning") else ("embedding" if key == "embed" else "rerank")
            values = self._model_choices(target_type)
            getattr(self, f"role_menu_{key}").configure(values=values)
            if var.get() not in values:
                var.set(self._preferred_role_model(key))
        self._persist_state_snapshot("role_menu_refreshed")

    def _preferred_role_model(self, role_key):
        target_type = "llm" if role_key in ("main", "reasoning") else ("embedding" if role_key == "embed" else "rerank")
        choices = self._model_choices(target_type)
        if not choices or choices[0].startswith("No local models"):
            return self.model_roles.get(role_key, "")
        low_resource = bool(
            self.hardware
            and ((self.hardware.gpu_vram_gb or 0) <= 8.5 or (self.hardware.system_ram_gb or 0) <= 16.5)
        )
        if role_key == "embed":
            if low_resource:
                for candidate in choices:
                    if "nomic" in candidate.lower():
                        return candidate
            for candidate in choices:
                if "qwen3-embedding" in candidate.lower():
                    return candidate
        if role_key == "rerank":
            if low_resource:
                for candidate in choices:
                    if "0.6b" in candidate.lower():
                        return candidate
            for candidate in choices:
                if "4b" in candidate.lower():
                    return candidate
        return choices[0]

    def _update_model_inventory_box(self):
        if not hasattr(self, "model_inventory_box"):
            return
        lines = []
        if not self.models:
            lines.append("No local models discovered yet.")
        else:
            for title, model_type in (("LLM", "llm"), ("EMBED", "embedding"), ("RERANK", "rerank")):
                subset = [item for item in self.models if item.get("type", "llm") == model_type]
                if not subset:
                    continue
                lines.append(f"[{title}]")
                for model in subset:
                    lines.append(f"- {model['id']}")
                    lines.append(
                        "  "
                        f"arch={model.get('arch') or 'unknown-arch'} "
                        f"params={model.get('params') or '?'} "
                        f"size={model.get('size') or '?'} "
                        f"state={'loaded' if model.get('loaded') else 'idle'}"
                    )
                lines.append("")
        self.model_inventory_box.configure(state="normal")
        self.model_inventory_box.delete("1.0", tk.END)
        self.model_inventory_box.insert("1.0", "\n".join(lines).strip() or "No local models discovered yet.")
        self.model_inventory_box.configure(state="disabled")

    def _preset_dirs(self):
        home = Path.home()
        return [
            home / ".cache" / "lm-studio" / "config-presets",
            home / ".cache" / "lm-studio" / "presets",
        ]

    def _load_available_presets(self):
        presets = []
        for directory in self._preset_dirs():
            if not directory.exists():
                continue
            for path in directory.glob("*.json"):
                presets.append(path)
        presets.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        self.available_presets = presets
        labels = [f"{path.parent.name}/{path.name}" for path in presets] or ["No presets found"]
        self.preset_menu.configure(values=labels)
        self.selected_preset_var.set(labels[0])
        self.status_var.set(f"Loaded {len(presets)} presets")

    def _selected_preset_path(self):
        label = self.selected_preset_var.get()
        for path in self.available_presets:
            if label == f"{path.parent.name}/{path.name}":
                return path
        return None

    def _get_selected_model_id(self):
        value = self.model_menu.get()
        if value.startswith("No local models"):
            return None
        return value

    def _get_model_specs(self, model_id):
        if model_id not in self.spec_cache:
            path = get_model_path(model_id)
            self.spec_cache[model_id] = extract_model_specs(path) or {}
        return self.spec_cache[model_id]

    def _parse_params_b(self, model_id):
        model_info = next((item for item in self.models if item["id"] == model_id), {})
        raw = model_info.get("params", "7")
        digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
        return float(digits) if digits else 7.0

    def _on_model_select(self, _model_id):
        self._recompute_profile()

    def _on_backend_change(self, label):
        self.active_backend_key = label.lower()
        self._recompute_profile()

    def _recompute_profile(self):
        model_id = self._get_selected_model_id()
        if not model_id or not self.engine:
            return
        usecase_label = self.usecase_menu.get()
        usecase_key = next((key for key, value in USE_CASE_PROFILES.items() if value == usecase_label), "balanced")
        specs = self._get_model_specs(model_id)
        recommendations = self.engine.recommend(
            model_id=model_id,
            params_b=self._parse_params_b(model_id),
            num_layers=specs.get("num_layers", 32),
            hidden_size=specs.get("hidden_size", 4096),
            num_heads=specs.get("num_heads", 32),
            kv_heads=specs.get("kv_heads"),
            use_case=usecase_key,
            backend=self.active_backend_key,
            flash_attention=self.force_flash_var.get(),
        )
        fallback_backend = None
        if not recommendations and self.active_backend_key != "cpu":
            fallback_backend = "cpu"
            recommendations = self.engine.recommend(
                model_id=model_id,
                params_b=self._parse_params_b(model_id),
                num_layers=specs.get("num_layers", 32),
                hidden_size=specs.get("hidden_size", 4096),
                num_heads=specs.get("num_heads", 32),
                kv_heads=specs.get("kv_heads"),
                use_case=usecase_key,
                backend="cpu",
                flash_attention=False,
                min_quality=0.0,
            )
        if not recommendations:
            recommendations = self.engine.recommend(
                model_id=model_id,
                params_b=self._parse_params_b(model_id),
                num_layers=specs.get("num_layers", 32),
                hidden_size=specs.get("hidden_size", 4096),
                num_heads=specs.get("num_heads", 32),
                kv_heads=specs.get("kv_heads"),
                use_case=usecase_key,
                backend=self.active_backend_key,
                flash_attention=self.force_flash_var.get(),
                min_quality=0.0,
            )
        self.recommendations = recommendations
        self.selected_rec = recommendations[0] if recommendations else None
        self.validated_probe = None
        self.validated_tuned_profile = None
        self._render_recommendations()
        self._reset_profile_to_predicted()
        self._render_vram_report()
        if fallback_backend and recommendations:
            self.status_var.set(f"Predicted profile updated with CPU fallback for {model_id}")
        elif recommendations:
            self.status_var.set("Predicted profile updated")
        else:
            self.status_var.set("No strong fit found; adjust backend or load a smaller model")

    def _predicted_profile_from_rec(self, rec):
        mode = "think" if rec.enable_thinking else ("architect" if rec.context_length >= 16384 else "fast")
        retrieval_defaults = dict(DEFAULT_RETRIEVAL)
        low_resource = bool(
            self.hardware
            and ((self.hardware.gpu_vram_gb or 0) <= 8.5 or (self.hardware.system_ram_gb or 0) <= 16.5)
        )
        if rec.context_length <= 4096:
            retrieval_defaults["max_context_chars"] = 3200
            retrieval_defaults["top_k"] = 3
        elif rec.context_length >= 32768:
            retrieval_defaults["max_context_chars"] = 9000
            retrieval_defaults["top_k"] = 5
        if low_resource:
            mode = "fast" if mode == "architect" else mode
            retrieval_defaults["chunk_size"] = min(retrieval_defaults["chunk_size"], 700)
            retrieval_defaults["chunk_overlap"] = min(retrieval_defaults["chunk_overlap"], 100)
            retrieval_defaults["max_context_chars"] = min(retrieval_defaults["max_context_chars"], 2400)
            retrieval_defaults["top_k"] = min(retrieval_defaults["top_k"], 3)
        return {
            "status": "predicted",
            "model_id": rec.model_id,
            "system_prompt": rec.system_prompt or "You are a helpful local AI assistant.",
            "temperature": rec.temperature,
            "top_p": rec.top_p,
            "top_k": rec.top_k,
            "repeat_penalty": rec.repeat_penalty,
            "max_tokens": min(rec.max_tokens, 1024) if low_resource else rec.max_tokens,
            "context_length": min(rec.context_length, 4096) if low_resource else rec.context_length,
            "mode": mode,
            "chunk_size": retrieval_defaults["chunk_size"],
            "chunk_overlap": retrieval_defaults["chunk_overlap"],
            "retrieval_top_k": retrieval_defaults["top_k"],
            "max_context_chars": retrieval_defaults["max_context_chars"],
            "embed_model": self._preferred_role_model("embed"),
            "rerank_model": self._preferred_role_model("rerank"),
            "thinking_recommended": rec.enable_thinking,
            "quality_score": rec.quality_score,
        }

    def _reset_profile_to_predicted(self):
        if not self.selected_rec:
            self.profile_status_var.set("No valid recommendation for the selected model")
            self.export_preset_btn.configure(state="disabled")
            self.export_env_btn.configure(state="disabled")
            return
        profile = self._predicted_profile_from_rec(self.selected_rec)
        self._apply_profile_to_widgets(profile, status_text="Predicted profile applied")
        self.export_preset_btn.configure(state="normal")
        self.export_env_btn.configure(state="normal")

    def _apply_profile_to_widgets(self, profile, status_text=None):
        self.current_profile = dict(profile)
        self.profile_vars["temperature"].set(str(profile["temperature"]))
        self.profile_vars["top_p"].set(str(profile["top_p"]))
        self.profile_vars["top_k"].set(str(profile["top_k"]))
        self.profile_vars["repeat_penalty"].set(str(profile["repeat_penalty"]))
        self.profile_vars["max_tokens"].set(str(profile["max_tokens"]))
        self.profile_vars["context_length"].set(str(profile["context_length"]))
        self.profile_vars["mode"].set(profile["mode"])
        self.profile_vars["chunk_size"].set(str(profile["chunk_size"]))
        self.profile_vars["chunk_overlap"].set(str(profile["chunk_overlap"]))
        self.profile_vars["retrieval_top_k"].set(str(profile["retrieval_top_k"]))
        self.profile_vars["max_context_chars"].set(str(profile["max_context_chars"]))
        self.profile_vars["embed_model"].set(profile["embed_model"])
        self.profile_vars["rerank_model"].set(profile["rerank_model"])
        self.system_prompt_box.delete("1.0", tk.END)
        self.system_prompt_box.insert("1.0", profile["system_prompt"])
        self.profile_header_label.configure(
            text=f"{profile['model_id']} | {profile['status'].capitalize()} | mode={profile['mode']}"
        )
        self.profile_status_var.set(status_text or f"Profile status: {profile['status']}")
        self._sync_profile_to_retrieval()
        self._update_env_preview()
        self._update_responses_preview()
        self._update_chat_preview()
        self._persist_state_snapshot("profile_applied")

    def _collect_profile_from_widgets(self):
        profile = {
            "status": self.current_profile.get("status", "predicted"),
            "model_id": self._get_selected_model_id(),
            "system_prompt": self.system_prompt_box.get("1.0", "end").strip(),
            "temperature": self._safe_float(self.profile_vars["temperature"].get(), self.current_profile.get("temperature", 0.3)),
            "top_p": self._safe_float(self.profile_vars["top_p"].get(), self.current_profile.get("top_p", 0.95)),
            "top_k": self._safe_int(self.profile_vars["top_k"].get(), self.current_profile.get("top_k", 40)),
            "repeat_penalty": self._safe_float(self.profile_vars["repeat_penalty"].get(), self.current_profile.get("repeat_penalty", 1.1)),
            "max_tokens": self._safe_int(self.profile_vars["max_tokens"].get(), self.current_profile.get("max_tokens", 2048)),
            "context_length": self._safe_int(self.profile_vars["context_length"].get(), self.current_profile.get("context_length", 8192)),
            "mode": self.profile_vars["mode"].get().strip() or self.current_profile.get("mode", "fast"),
            "chunk_size": self._safe_int(self.profile_vars["chunk_size"].get(), DEFAULT_RETRIEVAL["chunk_size"]),
            "chunk_overlap": self._safe_int(self.profile_vars["chunk_overlap"].get(), DEFAULT_RETRIEVAL["chunk_overlap"]),
            "retrieval_top_k": self._safe_int(self.profile_vars["retrieval_top_k"].get(), DEFAULT_RETRIEVAL["top_k"]),
            "max_context_chars": self._safe_int(self.profile_vars["max_context_chars"].get(), DEFAULT_RETRIEVAL["max_context_chars"]),
            "embed_model": self.profile_vars["embed_model"].get().strip() or self.role_vars["embed"].get(),
            "rerank_model": self.profile_vars["rerank_model"].get().strip() or self.role_vars["rerank"].get(),
        }
        self.current_profile = dict(profile)
        self._update_env_preview()
        return profile

    def _render_recommendations(self):
        for child in self.recommendation_scroll.winfo_children():
            child.destroy()
        if not self.recommendations:
            ctk.CTkLabel(self.recommendation_scroll, text="No valid configs found").pack(pady=18)
            return
        for idx, rec in enumerate(self.recommendations):
            top = idx == 0
            frame = ctk.CTkFrame(
                self.recommendation_scroll,
                fg_color="#172235" if top else "transparent",
                border_width=2 if top else 1,
                border_color="#58a6ff" if top else "#30363d",
            )
            frame.pack(fill="x", padx=4, pady=4)
            summary = (
                f"{'#1' if top else f'#{idx + 1}'} | {rec.quantization.value} | "
                f"ctx {rec.context_length:,} | vram {rec.estimated_vram_gb:.1f} GB | "
                f"score {rec.quality_score:.2f}"
            )
            label = ctk.CTkLabel(frame, text=summary, wraplength=280, justify="left")
            label.pack(padx=10, pady=10, anchor="w")
            frame.bind("<Button-1>", lambda _e, choice=rec: self._select_recommendation(choice))
            label.bind("<Button-1>", lambda _e, choice=rec: self._select_recommendation(choice))

    def _select_recommendation(self, rec):
        self.selected_rec = rec
        self.validated_probe = None
        self.validated_tuned_profile = None
        self._reset_profile_to_predicted()
        self._render_vram_report()

    def _render_vram_report(self):
        self.vram_box.delete("1.0", tk.END)
        rec = self.selected_rec
        if not rec:
            self.vram_box.insert("1.0", "No recommendation selected.")
            return
        breakdown = VRAMCalculator.calculate(
            self._parse_params_b(rec.model_id),
            rec.quantization,
            rec.context_length,
            backend=rec.inference_backend,
            cuda_compute=self.hardware.cuda_compute if self.hardware else None,
        )
        lines = [
            f"Model: {rec.model_id}",
            f"Backend: {rec.inference_backend.value}",
            f"Quantization: {rec.quantization.value}",
            f"Context Length: {rec.context_length}",
            "-" * 48,
            f"Weights:      {breakdown['weights_gb']:.2f} GB",
            f"KV Cache:     {breakdown['kv_cache_gb']:.2f} GB",
            f"Activations:  {breakdown['activations_gb']:.2f} GB",
            f"Overhead:     {breakdown['overhead_gb']:.2f} GB",
            f"Total:        {breakdown['total_gb']:.2f} GB",
            "-" * 48,
            f"GPU Layers:   {rec.gpu_layers}",
            f"Threads:      {rec.threads}",
            f"Batch Size:   {rec.batch_size}",
            f"KV Quant:     {rec.kv_cache_quant}",
        ]
        self.vram_box.insert("1.0", "\n".join(lines))

    def _toggle_bridge(self):
        if self.proxy_process is None:
            self._start_bridge()
        else:
            self._stop_bridge()

    def _start_bridge(self):
        env = os.environ.copy()
        env["LMSTUDIO_BASE_URL"] = self.lmstudio_base_var.get().strip()
        env["BRIDGE_HOST"] = self._bridge_host()
        env["BRIDGE_PORT"] = str(self._bridge_port())
        env["MAIN_MODEL"] = self.role_vars["main"].get()
        env["REASONING_MODEL"] = self.role_vars["reasoning"].get()
        env["EMBED_MODEL"] = self.role_vars["embed"].get()
        env["RERANK_MODEL"] = self.role_vars["rerank"].get()
        self._append_runtime_log(
            f"Bridge launch config: host={env['BRIDGE_HOST']} port={env['BRIDGE_PORT']} lmstudio={env['LMSTUDIO_BASE_URL']}"
        )
        try:
            self.proxy_process = subprocess.Popen(
                ["python", "proxy.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.getcwd(),
            )
            self.start_bridge_btn.configure(text="Stop Bridge")
            self.runtime_status_var.set(f"Bridge starting on {self.bridge_base_var.get().strip()}")
            self._append_runtime_log("Starting bridge...")
            self._persist_state_snapshot("bridge_started")
            threading.Thread(target=self._monitor_proxy_output, args=(self.proxy_process,), daemon=True).start()
            self.after(1500, self._refresh_runtime_status)
        except Exception as exc:
            self.proxy_process = None
            self._show_error("Failed to start bridge", exc)

    def _stop_bridge(self):
        if self.proxy_process is not None:
            self.proxy_process.terminate()
            self.proxy_process = None
        self._set_bridge_stopped("Bridge stopped")

    def _set_bridge_stopped(self, message):
        self.start_bridge_btn.configure(text="Start Bridge")
        self.runtime_status_var.set(message)
        self._append_runtime_log(message)
        self._persist_state_snapshot("bridge_stopped")

    def _save_role_mapping(self):
        env_path = Path(".env")
        existing = []
        if env_path.exists():
            existing = env_path.read_text(encoding="utf-8").splitlines()
        keys = {"MAIN_MODEL", "REASONING_MODEL", "EMBED_MODEL", "RERANK_MODEL"}
        kept = [line for line in existing if not any(line.startswith(f"{key}=") for key in keys)]
        kept.extend(
            [
                f"MAIN_MODEL={self._clean_model_id(self.role_vars['main'].get())}",
                f"REASONING_MODEL={self._clean_model_id(self.role_vars['reasoning'].get())}",
                f"EMBED_MODEL={self._clean_model_id(self.role_vars['embed'].get())}",
                f"RERANK_MODEL={self._clean_model_id(self.role_vars['rerank'].get())}",
            ]
        )
        env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        self.status_var.set("Role mapping saved to .env (applies on next bridge start or after manual model load)")
        self._update_env_preview()
        self._persist_state_snapshot("role_mapping_saved")

    def _load_role_model(self, role_key):
        model_id = self._clean_model_id(self.role_vars[role_key].get())
        if not model_id:
            return
        payload = {"model": model_id}
        context_value = self.profile_vars["context_length"].get().strip()
        if context_value:
            payload["context_length"] = self._safe_int(context_value, 8192)
        self._append_runtime_log(f"Loading role `{role_key}` with model `{model_id}`")

        def worker():
            with httpx.Client(timeout=60) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/api/v1/models/load", json=payload)
                response.raise_for_status()
                return response.json()

        def on_error(exc):
            self._append_runtime_log(f"Load failed for `{role_key}` / `{model_id}`: {exc}")
            self._show_error(f"Loading {role_key} model failed", exc)

        self._run_background(
            f"Loading {role_key} model...",
            worker,
            on_success=lambda data: self._on_model_loaded(role_key, model_id, data),
            on_error=on_error,
        )

    def _on_model_loaded(self, role_key, model_id, data):
        self.status_var.set(f"Loaded {role_key} model: {model_id}")
        self._append_runtime_log(json.dumps({"role": role_key, "model": model_id, "result": data}, indent=2))
        self._persist_state_snapshot("role_model_loaded", {"role": role_key, "model": model_id, "result": data})
        self._refresh_runtime_status()

    def _refresh_runtime_status(self):
        self._bump_request_count("runtime_health_refresh")
        bridge_base = self.bridge_base_var.get().strip()
        lmstudio_base = self.lmstudio_base_var.get().strip()

        def worker():
            report = {}
            with httpx.Client(timeout=10) as client:
                try:
                    bridge_models = client.get(f"{bridge_base}/api/v1/models")
                    report["bridge_models"] = bridge_models.json()
                except Exception as exc:
                    report["bridge_error"] = str(exc)
                try:
                    bridge_hw = client.get(f"{bridge_base}/api/v1/hardware")
                    report["bridge_hardware"] = bridge_hw.json()
                except Exception:
                    pass
                try:
                    lm_models = client.get(f"{lmstudio_base}/api/v1/models")
                    report["lmstudio_models"] = lm_models.json()
                except Exception as exc:
                    report["lmstudio_error"] = str(exc)
            return report

        self._run_background("Refreshing runtime status...", worker, on_success=self._apply_runtime_status)

    def _apply_runtime_status(self, report):
        loaded = []
        bridge_payload = report.get("bridge_models")
        for item in self._runtime_model_list(bridge_payload):
            if self._runtime_model_is_loaded(item):
                loaded.append(item.get("id") or item.get("key") or item.get("display_name"))
        lm_payload = report.get("lmstudio_models")
        if not loaded:
            for item in self._runtime_model_list(lm_payload):
                if self._runtime_model_is_loaded(item):
                    loaded.append(item.get("id") or item.get("key") or item.get("display_name"))
        if not loaded:
            loaded = [model.get("id") for model in self.models if model.get("loaded")]
        self.loaded_models_var.set("\n".join(loaded) if loaded else "No loaded models detected")
        bridge_ok = "bridge_models" in report
        lm_ok = "lmstudio_models" in report
        self.runtime_status_var.set(
            f"Bridge {'ok' if bridge_ok else 'down'} | LM Studio {'ok' if lm_ok else 'down'} | loaded {len(loaded)}"
        )
        self.runtime_health_box.configure(state="normal")
        self.runtime_health_box.delete("1.0", tk.END)
        summary = {
            "bridge_ok": bridge_ok,
            "lmstudio_ok": lm_ok,
            "loaded_models": loaded,
            "role_map": {key: var.get() for key, var in self.role_vars.items()},
        }
        self.runtime_health_box.insert(
            "1.0",
            "Runtime Summary\n\n"
            + json.dumps(summary, indent=2)
            + "\n\nRaw Health\n\n"
            + json.dumps(report, indent=2),
        )
        self.runtime_health_box.configure(state="disabled")
        self._persist_state_snapshot("runtime_status", {"report": report, "loaded_models": loaded})
        self.status_var.set("Runtime status refreshed")

    def _runtime_model_list(self, payload):
        if isinstance(payload, dict):
            models = payload.get("data")
            if isinstance(models, list):
                return models
            models = payload.get("models")
            if isinstance(models, list):
                return models
        if isinstance(payload, list):
            return payload
        return []

    def _runtime_model_is_loaded(self, item):
        if item.get("state") == "loaded":
            return True
        loaded_instances = item.get("loaded_instances")
        return isinstance(loaded_instances, list) and len(loaded_instances) > 0

    def _add_file_source(self):
        path = filedialog.askopenfilename()
        if path:
            self._add_source({"kind": "file", "path": path, "label": Path(path).name, "enabled": True})

    def _add_folder_source(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        count = 0
        for ext in ("*.md", "*.txt", "*.py", "*.json", "*.yaml", "*.yml"):
            for path in Path(folder).glob(ext):
                self._add_source({"kind": "file", "path": str(path), "label": path.name, "enabled": True}, rerender=False)
                count += 1
        self._render_sources()
        self.status_var.set(f"Added {count} files from folder")

    def _add_url_source(self):
        url = simpledialog.askstring("Add URL", "Enter a URL to use as context:")
        if url:
            self._add_source({"kind": "url", "path": url.strip(), "label": url.strip(), "enabled": True})

    def _add_source(self, source, rerender=True):
        self.retrieval_sources.append(source)
        self.selected_source_index = len(self.retrieval_sources) - 1
        if rerender:
            self._render_sources()
        self._persist_state_snapshot("retrieval_source_added")

    def _remove_selected_source(self):
        if self.selected_source_index is None:
            return
        if 0 <= self.selected_source_index < len(self.retrieval_sources):
            self.retrieval_sources.pop(self.selected_source_index)
        self.selected_source_index = None
        self._render_sources()
        self.source_preview_box.delete("1.0", tk.END)
        self._persist_state_snapshot("retrieval_source_removed")

    def _render_sources(self):
        for child in self.sources_scroll.winfo_children():
            child.destroy()
        if not self.retrieval_sources:
            ctk.CTkLabel(self.sources_scroll, text="No workspace sources yet").pack(pady=18)
            return
        for idx, source in enumerate(self.retrieval_sources):
            enabled_var = tk.BooleanVar(value=source.get("enabled", True))
            source["enabled_var"] = enabled_var
            row = ctk.CTkFrame(self.sources_scroll)
            row.pack(fill="x", padx=2, pady=4)
            check = ctk.CTkCheckBox(row, text="", variable=enabled_var, command=lambda i=idx: self._toggle_source(i))
            check.pack(side="left", padx=(8, 4), pady=8)
            label = ctk.CTkLabel(row, text=f"{source['kind']}: {source['label']}", wraplength=360, justify="left")
            label.pack(side="left", padx=6, pady=8, anchor="w")
            btn = ctk.CTkButton(row, text="Preview", width=70, command=lambda i=idx: self._preview_source(i))
            btn.pack(side="right", padx=8, pady=8)
            row.bind("<Button-1>", lambda _e, i=idx: self._select_source(i))
            label.bind("<Button-1>", lambda _e, i=idx: self._select_source(i))

    def _toggle_source(self, idx):
        source = self.retrieval_sources[idx]
        source["enabled"] = bool(source["enabled_var"].get())
        self._persist_state_snapshot("retrieval_source_toggled")

    def _select_source(self, idx):
        self.selected_source_index = idx
        self._preview_source(idx)

    def _preview_source(self, idx):
        if not (0 <= idx < len(self.retrieval_sources)):
            return
        source = self.retrieval_sources[idx]
        preview = self._read_source_preview(source)
        self.source_preview_box.delete("1.0", tk.END)
        self.source_preview_box.insert("1.0", preview)

    def _read_source_preview(self, source):
        try:
            if source["kind"] == "file":
                return Path(source["path"]).read_text(encoding="utf-8", errors="ignore")[:5000]
            if source["kind"] == "url":
                return f"URL source:\n{source['path']}\n\nUse Preview Retrieval to fetch and rerank via the bridge."
        except Exception as exc:
            return f"Failed to preview source: {exc}"
        return "Unknown source"

    def _enabled_context_docs(self):
        docs = []
        for source in self.retrieval_sources:
            enabled = source.get("enabled")
            if "enabled_var" in source:
                enabled = bool(source["enabled_var"].get())
                source["enabled"] = enabled
            if not enabled:
                continue
            if source["kind"] == "file":
                docs.append({"path": source["path"]})
            elif source["kind"] == "url":
                docs.append({"url": source["path"]})
        return docs

    def _sync_profile_to_retrieval(self):
        self.retrieval_var_map["chunk_size"].set(self.profile_vars["chunk_size"].get() or str(DEFAULT_RETRIEVAL["chunk_size"]))
        self.retrieval_var_map["chunk_overlap"].set(self.profile_vars["chunk_overlap"].get() or str(DEFAULT_RETRIEVAL["chunk_overlap"]))
        self.retrieval_var_map["top_k"].set(self.profile_vars["retrieval_top_k"].get() or str(DEFAULT_RETRIEVAL["top_k"]))
        self.retrieval_var_map["max_context_chars"].set(
            self.profile_vars["max_context_chars"].get() or str(DEFAULT_RETRIEVAL["max_context_chars"])
        )
        self.include_sources_var.set(True)

    def _retrieval_payload(self):
        return {
            "query": self.retrieval_query_entry.get().strip() or self.responses_input_box.get("1.0", "end").strip() or self.chat_user_box.get("1.0", "end").strip(),
            "context_docs": self._enabled_context_docs(),
            "retrieval": {
                "chunk_size": self._safe_int(self.retrieval_var_map["chunk_size"].get(), DEFAULT_RETRIEVAL["chunk_size"]),
                "chunk_overlap": self._safe_int(self.retrieval_var_map["chunk_overlap"].get(), DEFAULT_RETRIEVAL["chunk_overlap"]),
                "top_k": self._safe_int(self.retrieval_var_map["top_k"].get(), DEFAULT_RETRIEVAL["top_k"]),
                "max_chunks": self._safe_int(self.retrieval_var_map["max_chunks"].get(), DEFAULT_RETRIEVAL["max_chunks"]),
                "max_chunk_chars": self._safe_int(
                    self.retrieval_var_map["max_chunk_chars"].get(), DEFAULT_RETRIEVAL["max_chunk_chars"]
                ),
                "max_context_chars": self._safe_int(
                    self.retrieval_var_map["max_context_chars"].get(), DEFAULT_RETRIEVAL["max_context_chars"]
                ),
                "include_sources": bool(self.include_sources_var.get()),
            },
        }

    def _preview_retrieval(self):
        payload = self._retrieval_payload()
        if not payload["context_docs"]:
            messagebox.showinfo("Retrieval Preview", "Add at least one workspace source first.")
            return
        if not payload["query"]:
            messagebox.showinfo("Retrieval Preview", "Provide a query in Retrieval, Responses, or Chat.")
            return

        def worker():
            with httpx.Client(timeout=60) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/api/v1/retrieve", json=payload)
                response.raise_for_status()
                return response.json()

        self._run_background("Previewing retrieval...", worker, on_success=self._apply_retrieval_preview)

    def _apply_retrieval_preview(self, result):
        chunks = result.get("chunks", [])
        lines = []
        for item in chunks:
            lines.append(
                f"source={item.get('source')} chunk={item.get('chunk_index')} score={item.get('score', 0):.4f}\n{item.get('text', '')}\n"
            )
        self.retrieval_chunks_box.delete("1.0", tk.END)
        self.retrieval_chunks_box.insert("1.0", "\n---\n".join(lines) if lines else "No chunks returned.")
        self.retrieval_context_box.delete("1.0", tk.END)
        self.retrieval_context_box.insert("1.0", result.get("context_text", ""))
        self._persist_state_snapshot("retrieval_preview", {"result": result})
        self.status_var.set("Retrieval preview updated")
        self._update_responses_preview()
        self._update_chat_preview()

    def _current_retrieval_config(self):
        return {
            "top_k": self._safe_int(self.retrieval_var_map["top_k"].get(), DEFAULT_RETRIEVAL["top_k"]),
            "chunk_size": self._safe_int(self.retrieval_var_map["chunk_size"].get(), DEFAULT_RETRIEVAL["chunk_size"]),
            "chunk_overlap": self._safe_int(
                self.retrieval_var_map["chunk_overlap"].get(), DEFAULT_RETRIEVAL["chunk_overlap"]
            ),
            "max_chunks": self._safe_int(self.retrieval_var_map["max_chunks"].get(), DEFAULT_RETRIEVAL["max_chunks"]),
            "max_chunk_chars": self._safe_int(
                self.retrieval_var_map["max_chunk_chars"].get(), DEFAULT_RETRIEVAL["max_chunk_chars"]
            ),
            "max_context_chars": self._safe_int(
                self.retrieval_var_map["max_context_chars"].get(), DEFAULT_RETRIEVAL["max_context_chars"]
            ),
            "include_sources": bool(self.include_sources_var.get()),
        }

    def _bridge_behavior_summary(self):
        profile = self._collect_profile_from_widgets()
        timeout_s = self._effective_timeout_seconds(profile["mode"], stream=False)
        lines = [
            f"model: {profile['model_id']}",
            f"mode: {profile['mode']}",
            f"temp/top_p/top_k: {profile['temperature']} / {profile['top_p']} / {profile['top_k']}",
            f"repeat_penalty: {profile['repeat_penalty']}",
            f"max_tokens: {profile['max_tokens']}",
            f"context_target: {profile['context_length']}",
            f"retrieval: chunk={profile['chunk_size']} overlap={profile['chunk_overlap']} top_k={profile['retrieval_top_k']} max_context_chars={profile['max_context_chars']}",
            f"embed_model: {profile['embed_model']}",
            f"rerank_model: {profile['rerank_model']}",
            f"validated_probe: {'yes' if self.validated_probe else 'no'}",
            f"source_count: {len(self._enabled_context_docs())}",
            f"timeout_policy: {self._timeout_policy_label(profile['mode'])} ({'none' if timeout_s is None else str(timeout_s) + 's'})",
        ]
        return "\n".join(lines)

    def _build_responses_payload(self):
        profile = self._collect_profile_from_widgets()
        payload = {
            "model": profile["model_id"] or self.role_vars["main"].get(),
            "instructions": self.responses_instructions_box.get("1.0", "end").strip() or profile["system_prompt"],
            "input": self.responses_input_box.get("1.0", "end").strip(),
            "reasoning": {"effort": self.reasoning_effort_var.get()},
            "temperature": profile["temperature"],
            "max_output_tokens": profile["max_tokens"],
            "context_docs": self._enabled_context_docs(),
            "retrieval": self._current_retrieval_config(),
            "mode": profile["mode"],
            "embed_model": profile["embed_model"],
            "rerank_model": profile["rerank_model"],
            "stream": False,
        }
        return payload

    def _update_responses_preview(self):
        payload = self._build_responses_payload()
        preview = {
            "request": payload,
            "bridge_behavior": self._bridge_behavior_summary(),
        }
        self.responses_preview_box.delete("1.0", tk.END)
        self.responses_preview_box.insert("1.0", json.dumps(preview, indent=2))
        self.responses_output_box.delete("1.0", tk.END)
        self.responses_output_box.insert("1.0", self._bridge_behavior_summary())

    def _timeout_policy_label(self, mode):
        if mode == "fast":
            return "fast_short"
        if mode in {"think", "architect"}:
            return "deep_long"
        return "balanced_default"

    def _effective_timeout_seconds(self, mode, stream=False):
        if stream:
            return None
        if mode == "fast":
            return 90
        if mode in {"think", "architect"}:
            return 300
        return 120

    def _is_low_resource(self):
        return bool(
            self.hardware
            and ((self.hardware.gpu_vram_gb or 0) <= 8.5 or (self.hardware.system_ram_gb or 0) <= 16.5)
        )

    def _estimate_prompt_tokens(self, payload):
        chars = 0
        if "input" in payload:
            chars += len(str(payload.get("instructions", "")))
            chars += len(str(payload.get("input", "")))
            retrieval = payload.get("retrieval", {})
            if isinstance(retrieval, dict):
                chars += int(retrieval.get("max_context_chars", DEFAULT_RETRIEVAL["max_context_chars"]))
        else:
            for msg in payload.get("messages", []):
                chars += len(str(msg.get("content", "")))
            retrieval = payload.get("extra_body", {}).get("retrieval", {})
            if isinstance(retrieval, dict):
                chars += int(retrieval.get("max_context_chars", DEFAULT_RETRIEVAL["max_context_chars"]))
        return max(1, chars // 4)

    def _apply_runtime_guardrails(self, payload, profile, api_kind):
        warnings = []
        retrieval = payload.get("retrieval") if api_kind == "responses" else payload.get("extra_body", {}).get("retrieval")
        if isinstance(retrieval, dict) and self._is_low_resource():
            retrieval["top_k"] = min(self._safe_int(retrieval.get("top_k"), DEFAULT_RETRIEVAL["top_k"]), 3)
            retrieval["max_context_chars"] = min(
                self._safe_int(retrieval.get("max_context_chars"), DEFAULT_RETRIEVAL["max_context_chars"]), 2400
            )
            retrieval["max_chunks"] = min(self._safe_int(retrieval.get("max_chunks"), DEFAULT_RETRIEVAL["max_chunks"]), 24)
            retrieval["chunk_size"] = min(self._safe_int(retrieval.get("chunk_size"), DEFAULT_RETRIEVAL["chunk_size"]), 700)
            retrieval["chunk_overlap"] = min(
                self._safe_int(retrieval.get("chunk_overlap"), DEFAULT_RETRIEVAL["chunk_overlap"]), 100
            )
            warnings.append("Low-resource guardrails applied to retrieval and context.")

        key = "max_output_tokens" if api_kind == "responses" else "max_tokens"
        max_output = self._safe_int(payload.get(key), 1024)
        context_target = max(self._safe_int(profile.get("context_length"), 4096), 1024)
        estimated_prompt = self._estimate_prompt_tokens(payload)
        budget_left = context_target - estimated_prompt - max_output
        if budget_left < 256:
            cap_chars = max(800, budget_left * 4)
            if isinstance(retrieval, dict):
                old_chars = self._safe_int(retrieval.get("max_context_chars"), DEFAULT_RETRIEVAL["max_context_chars"])
                retrieval["max_context_chars"] = min(old_chars, cap_chars)
            warnings.append(
                f"Context budget is tight: prompt~{estimated_prompt}t, output={max_output}t, ctx={context_target}t."
            )

        return payload, warnings

    def _bump_request_count(self, key):
        self.request_counts[key] = int(self.request_counts.get(key, 0)) + 1

    def _run_responses_request(self):
        profile = self._collect_profile_from_widgets()
        payload = self._build_responses_payload()
        payload, warnings = self._apply_runtime_guardrails(payload, profile, api_kind="responses")
        timeout_s = self._effective_timeout_seconds(profile["mode"], stream=bool(payload.get("stream")))
        self._persist_state_snapshot("responses_request_built", {"request": payload})
        self._bump_request_count("responses_request")
        if warnings:
            self.status_var.set(" | ".join(warnings))

        def worker():
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/v1/responses", json=payload)
                response.raise_for_status()
                return response.json()

        self._run_background("Running responses request...", worker, on_success=self._apply_responses_result)

    def _apply_responses_result(self, result):
        assistant_text = result.get("output_text", "")
        reasoning_texts = []
        tool_calls = result.get("tool_calls", [])
        for item in result.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "reasoning":
                    reasoning_texts.append(content.get("text", ""))
        self.responses_text_box.delete("1.0", tk.END)
        self.responses_text_box.insert("1.0", assistant_text)
        self.responses_reasoning_box.delete("1.0", tk.END)
        blocks = []
        if reasoning_texts:
            blocks.append("Reasoning:\n" + "\n\n".join(reasoning_texts))
        if tool_calls:
            blocks.append("Tool Calls:\n" + json.dumps(tool_calls, indent=2))
        if not blocks:
            blocks.append("No reasoning or tool calls returned.")
        self.responses_reasoning_box.insert("1.0", "\n\n".join(blocks))
        self.responses_meta_box.delete("1.0", tk.END)
        self.responses_meta_box.insert("1.0", json.dumps(result, indent=2))
        self.responses_output_box.delete("1.0", tk.END)
        self.responses_output_box.insert(
            "1.0",
            json.dumps(
                {
                    "retrieval_sources": result.get("metadata", {}).get("retrieval_sources", []),
                    "lmstudio_raw_id": result.get("metadata", {}).get("lmstudio_raw_id"),
                    "usage": result.get("usage"),
                },
                indent=2,
            ),
        )
        self._persist_state_snapshot("responses_result", {"result": result})
        self.status_var.set("Responses request completed")

    def _clear_responses_output(self):
        for widget in (
            self.responses_output_box,
            self.responses_text_box,
            self.responses_reasoning_box,
            self.responses_meta_box,
        ):
            widget.delete("1.0", tk.END)

    def _build_chat_payload(self):
        profile = self._collect_profile_from_widgets()
        messages = []
        system_text = self.chat_system_box.get("1.0", "end").strip() or profile["system_prompt"]
        user_text = self.chat_user_box.get("1.0", "end").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        if user_text:
            messages.append({"role": "user", "content": user_text})
        return {
            "model": profile["model_id"] or self.role_vars["main"].get(),
            "messages": messages,
            "temperature": profile["temperature"],
            "max_tokens": profile["max_tokens"],
            "stream": bool(self.chat_stream_var.get()),
            "extra_body": {
                "context_docs": self._enabled_context_docs(),
                "retrieval": self._current_retrieval_config(),
                "mode": profile["mode"],
                "embed_model": profile["embed_model"],
                "rerank_model": profile["rerank_model"],
            },
        }

    def _update_chat_preview(self):
        payload = self._build_chat_payload()
        self.chat_preview_box.delete("1.0", tk.END)
        self.chat_preview_box.insert(
            "1.0",
            json.dumps({"request": payload, "bridge_behavior": self._bridge_behavior_summary()}, indent=2),
        )

    def _run_chat_request(self):
        profile = self._collect_profile_from_widgets()
        payload = self._build_chat_payload()
        payload["stream"] = False
        payload, warnings = self._apply_runtime_guardrails(payload, profile, api_kind="chat")
        timeout_s = self._effective_timeout_seconds(profile["mode"], stream=bool(payload.get("stream")))
        self._persist_state_snapshot("chat_request_built", {"request": payload})
        self._bump_request_count("chat_request")
        if warnings:
            self.status_var.set(" | ".join(warnings))

        def worker():
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()

        self._run_background("Running chat request...", worker, on_success=self._apply_chat_result)

    def _apply_chat_result(self, result):
        choices = result.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        self.chat_text_box.delete("1.0", tk.END)
        self.chat_text_box.insert("1.0", message.get("content", ""))
        self.chat_raw_box.delete("1.0", tk.END)
        self.chat_raw_box.insert("1.0", json.dumps(result, indent=2))
        self.chat_output_box.delete("1.0", tk.END)
        self.chat_output_box.insert("1.0", message.get("content", ""))
        self._persist_state_snapshot("chat_result", {"result": result})
        self.status_var.set("Chat request completed")

    def _selected_model_or_warn(self):
        model_id = self._get_selected_model_id()
        if not model_id:
            messagebox.showinfo("Select Model", "Choose a model first.")
            return None
        return model_id

    def _run_connectivity_check(self):
        base = self.bridge_base_var.get().strip()

        def worker():
            bench = StreamingBenchmark(base_url=base)
            return asyncio.run(bench.check_server())

        self._bump_request_count("probe_connectivity")
        self._run_background("Checking connectivity...", worker, on_success=lambda data: self._show_benchmark_result("Connectivity", data))

    def _run_model_list_check(self):
        base = self.lmstudio_base_var.get().strip()

        def worker():
            bench = StreamingBenchmark(base_url=base)
            return asyncio.run(bench.list_models())

        self._bump_request_count("probe_model_list")
        self._run_background("Listing models...", worker, on_success=lambda data: self._show_benchmark_result("Models", data))

    def _probe_selected_model(self):
        model_id = self._selected_model_or_warn()
        if not model_id:
            return
        self.probe_mode_var.set("quick")
        base = self.bridge_base_var.get().strip()

        def worker():
            started = time.time()
            bench = StreamingBenchmark(base_url=base, timeout=45)
            connectivity = asyncio.run(bench.check_server())
            model_list = asyncio.run(bench.list_models())
            caps = asyncio.run(bench.detect_capabilities(model_id))
            elapsed = round((time.time() - started) * 1000, 2)
            return {
                "probe_mode": "quick",
                "execution": {
                    "checks": ["connectivity", "model_list", "capabilities"],
                    "requests_sent": 6,
                    "elapsed_ms": elapsed,
                },
                "connectivity": connectivity,
                "model_list": model_list,
                "capabilities": vars(caps),
                "comparison": {},
            }

        self._bump_request_count("probe_quick")
        self._run_background("Probing selected model...", worker, on_success=self._apply_probe_result)

    def _compare_streaming_modes(self):
        model_id = self._selected_model_or_warn()
        if not model_id:
            return
        self.probe_mode_var.set("full-streaming")
        base = self.bridge_base_var.get().strip()

        def worker():
            bench = StreamingBenchmark(base_url=base)
            return asyncio.run(bench.compare_streaming_vs_nonstreaming(model_id, prompt_key="code"))

        self._bump_request_count("probe_full_streaming_compare")
        self._run_background("Comparing streaming modes...", worker, on_success=lambda data: self._show_benchmark_result("Streaming Compare", data))

    def _apply_probe_result(self, result):
        self.validated_probe = result
        capabilities = result["capabilities"]
        comparison = result.get("comparison", {}) or {}
        warnings = self._derive_probe_warnings(capabilities, comparison)
        tuned = self._build_tuned_profile(capabilities, comparison)
        self.validated_tuned_profile = tuned
        summary = {
            "probe_mode": result.get("probe_mode", "quick"),
            "execution": result.get("execution", {}),
            "capabilities": capabilities,
            "tuned_profile_preview": tuned,
            "comparison": comparison.get("comparison", {}),
        }
        self.benchmark_summary_box.delete("1.0", tk.END)
        self.benchmark_summary_box.insert("1.0", json.dumps(summary, indent=2))
        self.benchmark_raw_box.delete("1.0", tk.END)
        self.benchmark_raw_box.insert("1.0", json.dumps(result, indent=2))
        self.benchmark_warnings_box.delete("1.0", tk.END)
        self.benchmark_warnings_box.insert("1.0", "\n".join(warnings) if warnings else "No major warnings.")
        self.profile_status_var.set("Validated probe available. Use Apply Tuned Settings to promote it.")
        self._persist_state_snapshot("probe_result", {"result": result, "warnings": warnings})
        self.status_var.set("Model probe completed")

    def _show_benchmark_result(self, title, data):
        self.benchmark_summary_box.delete("1.0", tk.END)
        self.benchmark_summary_box.insert("1.0", f"{title}\n\n{json.dumps(data, indent=2)}")
        self.benchmark_raw_box.delete("1.0", tk.END)
        self.benchmark_raw_box.insert("1.0", json.dumps(data, indent=2))
        self._persist_state_snapshot("benchmark_result", {"title": title, "result": data})
        self.status_var.set(f"{title} finished")

    def _run_chat_smoke_test(self):
        profile = self._collect_profile_from_widgets()
        payload = {
            "model": profile["model_id"] or self.role_vars["main"].get(),
            "messages": [
                {"role": "system", "content": "You are a precise local assistant."},
                {"role": "user", "content": "Reply with the text CHAT_OK and one short sentence."},
            ],
            "max_tokens": min(profile["max_tokens"], 128),
            "temperature": min(profile["temperature"], 0.3),
            "stream": False,
        }

        def worker():
            with httpx.Client(timeout=60) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/v1/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()

        self._bump_request_count("chat_smoke_test")
        self._run_background("Running chat smoke test...", worker, on_success=lambda data: self._show_benchmark_result("Chat Smoke Test", data))

    def _run_agent_smoke_test(self):
        profile = self._collect_profile_from_widgets()
        payload = {
            "model": profile["model_id"] or self.role_vars["main"].get(),
            "instructions": "Act like a local coding agent and stay concise.",
            "input": "Reply with AGENT_OK and mention whether retrieval sources are attached.",
            "reasoning": {"effort": self.reasoning_effort_var.get()},
            "max_output_tokens": min(profile["max_tokens"], 160),
            "temperature": min(profile["temperature"], 0.3),
            "context_docs": self._enabled_context_docs(),
            "retrieval": self._current_retrieval_config(),
            "mode": profile["mode"],
            "stream": False,
        }

        def worker():
            with httpx.Client(timeout=60) as client:
                response = client.post(f"{self.bridge_base_var.get().strip()}/v1/responses", json=payload)
                response.raise_for_status()
                return response.json()

        self._bump_request_count("agent_smoke_test")
        self._run_background("Running agent smoke test...", worker, on_success=lambda data: self._show_benchmark_result("Agent Smoke Test", data))

    def _derive_probe_warnings(self, capabilities, comparison):
        warnings = []
        comp = comparison.get("comparison", {})
        streaming_overhead = comp.get("streaming_overhead_ms", 0) or 0
        if streaming_overhead > 5000:
            warnings.append(f"Poor streaming overhead detected: {streaming_overhead} ms.")
        if comp.get("recommendation") == "non_streaming":
            warnings.append("Benchmark prefers non-streaming for this model.")
        stream_content = comparison.get("streaming", {}).get("content_analysis", {})
        if stream_content.get("truncation_issues"):
            warnings.append("Streaming output showed truncation risks: " + ", ".join(stream_content["truncation_issues"]))
        if not capabilities.get("reasoning"):
            warnings.append("Weak reasoning support detected; avoid forcing think mode.")
        if not capabilities.get("tool_use"):
            warnings.append("Tool use support looks weak or absent.")
        if capabilities.get("max_context", 4096) <= 4096:
            warnings.append("Low effective context fit; keep injected context compact.")
        return warnings

    def _build_tuned_profile(self, capabilities, comparison):
        base = self._predicted_profile_from_rec(self.selected_rec) if self.selected_rec else dict(self.current_profile)
        tuned = dict(base)
        tuned["status"] = "validated"
        if capabilities.get("reasoning"):
            tuned["mode"] = "think"
        elif tuned.get("mode") == "think":
            tuned["mode"] = "fast"
        max_context = capabilities.get("max_context", tuned.get("context_length", 8192))
        tuned["context_length"] = min(tuned["context_length"], max_context)
        if max_context <= 4096:
            tuned["max_context_chars"] = min(tuned["max_context_chars"], 3200)
            tuned["retrieval_top_k"] = min(tuned["retrieval_top_k"], 3)
        if comparison.get("comparison", {}).get("recommendation") == "non_streaming":
            tuned["temperature"] = min(tuned["temperature"], 0.3)
        return tuned

    def _apply_tuned_profile(self):
        if not self.validated_tuned_profile:
            messagebox.showinfo("Apply Tuned Settings", "Run a model probe first.")
            return
        self._apply_profile_to_widgets(self.validated_tuned_profile, status_text="Validated benchmark profile applied")
        self.status_var.set("Validated settings promoted to active profile")

    def _apply_selected_preset(self):
        preset_path = self._selected_preset_path()
        if not preset_path:
            messagebox.showinfo("Preset", "No preset selected.")
            return
        try:
            preset = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._show_error("Failed to read preset", exc)
            return

        profile = self._predicted_profile_from_rec(self.selected_rec) if self.selected_rec else dict(DEFAULT_RETRIEVAL)
        profile.update({
            "status": "preset-loaded",
            "model_id": self._get_selected_model_id() or self.role_vars["main"].get(),
            "system_prompt": self.system_prompt_box.get("1.0", "end").strip() or "You are a helpful local AI assistant.",
            "temperature": self.current_profile.get("temperature", 0.3),
            "top_p": self.current_profile.get("top_p", 0.95),
            "top_k": self.current_profile.get("top_k", 40),
            "repeat_penalty": self.current_profile.get("repeat_penalty", 1.1),
            "max_tokens": self.current_profile.get("max_tokens", 2048),
            "context_length": self.current_profile.get("context_length", 8192),
            "mode": self.current_profile.get("mode", "fast"),
            "chunk_size": self.current_profile.get("chunk_size", DEFAULT_RETRIEVAL["chunk_size"]),
            "chunk_overlap": self.current_profile.get("chunk_overlap", DEFAULT_RETRIEVAL["chunk_overlap"]),
            "retrieval_top_k": self.current_profile.get("retrieval_top_k", DEFAULT_RETRIEVAL["top_k"]),
            "max_context_chars": self.current_profile.get("max_context_chars", DEFAULT_RETRIEVAL["max_context_chars"]),
            "embed_model": self.current_profile.get("embed_model", self.role_vars["embed"].get()),
            "rerank_model": self.current_profile.get("rerank_model", self.role_vars["rerank"].get()),
        })

        def get_field(section, key):
            for field in preset.get(section, {}).get("fields", []):
                if field.get("key") == key:
                    return field.get("value")
            return None

        def get_legacy(section, key, default=None):
            return preset.get(section, {}).get(key, default)

        profile["temperature"] = get_field("operation", "llm.prediction.temperature") or profile["temperature"]
        if get_legacy("inference_params", "temp") is not None:
            profile["temperature"] = get_legacy("inference_params", "temp", profile["temperature"])
        top_p_val = get_field("operation", "llm.prediction.topPSampling")
        if isinstance(top_p_val, dict):
            profile["top_p"] = top_p_val.get("value", profile["top_p"])
        elif get_legacy("inference_params", "top_p") is not None:
            profile["top_p"] = get_legacy("inference_params", "top_p", profile["top_p"])
        top_k_val = get_field("operation", "llm.prediction.topKSampling")
        if isinstance(top_k_val, dict):
            profile["top_k"] = top_k_val.get("value", profile["top_k"])
        elif get_legacy("inference_params", "top_k") is not None:
            profile["top_k"] = get_legacy("inference_params", "top_k", profile["top_k"])
        repeat_penalty = get_field("operation", "llm.prediction.repeatPenalty")
        if isinstance(repeat_penalty, dict):
            profile["repeat_penalty"] = repeat_penalty.get("value", profile["repeat_penalty"])
        elif get_legacy("inference_params", "repeat_penalty") is not None:
            profile["repeat_penalty"] = get_legacy("inference_params", "repeat_penalty", profile["repeat_penalty"])
        profile["max_tokens"] = (
            get_field("operation", "llm.prediction.maxTokens")
            or get_legacy("inference_params", "n_predict")
            or profile["max_tokens"]
        )
        profile["system_prompt"] = (
            get_field("operation", "llm.prediction.systemPrompt")
            or get_legacy("inference_params", "pre_prompt")
            or profile["system_prompt"]
        )
        profile["context_length"] = (
            get_field("load", "llm.load.contextLength")
            or get_legacy("load_params", "n_ctx")
            or profile["context_length"]
        )
        profile["mode"] = preset.get("_bridge_profile", {}).get("mode", profile["mode"])
        profile["embed_model"] = preset.get("_bridge_profile", {}).get("embed_model", profile["embed_model"])
        profile["rerank_model"] = preset.get("_bridge_profile", {}).get("rerank_model", profile["rerank_model"])
        retrieval = preset.get("_bridge_profile", {}).get("retrieval", {})
        if isinstance(retrieval, dict):
            profile["chunk_size"] = retrieval.get("chunk_size") or profile["chunk_size"]
            profile["chunk_overlap"] = retrieval.get("chunk_overlap") or profile["chunk_overlap"]
            profile["retrieval_top_k"] = retrieval.get("top_k") or profile["retrieval_top_k"]
            profile["max_context_chars"] = retrieval.get("max_context_chars") or profile["max_context_chars"]
        self._apply_profile_to_widgets(profile, status_text=f"Preset applied from {preset_path.name}")
        self._persist_state_snapshot("preset_applied", {"preset": str(preset_path)})
        self.status_var.set(f"Applied preset {preset_path.name}")

    def _rec_to_export(self):
        rec = self.selected_rec
        if not rec:
            return None
        profile = self._collect_profile_from_widgets()
        rec.temperature = profile["temperature"]
        rec.top_p = profile["top_p"]
        rec.top_k = profile["top_k"]
        rec.repeat_penalty = profile["repeat_penalty"]
        rec.max_tokens = profile["max_tokens"]
        rec.context_length = profile["context_length"]
        rec.system_prompt = profile["system_prompt"]
        return rec

    def _export_selected_preset(self):
        rec = self._rec_to_export()
        if not rec:
            return
        target_dir = Path.home() / ".cache" / "lm-studio" / "config-presets"
        safe_id = rec.model_id.replace("/", "_")
        target_path = target_dir / f"{safe_id}_agent_console_v7.preset.json"
        try:
            profile = self._collect_profile_from_widgets()
            hardware = self.hardware.model_dump() if self.hardware else {}
            ConfigExporter.export_preset(
                rec,
                target_path,
                profile=profile,
                hardware=hardware,
                name=f"{rec.model_id} - Agent Console V7",
                identifier=f"@local:{rec.model_id.replace('/', '-').lower()}-agent-v7",
            )
            self._load_available_presets()
            self.status_var.set(f"Preset exported to {target_path}")
            self._persist_state_snapshot("preset_exported", {"path": str(target_path)})
            messagebox.showinfo("Preset Exported", f"Saved preset to:\n{target_path}")
        except Exception as exc:
            self._show_error("Preset export failed", exc)

    def _env_preview_text(self):
        profile = {
            "embed_model": self._clean_model_id(self.profile_vars["embed_model"].get().strip() or self.role_vars["embed"].get()),
            "rerank_model": self._clean_model_id(self.profile_vars["rerank_model"].get().strip() or self.role_vars["rerank"].get()),
            "retrieval_top_k": self._safe_int(self.profile_vars["retrieval_top_k"].get(), DEFAULT_RETRIEVAL["top_k"]),
            "chunk_size": self._safe_int(self.profile_vars["chunk_size"].get(), DEFAULT_RETRIEVAL["chunk_size"]),
            "chunk_overlap": self._safe_int(self.profile_vars["chunk_overlap"].get(), DEFAULT_RETRIEVAL["chunk_overlap"]),
        }
        lines = [
            f"LMSTUDIO_BASE_URL={self.lmstudio_base_var.get().strip()}",
            f"BRIDGE_HOST={self._bridge_host()}",
            f"BRIDGE_PORT={self._bridge_port()}",
            f"MAIN_MODEL={self._clean_model_id(self.role_vars['main'].get())}",
            f"REASONING_MODEL={self._clean_model_id(self.role_vars['reasoning'].get())}",
            f"EMBED_MODEL={profile['embed_model']}",
            f"RERANK_MODEL={profile['rerank_model']}",
            f"RERANK_TOP_K={profile['retrieval_top_k']}",
            f"DEFAULT_CHUNK_SIZE={profile['chunk_size']}",
            f"DEFAULT_CHUNK_OVERLAP={profile['chunk_overlap']}",
        ]
        return "\n".join(lines)

    def _update_env_preview(self):
        self.env_preview_box.delete("1.0", tk.END)
        self.env_preview_box.insert("1.0", self._env_preview_text())

    def _copy_env_profile(self):
        text = self._env_preview_text()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Bridge env profile copied to clipboard")
        self._persist_state_snapshot("env_profile_copied")

    def _clean_model_id(self, value):
        text = str(value or "").strip()
        match = re.search(r"model_key='([^']+)'", text)
        if match:
            return match.group(1)
        return text

    def _normalized_model_name(self, value):
        text = self._clean_model_id(value).lower().replace("\\", "/")
        text = text.rsplit("/", 1)[-1]
        text = text.rsplit(":", 1)[-1]
        text = re.sub(r"[^a-z0-9._-]+", "", text)
        return text

    def _state_snapshot(self):
        profile = self._collect_profile_from_widgets()
        timeout_s = self._effective_timeout_seconds(profile.get("mode", "fast"), stream=False)
        return {
            "selected_model": self._get_selected_model_id(),
            "profile": profile,
            "roles": {key: var.get() for key, var in self.role_vars.items()},
            "models": self.models,
            "retrieval_sources": [
                {
                    "kind": source.get("kind"),
                    "path": source.get("path"),
                    "label": source.get("label"),
                    "enabled": bool(source.get("enabled", True)),
                }
                for source in self.retrieval_sources
            ],
            "validated_probe": self.validated_probe,
            "probe_mode": self.probe_mode_var.get(),
            "timeout_policy": {
                "label": self._timeout_policy_label(profile.get("mode", "fast")),
                "seconds": timeout_s,
            },
            "request_counts": dict(self.request_counts),
            "runtime_status": self.runtime_status_var.get(),
            "bridge_base": self.bridge_base_var.get().strip(),
            "lmstudio_base": self.lmstudio_base_var.get().strip(),
        }

    def _workflow_map_text(self, event_name):
        profile = self.current_profile or {}
        lines = [
            "# GUI Workflow Map",
            "",
            f"- Event: `{event_name}`",
            f"- Selected model: `{self._get_selected_model_id() or 'none'}`",
            f"- Main role: `{self.role_vars['main'].get()}`",
            f"- Embed role: `{self.role_vars['embed'].get()}`",
            f"- Rerank role: `{self.role_vars['rerank'].get()}`",
            f"- Profile status: `{profile.get('status', 'unknown')}`",
            f"- Mode: `{profile.get('mode', 'fast')}`",
            f"- Probe mode: `{self.probe_mode_var.get()}`",
            f"- Timeout policy: `{self._timeout_policy_label(profile.get('mode', 'fast'))}`",
            f"- Request counters: `{json.dumps(self.request_counts, sort_keys=True)}`",
            f"- Retrieval sources enabled: `{len(self._enabled_context_docs())}`",
            "",
            "## Workflows",
            "",
            "- `Runtime`: bridge health, role loading, loaded-model reuse.",
            "- `Responses`: OpenAI-style responses request builder and result viewer.",
            "- `Chat`: compatibility path for `chat.completions`.",
            "- `Retrieval`: source registration, chunk preview, rerank preview, injected context preview.",
            "- `Benchmark`: connectivity, model list, probe, smoke tests.",
            "- `Profile`: hardware-aware prediction, preset load/export, env preview.",
        ]
        return "\n".join(lines) + "\n"

    def _persist_state_snapshot(self, event_name, extra=None):
        try:
            self.state_dir.mkdir(exist_ok=True)
            payload = {
                "event": event_name,
                "snapshot": self._state_snapshot(),
            }
            if extra is not None:
                payload["extra"] = extra
            (self.state_dir / "console_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (self.state_dir / "workflows.md").write_text(self._workflow_map_text(event_name), encoding="utf-8")
        except Exception:
            pass

    def _safe_int(self, value, default):
        try:
            return int(float(value))
        except Exception:
            return default

    def _safe_float(self, value, default):
        try:
            return float(value)
        except Exception:
            return default

    def _bridge_host(self):
        base = self.bridge_base_var.get().strip()
        if not base:
            return DEFAULT_BRIDGE_HOST
        if "://" not in base:
            return base.split(":", 1)[0] or DEFAULT_BRIDGE_HOST
        parsed = urlparse(base)
        return parsed.hostname or DEFAULT_BRIDGE_HOST

    def _bridge_port(self):
        base = self.bridge_base_var.get().strip()
        if not base:
            return DEFAULT_BRIDGE_PORT
        if "://" not in base:
            try:
                return int(base.rsplit(":", 1)[1])
            except Exception:
                return DEFAULT_BRIDGE_PORT
        parsed = urlparse(base)
        return parsed.port or DEFAULT_BRIDGE_PORT

    def _show_error(self, title, exc):
        self.status_var.set(f"{title}: {exc}")
        messagebox.showerror(title, str(exc))


if __name__ == "__main__":
    app = LMStudioAgentConsole()
    app.mainloop()
