"""
LM Studio Config Wizard — V3 TUI
Rich terminal UI powered by Textual. Shows hardware profiling, model discovery,
smart VRAM-aware recommendations, context engineering profiles, and live preset preview.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, DataTable, Static, Button,
    Select, Label, Rule, TabbedContent, TabPane, Collapsible
)
from textual.screen import Screen
from textual.reactive import reactive
from textual import work

from pathlib import Path
import json
import time

from hardware_detector import HardwareDetector
from engine import RecommendationEngine, USE_CASE_PROFILES
from model_discovery import get_local_models, get_model_path, extract_model_specs
from exporters import ConfigExporter
from models import ModelRecommendation


# ─── Palette helpers ─────────────────────────────────────────────────────────
def _vram_bar(used: float, total: float, width: int = 20) -> str:
    """Render a simple block-character VRAM bar."""
    if not total:
        return "N/A"
    pct = min(used / total, 1.0)
    filled = int(pct * width)
    color = "green" if pct < 0.7 else ("yellow" if pct < 0.9 else "red")
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {used:.1f}/{total:.1f} GB"


def _quality_badge(score: float) -> str:
    if score >= 9.0:
        return f"[bold green]★ {score:.1f}[/bold green]"
    elif score >= 7.5:
        return f"[bold yellow]◆ {score:.1f}[/bold yellow]"
    else:
        return f"[dim red]▼ {score:.1f}[/dim red]"


def _offload_label(gpu_layers: int) -> str:
    if gpu_layers >= 999:
        return "[bold green]GPU 100%[/bold green]"
    elif gpu_layers == 0:
        return "[dim]CPU only[/dim]"
    else:
        return f"[yellow]Hybrid ({gpu_layers}L)[/yellow]"


# ─── Export Confirmation Modal ────────────────────────────────────────────────
class ExportModal(Screen):
    """Confirmation + preview screen before writing preset JSON."""

    def __init__(self, rec: ModelRecommendation, target_path: Path):
        super().__init__()
        self.rec = rec
        self.target_path = target_path

    def compose(self) -> ComposeResult:
        preset = ConfigExporter.build_preset_dict(self.rec)
        preview = json.dumps(preset, indent=2)[:1200]   # truncate for display
        
        yield Vertical(
            Label("📋  Export Preview", id="modal-title"),
            Rule(),
            Static(
                f"[bold]Model:[/bold]  {self.rec.model_id}\n"
                f"[bold]Quant:[/bold]  {self.rec.quantization.value}  "
                f"[bold]Context:[/bold]  {self.rec.context_length:,} tokens\n"
                f"[bold]GPU Layers:[/bold]  {self.rec.gpu_layers}  "
                f"[bold]Est. VRAM:[/bold]  {self.rec.estimated_vram_gb:.2f} GB\n"
                f"[bold]Thinking:[/bold]  {'✓ Enabled' if self.rec.enable_thinking else '✗ Off'}  "
                f"[bold]Temp:[/bold]  {self.rec.temperature}  "
                f"[bold]Top-P:[/bold]  {self.rec.top_p}",
                id="modal-summary",
            ),
            Rule(),
            Static(f"[dim]{preview}[/dim]", id="modal-preview"),
            Rule(),
            Static(f"[bold cyan]→ {self.target_path}[/bold cyan]", id="modal-path"),
            Horizontal(
                Button("✅  Confirm Export", id="btn-confirm", variant="success"),
                Button("✗  Cancel", id="btn-cancel", variant="error"),
                id="modal-buttons",
            ),
            id="modal-box",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.app.pop_screen()
        elif event.button.id == "btn-confirm":
            self.app.pop_screen()
            self.app.perform_export(self.rec, self.target_path)


# ─── Main Application ─────────────────────────────────────────────────────────
class LMStudioConfigApp(App):
    """LM Studio Config Wizard V3 — Full-featured TUI."""

    TITLE = "LM Studio Config Wizard"
    SUB_TITLE = "Best Configuration Tool"

    CSS = """
    /* ── Global ── */
    Screen { background: #0d1117; }

    /* ── Header banner ── */
    #banner {
        padding: 1 2;
        background: #1a1f2e;
        color: #58a6ff;
        text-align: center;
        text-style: bold;
        border-bottom: solid #30363d;
    }

    /* ── Hardware profiling panel ── */
    #hw-panel {
        height: auto;
        padding: 1 2;
        border: solid #30363d;
        background: #161b22;
        margin-bottom: 1;
    }
    #hw-title { text-style: bold; color: #58a6ff; }
    #hw-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1 4;
        height: auto;
    }
    .hw-cell { color: #c9d1d9; }
    .hw-label { color: #8b949e; }

    /* ── Model selector ── */
    #model-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }
    #model-select { width: 60%; }
    #use-case-select { width: 36%; margin-left: 2; }

    /* ── Results tabs ── */
    TabbedContent { height: 1fr; }

    /* ── Recommendations table ── */
    #results-table {
        height: 12;
        border: solid #30363d;
    }

    /* ── Inference / Context panel ── */
    #context-panel {
        padding: 1 2;
        border: solid #30363d;
        background: #161b22;
        height: auto;
    }
    #system-prompt-display {
        color: #7ee787;
        padding: 1;
        border: dashed #30363d;
        height: auto;
    }
    .inf-row { height: auto; margin: 0; }
    .inf-key { color: #8b949e; width: 18; }
    .inf-val { color: #f0883e; }

    /* ── Action bar ── */
    #action-bar {
        height: 3;
        align: center middle;
        background: #161b22;
        border-top: solid #30363d;
        padding: 0 2;
    }
    #btn-export { margin: 0 1; }
    #btn-refresh { margin: 0 1; }

    /* ── Modal ── */
    ExportModal > Vertical {
        background: #1a1f2e;
        border: heavy #58a6ff;
        padding: 2 3;
        width: 70%;
        height: auto;
        align: center middle;
    }
    #modal-title { text-style: bold; color: #58a6ff; text-align: center; }
    #modal-summary { color: #c9d1d9; }
    #modal-preview { color: #6e7681; height: 10; overflow: auto hidden; }
    #modal-path { color: #58a6ff; text-align: center; }
    #modal-buttons { align: center middle; height: 3; margin-top: 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_hw", "Refresh Hardware"),
        ("e", "export", "Export"),
        ("ctrl+a", "export_all", "Export All"),
    ]

    # Reactive: model currently being analysed
    selected_model_id: reactive[str] = reactive("")

    def __init__(self):
        super().__init__()
        self.detector = HardwareDetector()
        self.hardware = None
        self.engine = None
        self.local_models: list[dict] = []
        self.current_recs: list[ModelRecommendation] = []
        self.selected_rec: ModelRecommendation | None = None
        self.active_use_case = "balanced"

    # ── Layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with ScrollableContainer(id="main-scroll"):
            yield Static(
                "🚀  LM Studio Config Wizard  ·  Best Configuration Engine  ·  V3",
                id="banner",
            )

            # Hardware Profiling Panel
            with Vertical(id="hw-panel"):
                yield Static("⚙  Hardware Profile", id="hw-title")
                with Horizontal(id="hw-grid"):
                    yield Static("", id="hw-cpu", classes="hw-cell")
                    yield Static("", id="hw-ram", classes="hw-cell")
                    yield Static("", id="hw-gpu", classes="hw-cell")
                    yield Static("", id="hw-vram", classes="hw-cell")
                    yield Static("", id="hw-cuda", classes="hw-cell")
                    yield Static("", id="hw-platform", classes="hw-cell")

            # Model + use-case row
            with Horizontal(id="model-row"):
                yield Select(
                    [("  Scanning local models…", "")],
                    id="model-select",
                    prompt="Select model…",
                )
                yield Select(
                    [(label, key) for key, label in USE_CASE_PROFILES.items()],
                    id="use-case-select",
                    prompt="Use-case profile…",
                    value="balanced",
                )

            # Tabbed results
            with TabbedContent():
                with TabPane("📊 Recommendations", id="tab-recs"):
                    yield DataTable(id="results-table")
                    with Horizontal(id="action-bar"):
                        yield Button(
                            "💾  Export Selected", id="btn-export",
                            disabled=True, variant="primary"
                        )
                        yield Button(
                            "📤  Export All Top-5", id="btn-export-all",
                            disabled=True, variant="default"
                        )
                        yield Button(
                            "🔄  Refresh HW", id="btn-refresh", variant="default"
                        )

                with TabPane("🧠 Context Engineering", id="tab-ctx"):
                    with Vertical(id="context-panel"):
                        yield Static("", id="ctx-header")
                        yield Rule()
                        yield Static(
                            "Thinking / Chain-of-Thought:", classes="hw-label"
                        )
                        yield Static("", id="ctx-thinking")
                        yield Rule()
                        yield Static("System Prompt:", classes="hw-label")
                        yield Static("", id="system-prompt-display")
                        yield Rule()
                        yield Static("Inference Parameters:", classes="hw-label")
                        with Horizontal(classes="inf-row"):
                            yield Static("Temperature", classes="inf-key")
                            yield Static("", id="inf-temp", classes="inf-val")
                        with Horizontal(classes="inf-row"):
                            yield Static("Top-P", classes="inf-key")
                            yield Static("", id="inf-topp", classes="inf-val")
                        with Horizontal(classes="inf-row"):
                            yield Static("Top-K", classes="inf-key")
                            yield Static("", id="inf-topk", classes="inf-val")
                        with Horizontal(classes="inf-row"):
                            yield Static("Repeat Penalty", classes="inf-key")
                            yield Static("", id="inf-rep", classes="inf-val")
                        with Horizontal(classes="inf-row"):
                            yield Static("Max Tokens", classes="inf-key")
                            yield Static("", id="inf-maxtok", classes="inf-val")

                with TabPane("🔩 VRAM Breakdown", id="tab-vram"):
                    with Vertical(id="vram-panel"):
                        yield Static("", id="vram-detail")

        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def on_mount(self) -> None:
        self._setup_table()
        self._detect_hardware()
        self._load_models()

    def _setup_table(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Rank", "Quantization", "Context", "GPU Offload",
            "Est. VRAM", "Quality", "Temp", "Thinking"
        )

    def _detect_hardware(self) -> None:
        self.hardware = self.detector.detect()
        self.engine = RecommendationEngine(self.hardware)
        self._render_hw_panel()

    def _render_hw_panel(self) -> None:
        hw = self.hardware
        vram_total = hw.gpu_vram_gb or 0
        self.query_one("#hw-cpu").update(
            f"[dim]CPU[/dim]  {hw.cpu_cores}P / {hw.logical_cores}L cores"
        )
        self.query_one("#hw-ram").update(
            f"[dim]RAM[/dim]  {hw.system_ram_gb:.1f} GB"
        )
        self.query_one("#hw-gpu").update(
            f"[dim]GPU[/dim]  {hw.gpu_name or '[red]None detected[/red]'}"
        )
        self.query_one("#hw-vram").update(
            f"[dim]VRAM[/dim]  {_vram_bar(0, vram_total)} available"
            if vram_total else "[dim]VRAM[/dim]  Shared / N/A"
        )
        self.query_one("#hw-cuda").update(
            f"[dim]CUDA[/dim]  {hw.cuda_version or 'Not detected'}"
        )
        self.query_one("#hw-platform").update(
            f"[dim]Platform[/dim]  {hw.platform.title()}"
        )

    def _load_models(self) -> None:
        self.local_models = get_local_models()
        select = self.query_one("#model-select", Select)
        options = []
        for m in self.local_models:
            label = f"{m['id']}  ({m.get('params','?')}, {m.get('arch','?')}, {m.get('size','?')})"
            options.append((label, m["id"]))
        if options:
            select.set_options(options)
        else:
            select.set_options([("  No local models found via lms ls", "none")])

    # ── Reactive Events ───────────────────────────────────────────────────────
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.control.id == "model-select" and event.value and event.value != "none":
            self.selected_model_id = str(event.value)
            self._run_optimization()
        elif event.control.id == "use-case-select" and event.value:
            self.active_use_case = str(event.value)
            if self.selected_model_id:
                self._run_optimization()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(event.row_key.value)
            self.selected_rec = self.current_recs[idx]
            self._render_context_tab(self.selected_rec)
            self._render_vram_tab(self.selected_rec)
        except (ValueError, IndexError):
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-export":
            self._trigger_export()
        elif event.button.id == "btn-export-all":
            self._export_all()
        elif event.button.id == "btn-refresh":
            self.action_refresh_hw()

    # ── Actions ───────────────────────────────────────────────────────────────
    def action_refresh_hw(self) -> None:
        self._detect_hardware()
        self.notify("Hardware profile refreshed", severity="information")

    def action_export(self) -> None:
        self._trigger_export()

    def action_export_all(self) -> None:
        self._export_all()

    # ── Optimization Core ─────────────────────────────────────────────────────
    def _run_optimization(self) -> None:
        model_id = self.selected_model_id
        m_info = next((m for m in self.local_models if m["id"] == model_id), None)
        if not m_info:
            return

        # Parse params
        params_str = m_info.get("params", "7B")
        try:
            params_b = float("".join(c for c in params_str if c.isdigit() or c == "."))
        except Exception:
            params_b = 7.0

        # Deep spec extraction
        layers, hidden_size = 32, 4096
        m_path = get_model_path(model_id)
        if m_path:
            specs = extract_model_specs(m_path)
            if specs:
                layers = specs.get("num_layers") or 32
                hidden_size = specs.get("hidden_size") or 4096

        self.current_recs = self.engine.recommend(
            model_id=model_id,
            params_b=params_b,
            num_layers=layers,
            hidden_size=hidden_size,
            use_case=self.active_use_case,
        )

        self._render_results_table()

        # Auto-select top recommendation
        if self.current_recs:
            self.selected_rec = self.current_recs[0]
            self._render_context_tab(self.selected_rec)
            self._render_vram_tab(self.selected_rec)

        # Enable action buttons
        has_recs = bool(self.current_recs)
        self.query_one("#btn-export", Button).disabled = not has_recs
        self.query_one("#btn-export-all", Button).disabled = not has_recs

    def _render_results_table(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        for idx, rec in enumerate(self.current_recs):
            rank = f"#{idx + 1}"
            thinking = "🧠 Yes" if rec.enable_thinking else "—"
            table.add_row(
                rank,
                rec.quantization.value,
                f"{rec.context_length:,}",
                _offload_label(rec.gpu_layers),
                f"{rec.estimated_vram_gb:.2f} GB",
                _quality_badge(rec.quality_score),
                str(rec.temperature),
                thinking,
                key=str(idx),
            )

    def _render_context_tab(self, rec: ModelRecommendation) -> None:
        self.query_one("#ctx-header").update(
            f"[bold cyan]{rec.model_id}[/bold cyan]  ·  "
            f"[yellow]{rec.quantization.value}[/yellow]  ·  "
            f"[dim]Context {rec.context_length:,} tokens[/dim]"
        )
        thinking_text = (
            "[bold green]✓ Enabled  —  <reasoning> tags active[/bold green]"
            if rec.enable_thinking
            else "[dim red]✗ Disabled[/dim red]"
        )
        self.query_one("#ctx-thinking").update(thinking_text)
        prompt = rec.system_prompt or "[dim]No system prompt set[/dim]"
        self.query_one("#system-prompt-display").update(prompt)
        self.query_one("#inf-temp").update(str(rec.temperature))
        self.query_one("#inf-topp").update(str(rec.top_p))
        self.query_one("#inf-topk").update(str(rec.top_k))
        self.query_one("#inf-rep").update(str(rec.repeat_penalty))
        self.query_one("#inf-maxtok").update(f"{rec.max_tokens:,}")

    def _render_vram_tab(self, rec: ModelRecommendation) -> None:
        from vram_calculator import VRAMCalculator
        m_info = next(
            (m for m in self.local_models if m["id"] == rec.model_id), {}
        )
        params_str = m_info.get("params", "7B")
        try:
            params_b = float("".join(c for c in params_str if c.isdigit() or c == "."))
        except Exception:
            params_b = 7.0

        calc = VRAMCalculator()
        breakdown = calc.calculate(params_b, rec.quantization, rec.context_length)
        total = breakdown["total_gb"]
        vram_avail = self.hardware.gpu_vram_gb or 0

        bar_w = 30
        sections = [
            ("Weights", breakdown["weights_gb"], "#f0883e"),
            ("KV Cache", breakdown["kv_cache_gb"], "#58a6ff"),
            ("Activations", breakdown["activations_gb"], "#3fb950"),
            ("Overhead", breakdown["overhead_gb"], "#8b949e"),
        ]
        lines = ["[bold]VRAM Breakdown[/bold]\n"]
        for name, val, col in sections:
            pct = val / total if total else 0
            bar = "█" * int(pct * bar_w) + "░" * (bar_w - int(pct * bar_w))
            lines.append(
                f"[{col}]{name:<14}[/{col}]  [{col}]{bar}[/{col}]  {val:.2f} GB  ({pct*100:.0f}%)"
            )
        lines.append("")
        lines.append(
            f"[bold]Total Estimated:[/bold]  [bold yellow]{total:.2f} GB[/bold yellow]"
        )
        if vram_avail:
            pct_used = total / vram_avail
            status = (
                "[bold green]✓ Fits in VRAM[/bold green]"
                if pct_used <= 0.95
                else "[bold red]⚠ May exceed VRAM[/bold red]"
            )
            lines.append(
                f"[bold]GPU Budget:[/bold]    {total:.2f} / {vram_avail:.1f} GB  {status}"
            )
        self.query_one("#vram-detail").update("\n".join(lines))

    # ── Export Logic ──────────────────────────────────────────────────────────
    def _trigger_export(self) -> None:
        if not self.selected_rec and self.current_recs:
            self.selected_rec = self.current_recs[0]
        if not self.selected_rec:
            self.notify("No recommendation selected", severity="warning")
            return
        target = self._build_export_path(self.selected_rec)
        self.push_screen(ExportModal(self.selected_rec, target))

    def _export_all(self) -> None:
        if not self.current_recs:
            self.notify("No recommendations to export", severity="warning")
            return
        ok = 0
        for rec in self.current_recs:
            target = self._build_export_path(rec, suffix=f"_{rec.quantization.value.lower()}")
            try:
                self.perform_export(rec, target, silent=True)
                ok += 1
            except Exception:
                pass
        self.notify(
            f"Exported {ok}/{len(self.current_recs)} presets to ~/.cache/lm-studio/config-presets/",
            title="Batch Export", severity="information"
        )

    def _build_export_path(self, rec: ModelRecommendation, suffix: str = "_v3") -> Path:
        base = Path.home() / ".cache" / "lm-studio" / "config-presets"
        safe_id = rec.model_id.replace("/", "_").replace(" ", "-")
        return base / f"{safe_id}{suffix}.json"

    def perform_export(
        self,
        rec: ModelRecommendation,
        path: Path,
        silent: bool = False,
    ) -> None:
        try:
            ConfigExporter.export_preset(rec, path)
            if not silent:
                self.notify(
                    f"✅  Saved → {path.name}",
                    title="Export Success", severity="information"
                )
        except Exception as exc:
            if not silent:
                self.notify(
                    f"Export failed: {exc}",
                    title="Export Error", severity="error"
                )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LMStudioConfigApp().run()
