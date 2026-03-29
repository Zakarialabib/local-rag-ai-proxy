import os
import sys
import json
import secrets
import threading
import time
import uuid
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from flask import Flask, Response, abort, jsonify, render_template, request, session

# Setup path for parent directory imports FIRST
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger_config import setup_logger, log_api_error, error_to_dict
from engine import USE_CASE_PROFILES
from shared.ace import ACESessionStore
from shared.agent_state import AgentSessionStore
from shared.operator_state import OperatorState, OperatorStateStore
from shared.preset_service import PresetService
from shared.profile_service import ProfileService
from shared.request_service import RequestService
from shared.runtime_service import RuntimeService, clean_model_id
from shared.workspace_samples import WORKSPACE_PRESETS, WORKSPACE_TESTS
from webapp.acid_store import ACIDSessionStore
from webapp.hardware_tuner import hardware_tuner
from webapp.fluid_orchestrator import fluid_orchestrator
from webapp.embedding_cache import predictive_embedding_cache
from webapp.sse import sse_stream
from webapp.lite_dspy import lite_dspy_optimizer
from webapp.meta_agent import meta_agent

# Centralized logger
logger = setup_logger("webapp", log_file="webapp.log")

@dataclass
class TaskRecord:
    id: str
    name: str
    status: str = "queued"
    started_at: float = 0.0
    finished_at: float = 0.0
    result: Any = None
    error: str = ""

class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.RLock()

    def submit(self, name: str, fn: Callable[[], Any]) -> TaskRecord:
        task = TaskRecord(id=f"task_{uuid.uuid4().hex[:12]}", name=name, started_at=time.time())
        with self._lock:
            self._tasks[task.id] = task

        def runner():
            task.status = "running"
            try:
                task.result = fn()
                task.status = "completed"
            except Exception as exc:
                task.error = str(exc)
                task.status = "failed"
                logger.error(f"Task {name} failed", extra={"task_id": task.id, "error": str(exc)})
            finally:
                task.finished_at = time.time()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return task

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

def _default_bridge_base() -> str:
    host = os.getenv("BRIDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("BRIDGE_PORT", "8080").strip() or "8080"
    return f"http://{host}:{port}"

def _default_state() -> OperatorState:
    return OperatorState(
        bridge_base=_default_bridge_base(),
        lmstudio_base=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/"),
        selected_model=clean_model_id(os.getenv("MAIN_MODEL", "")),
        role_map={
            "main": clean_model_id(os.getenv("MAIN_MODEL", "qwen3.5-4b")),
            "reasoning": clean_model_id(os.getenv("REASONING_MODEL", "qwen3.5-4b")),
            "embed": clean_model_id(os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-4b")),
            "rerank": clean_model_id(os.getenv("RERANK_MODEL", "qwen.qwen3-reranker-4b")),
        },
    )

def create_app() -> Flask:
    root = ROOT
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("WEB_CONSOLE_SECRET", secrets.token_hex(24))

    # Instantiate stores and services
    state_store = OperatorStateStore(root / ".gui_state")
    agent_store = AgentSessionStore(root / ".gui_state" / "agent_sessions")
    ace_store = ACESessionStore(root / ".gui_state" / "ace_sessions")
    acid_store = ACIDSessionStore()
    state_lock = threading.RLock()
    state = state_store.load(_default_state())
    profile_service = ProfileService()
    preset_service = PresetService()
    tasks = TaskRegistry()

    # Start background processes
    hardware_tuner.start()
    fluid_orchestrator.start()
    predictive_embedding_cache.start()
    lite_dspy_optimizer.start()
    meta_agent.start()

    # --- Core Logic Helpers ---

    def get_state() -> OperatorState:
        with state_lock:
            return state

    def update_state(**changes: Any) -> OperatorState:
        nonlocal state
        with state_lock:
            state = state_store.update(state, **changes)
            return state

    def log_event(event_type: str, message: str, **data: Any) -> None:
        state_store.append_event(event_type, message, **data)

    def runtime_service() -> RuntimeService:
        current = get_state()
        return RuntimeService(current.bridge_base, current.lmstudio_base)

    def request_service() -> RequestService:
        current = get_state()
        return RequestService(current.bridge_base, profile_service.get_hardware_dict())

    def bridge_get(path: str) -> Dict[str, Any]:
        """Make a GET request to the bridge with error handling and logging."""
        try:
            current = get_state()
            with httpx.Client(timeout=15) as client:
                response = client.get(f"{current.bridge_base}{path}")
                response.raise_for_status()
                logger.info(f"Bridge GET {path}: {response.status_code}")
                return response.json()
        except httpx.TimeoutException as e:
            logger.error(f"Bridge GET timeout on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge request timeout: {path}") from e
        except httpx.HTTPError as e:
            logger.error(f"Bridge GET failed on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge connection error: {path}") from e
        except Exception as e:
            logger.error(f"Bridge GET unexpected error on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge request failed: {path}") from e

    def bridge_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the bridge with error handling and logging."""
        try:
            current = get_state()
            with httpx.Client(timeout=90) as client:
                response = client.post(f"{current.bridge_base}{path}", json=payload)
                response.raise_for_status()
                logger.info(f"Bridge POST {path}: {response.status_code}")
                return response.json()
        except httpx.TimeoutException as e:
            logger.error(f"Bridge POST timeout on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge request timeout: {path}") from e
        except httpx.HTTPError as e:
            logger.error(f"Bridge POST failed on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge connection error: {path}") from e
        except Exception as e:
            logger.error(f"Bridge POST unexpected error on {path}", extra=error_to_dict(e))
            raise RuntimeError(f"Bridge request failed: {path}") from e

    def ensure_profile() -> Dict[str, Any]:
        current = get_state()
        if current.profile:
            return current.profile
        models = runtime_service().list_local_models()
        selected = current.selected_model or current.role_map.get("main", "")
        profile = recompute_profile(selected, models=models, persist=True)
        return profile or {}

    def build_dashboard_payload() -> Dict[str, Any]:
        current = get_state()
        models = runtime_service().list_local_models()
        selected_model = current.selected_model or current.role_map.get("main", "")
        role_choices = {
            role: runtime_service().build_role_choices(models, role, show_all=True)
            for role in ("main", "reasoning", "embed", "rerank")
        }
        presets = [
            {"name": path.name, "path": str(path)}
            for path in preset_service.list_presets()
        ]
        inventory = runtime_service().build_inventory_rows(models)
        if not current.profile and selected_model:
            recompute_profile(selected_model, models=models, persist=True)
            current = get_state()
        
        return {
            "state": asdict(current),
            "models": inventory,
            "role_choices": role_choices,
            "presets": presets,
            "agent_sessions": agent_store.list_sessions(25),
            "ace_sessions": ace_store.list_sessions(25),
            "agent_workflows": [
                {"id": "coding_sprint", "label": "Coding Sprint"},
                {"id": "research_deep", "label": "Research Deep"},
                {"id": "debug_fix", "label": "Debug Fix"},
                {"id": "architect_review", "label": "Architect Review"},
            ],
            "workspace_tests": WORKSPACE_TESTS,
            "workspace_presets": WORKSPACE_PRESETS,
            "events": state_store.read_events(150),
            "use_cases": [{"key": key, "label": label} for key, label in USE_CASE_PROFILES.items()],
            "hardware": profile_service.get_hardware_dict(),
        }

    def recompute_profile(
        selected_model: str,
        use_case: str = "balanced",
        backend: str = "cuda",
        models: List[Dict[str, Any]] = None,
        persist: bool = False,
    ) -> Dict[str, Any]:
        current = get_state()
        models = models or runtime_service().list_local_models()
        model_entry = next((item for item in models if item.get("id") == selected_model), None)
        if not model_entry:
            return current.profile
        params_b = profile_service.estimate_params_b(model_entry)
        recs = profile_service.compute_recommendations(
            model_id=selected_model,
            params_b=params_b,
            use_case=use_case,
            backend=backend,
            flash_attention=True,
        )
        if not recs:
            return current.profile
        profile = profile_service.build_profile(recs[0], current.role_map)
        profile["use_case"] = use_case
        if persist:
            update_state(selected_model=selected_model, profile=profile)
            log_event("profile", "Profile recomputed", model=selected_model, use_case=use_case, backend=backend)
        return profile

    def increment_request_counter(name: str) -> None:
        current = get_state()
        counters = dict(current.request_counts or {})
        counters[name] = counters.get(name, 0) + 1
        update_state(request_counts=counters)

    # --- Middleware & Auth ---

    def require_local_request() -> None:
        allow_remote = os.getenv("WEB_CONSOLE_ALLOW_REMOTE", "false").lower() in {"1", "true", "yes", "on"}
        if allow_remote:
            return
        remote_addr = (request.headers.get("X-Forwarded-For", request.remote_addr or "")).split(",")[0].strip()
        if remote_addr not in {"127.0.0.1", "::1", "localhost"}:
            abort(403)

    def csrf_token() -> str:
        token = session.get("_csrf_token")
        if not token:
            token = secrets.token_hex(16)
            session["_csrf_token"] = token
        return token

    def verify_csrf() -> None:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
            if token != session.get("_csrf_token"):
                abort(400, description="Invalid CSRF token")

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _before_request() -> None:
        require_local_request()
        verify_csrf()

    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        if request.path.startswith("/api/"):
            return jsonify({"error": str(exc)}), getattr(exc, "code", 500)
        return render_template("base.html", title="Error", content=str(exc)), getattr(exc, "code", 500)

    # --- HTTP Routes ---

    @app.get("/")
    def dashboard():
        try:
            logger.info("Loading dashboard")
            payload = build_dashboard_payload()
            return render_template("dashboard.html", dashboard=payload, title="LM Studio Web Console")
        except Exception as e:
            logger.error("Dashboard render error", extra=error_to_dict(e))
            return render_template("dashboard.html", dashboard={}, title="LM Studio Web Console - Error"), 500

    @app.get("/api/state")
    def api_state():
        try:
            return jsonify(build_dashboard_payload())
        except Exception as e:
            logger.error("State fetch error", extra=error_to_dict(e))
            return jsonify({'error': 'Failed to fetch state', 'details': str(e)}), 500

    # System/Runtime Status
    @app.post("/api/runtime/refresh")
    def api_runtime_refresh():
        try:
            report = runtime_service().refresh_runtime_status()
            local_models = runtime_service().list_local_models()
            loaded_models = runtime_service().list_loaded_models(report, local_models)
            runtime_status = "Bridge reachable"
            if report.get("bridge_error"):
                runtime_status = f"Bridge issue: {report['bridge_error']}"
            elif report.get("lmstudio_error"):
                runtime_status = f"LM Studio issue: {report['lmstudio_error']}"
            update_state(
                runtime_status=runtime_status,
                loaded_models=loaded_models,
                last_error=report.get("bridge_error") or report.get("lmstudio_error") or "",
            )
            log_event("runtime", "Runtime refreshed", loaded_models=loaded_models, status=runtime_status)
            return jsonify({"ok": True, "report": report, "loaded_models": loaded_models, "runtime_status": runtime_status})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/runtime/update-config")
    def api_update_config():
        try:
            body = request.get_json(force=True) or {}
            bridge_base = str(body.get("bridge_base") or get_state().bridge_base).rstrip("/")
            lmstudio_base = str(body.get("lmstudio_base") or get_state().lmstudio_base).rstrip("/")
            update_state(bridge_base=bridge_base, lmstudio_base=lmstudio_base)
            log_event("runtime", "Runtime endpoints updated", bridge_base=bridge_base, lmstudio_base=lmstudio_base)
            return jsonify({"ok": True, "state": asdict(get_state())})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/profile/recompute")
    def api_profile_recompute():
        body = request.get_json(force=True) or {}
        selected_model = clean_model_id(body.get("selected_model") or get_state().selected_model or get_state().role_map.get("main"))
        use_case = str(body.get("use_case") or "balanced")
        backend = str(body.get("backend") or "cuda").lower()
        profile = recompute_profile(selected_model, use_case=use_case, backend=backend, persist=True)
        if not profile:
            return jsonify({"error": "Unable to compute profile"}), 400
        return jsonify({"ok": True, "profile": profile})

    @app.post("/api/roles/save")
    def api_roles_save():
        body = request.get_json(force=True) or {}
        role_map = {
            "main": clean_model_id(body.get("main") or ""),
            "reasoning": clean_model_id(body.get("reasoning") or ""),
            "embed": clean_model_id(body.get("embed") or ""),
            "rerank": clean_model_id(body.get("rerank") or ""),
        }
        runtime_service().save_role_mapping(role_map, root / ".env")
        profile = dict(get_state().profile or {})
        profile["embed_model"] = role_map.get("embed", "")
        profile["rerank_model"] = role_map.get("rerank", "")
        update_state(role_map=role_map, profile=profile)
        log_event("roles", "Role map saved", role_map=role_map)
        return jsonify({"ok": True, "role_map": role_map})

    # Model Operations
    @app.post("/api/models/load")
    def api_models_load():
        body = request.get_json(force=True) or {}
        role = str(body.get("role") or "main")
        model_id = clean_model_id(body.get("model") or get_state().role_map.get(role, ""))
        context_length = int(body.get("context_length") or get_state().profile.get("context_length") or 0)

        def work():
            log_event("task", "Model load started", role=role, model=model_id)
            result = runtime_service().load_role_model(role, model_id, context_length=context_length or None)
            api_runtime_refresh()
            log_event("task", "Model load completed", role=role, model=model_id)
            return result

        increment_request_counter("model_load")
        task = tasks.submit(f"load:{role}", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    @app.post("/api/models/unload")
    def api_models_unload():
        body = request.get_json(force=True) or {}
        model_id = clean_model_id(body.get("model") or "")

        def work():
            log_event("task", "Model unload started", model=model_id)
            result = runtime_service().unload_model(model_id)
            api_runtime_refresh()
            log_event("task", "Model unload completed", model=model_id)
            return result

        increment_request_counter("model_unload")
        task = tasks.submit("unload:model", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    # Chat & Responses
    @app.post("/api/requests/responses/preview")
    def api_responses_preview():
        body = request.get_json(force=True) or {}
        payload = request_service().build_responses_payload(body, get_state().role_map, ensure_profile())
        return jsonify(request_service().preview_payload(payload))

    @app.post("/api/requests/responses/run")
    def api_responses_run():
        body = request.get_json(force=True) or {}
        payload = request_service().build_responses_payload(body, get_state().role_map, ensure_profile())

        def work():
            log_event("task", "Responses started", model=payload.get("model"))
            try:
                result = request_service().run_responses(payload)
                update_state(last_response=result, last_error="")
                log_event("task", "Responses completed", model=payload.get("model"))
                return result
            except Exception as exc:
                update_state(last_error=str(exc))
                raise

        increment_request_counter("responses")
        task = tasks.submit("responses", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    @app.post("/api/requests/chat/run")
    def api_chat_run():
        body = request.get_json(force=True) or {}
        payload = request_service().build_chat_payload(body, get_state().role_map, ensure_profile())

        def work():
            log_event("task", "Chat started", model=payload.get("model"))
            try:
                result = request_service().run_chat(payload)
                update_state(last_chat=result, last_error="")
                log_event("task", "Chat completed", model=payload.get("model"))
                return result
            except Exception as exc:
                update_state(last_error=str(exc))
                raise

        increment_request_counter("chat")
        task = tasks.submit("chat", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    # Agent Sessions
    @app.get("/api/agent/sessions")
    def api_agent_sessions():
        try:
            try:
                remote = bridge_get("/v1/agent/sessions")
                sessions = remote.get("sessions", [])
            except Exception:
                sessions = agent_store.list_sessions(25)
            return jsonify({"sessions": sessions})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post("/api/agent/sessions")
    def api_agent_create_session():
        try:
            body = request.get_json(force=True) or {}
            workflow = body.get("workflow") or "coding_sprint"
            payload = {
                "workflow": workflow,
                "tool_budget": int(body.get("tool_budget") or 6),
                "cwd": body.get("cwd") or str(root),
                "main_model": get_state().role_map.get("main", ""),
                "reasoning_model": get_state().role_map.get("reasoning", ""),
                "embed_model": get_state().role_map.get("embed", ""),
                "rerank_model": get_state().role_map.get("rerank", ""),
            }
            result = bridge_post("/v1/agent/sessions", payload)
            session_id = result.get("session_id")
            
            # Store in ACID store too
            acid_store.create_session(session_id, "agent", workflow, meta=payload)
            acid_store.log_event(session_id, "create", {"payload": payload})
            meta_agent.emit("agent_session_created", {"session_id": session_id, "payload": payload})
            
            log_event("agent", "Agent session created", session_id=session_id, workflow=workflow)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.get("/api/agent/sessions/<session_id>/state")
    def api_agent_session_state(session_id: str):
        try:
            return jsonify(bridge_get(f"/v1/agent/sessions/{session_id}/state"))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.post("/api/agent/sessions/<session_id>/turn")
    def api_agent_turn(session_id: str):
        try:
            body = request.get_json(force=True) or {}
            def work():
                log_event("agent", "Agent turn started", session_id=session_id)
                result = bridge_post(f"/v1/agent/sessions/{session_id}/turn", body)
                log_event("agent", "Agent turn completed", session_id=session_id)
                return result
            task = tasks.submit("agent_turn", work)
            update_state(last_task_id=task.id)
            return jsonify({"ok": True, "task_id": task.id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ACE Sessions
    @app.route('/v1/ace/sessions', methods=['GET'])
    def get_ace_sessions():
        try:
            limit = int(request.args.get('limit') or 20)
            return jsonify(bridge_get(f"/v1/ace/sessions?limit={limit}"))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/v1/ace/generate', methods=['POST'])
    def generate_ace_session():
        try:
            data = request.get_json() or {}
            result = bridge_post("/v1/ace/generate", data)
            session_id = result.get("session_id")
            
            # Store in ACID store
            acid_store.create_session(session_id, "ace", data.get("workflow", "ace"), meta=data)
            acid_store.log_event(session_id, "create", {"payload": data})
            meta_agent.emit("ace_session_created", {"session_id": session_id, "data": data})
            
            log_event("ace", "ACE session generated", session_id=session_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/v1/ace/sessions/<session_id>/trace', methods=['GET'])
    def get_ace_trace(session_id):
        try:
            return jsonify(bridge_get(f"/v1/ace/sessions/{session_id}/trace"))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/v1/ace/analyze', methods=['POST'])
    def analyze_agent():
        try:
            data = request.get_json() or {}
            return jsonify(bridge_post("/v1/ace/analyze", data))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ACID Common API
    @app.get("/api/acid/sessions")
    def api_acid_sessions_list():
        try:
            with sqlite3.connect(acid_store.db_path) as conn:
                rows = conn.execute("SELECT id, type, created_at, workflow, status, meta FROM sessions ORDER BY created_at DESC LIMIT 50").fetchall()
                sessions = [dict(id=row[0], type=row[1], created_at=row[2], workflow=row[3], status=row[4], meta=json.loads(row[5] or '{}')) for row in rows]
            return jsonify({"sessions": sessions})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.get("/api/acid/sessions/<session_id>/timeline")
    def api_acid_session_timeline_list(session_id):
        try:
            return jsonify({"events": acid_store.get_session_events(session_id)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Tasks Status
    @app.get("/api/tasks/<task_id>")
    def api_task_status(task_id: str):
        task = tasks.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(asdict(task))

    # SSE Event Streams
    @app.get("/api/sse/<channel>")
    def api_sse_stream(channel):
        if channel not in {"hot", "warm", "cold"}:
            return "Invalid channel", 404
        return sse_stream(channel)

    @app.get("/events")
    def global_events():
        def generate():
            last_sent = 0
            while True:
                records = state_store.read_events(200)
                fresh = [item for item in records if int(item.get("id", 0)) > last_sent]
                if fresh:
                    for item in fresh:
                        last_sent = max(last_sent, int(item.get("id", 0)))
                        yield f"id: {item['id']}\ndata: {json.dumps(item)}\n\n"
                else:
                    yield ": ping\n\n"
                time.sleep(1)
        return Response(generate(), mimetype="text/event-stream")

    # --- TOOL HEALTH & PERFECTION ENDPOINTS ---

    @app.get("/api/tools/health")
    def api_tools_health():
        """Get tool health status from ecosystem monitor."""
        try:
            from webapp.tool_ecosystem import tool_health_monitor
            health_report = tool_health_monitor.get_health_report()
            return jsonify({
                "ok": True,
                "timestamp": health_report.get("timestamp"),
                "tools": health_report.get("tools", {}),
                "summary": health_report.get("summary", {})
            })
        except Exception as e:
            return jsonify({"error": str(e), "failed": True}), 500

    @app.get("/api/tools/perfection")
    def api_tools_perfection():
        """Get tool perfection index metrics."""
        try:
            from webapp.perfection_index import perfection_index
            metrics = perfection_index.get_index_report()
            return jsonify({
                "ok": True,
                "timestamp": metrics.get("timestamp"),
                "tool_metrics": metrics.get("tool_metrics", {}),
                "global_index": metrics.get("global_index", 0.0),
                "trends": metrics.get("trends", {})
            })
        except Exception as e:
            return jsonify({"error": str(e), "failed": True}), 500

    @app.post("/api/tools/perfection/reset")
    def api_tools_perfection_reset():
        """Reset perfection index metrics."""
        try:
            from webapp.perfection_index import perfection_index
            perfection_index.reset_metrics()
            return jsonify({"ok": True, "message": "Perfection index reset"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/tools/health/timeline")
    def api_tools_health_timeline():
        """Get health metric timeline (last 100 records)."""
        try:
            from webapp.tool_ecosystem import tool_health_monitor
            timeline = tool_health_monitor.get_health_timeline(limit=100)
            return jsonify({"ok": True, "timeline": timeline})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- ANALYTICS & REMEDIATION ENDPOINTS ---

    @app.get("/api/tools/analytics/<tool_name>")
    def api_tool_analytics(tool_name: str):
        """Get analytics and history for a specific tool."""
        try:
            from webapp.tool_analytics import analytics_store
            history = analytics_store.get_tool_history(tool_name, limit=100)
            return jsonify({"ok": True, "tool": tool_name, "history": history})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/tools/analytics/detect-anomalies")
    def api_detect_anomalies():
        """Run anomaly detection on all tools."""
        try:
            from webapp.tool_analytics import anomaly_detector
            body = request.get_json() or {}
            tool_name = body.get("tool_name")
            
            if tool_name:
                # Single tool
                anomalies = anomaly_detector.detect_anomalies(tool_name)
            else:
                # All tools
                anomalies = []
                for name in anomaly_detector.history.keys():
                    anomalies.extend(anomaly_detector.detect_anomalies(name))
            
            return jsonify({
                "ok": True,
                "anomalies_detected": len(anomalies),
                "anomalies": [a.to_dict() for a in anomalies]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/tools/remediation/pending")
    def api_remediation_pending():
        """Get pending remediation actions."""
        try:
            from webapp.tool_analytics import remediation_engine
            limit = int(request.args.get("limit", 50))
            actions = remediation_engine.get_pending_actions(limit=limit)
            return jsonify({"ok": True, "pending_actions": actions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- ADVANCED TEST ENDPOINTS ---
    
    @app.post("/api/retrieval/test")
    def api_retrieval_test():
        try:
            body = request.get_json(force=True) or {}
            result = bridge_post("/api/v1/retrieve", body)
            log_event("retrieval", "Retrieval test run", result=result)
            return jsonify({"ok": True, "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/rerank/test")
    def api_rerank_test():
        try:
            body = request.get_json(force=True) or {}
            result = bridge_post("/api/v1/rerank", body)
            log_event("rerank", "Rerank test run", result=result)
            return jsonify({"ok": True, "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/embeddings/test")
    def api_embeddings_test():
        try:
            body = request.get_json(force=True) or {}
            result = bridge_post("/v1/embeddings", body)
            log_event("embeddings", "Embeddings test run", result=result)
            return jsonify({"ok": True, "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/benchmark/test")
    def api_benchmark_test():
        try:
            body = request.get_json(force=True) or {}
            inputs = body.get("inputs") or ["Hello world!"]
            results = []
            for inp in inputs:
                payload = {
                    "model": get_state().role_map.get("main", ""),
                    "messages": [{"role": "user", "content": inp}],
                    "stream": False
                }
                results.append(bridge_post("/v1/chat/completions", payload))
            log_event("benchmark", "Benchmark run", count=len(inputs))
            return jsonify({"ok": True, "results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- ALERT & REPORTING ENDPOINTS (Phase 3-5) ---

    @app.get("/api/alerts/recent")
    def api_alerts_recent():
        """Get recent alerts."""
        try:
            from webapp.alerting_system import alert_manager
            limit = int(request.args.get("limit", 50))
            acknowledged = request.args.get("acknowledged")
            ack_filter = None
            if acknowledged in {"true", "1"}:
                ack_filter = True
            elif acknowledged in {"false", "0"}:
                ack_filter = False
            
            alerts = alert_manager.store.get_recent_alerts(limit=limit, acknowledged=ack_filter)
            return jsonify({"ok": True, "alerts": alerts})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/alerts/summary")
    def api_alerts_summary():
        """Get alert summary statistics."""
        try:
            from webapp.alerting_system import alert_manager
            summary = alert_manager.get_alerts_summary()
            return jsonify({"ok": True, "summary": summary})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/alerts/<alert_id>/acknowledge")
    def api_acknowledge_alert(alert_id: str):
        """Acknowledge an alert."""
        try:
            from webapp.alerting_system import alert_manager
            body = request.get_json() or {}
            acknowledged_by = body.get("acknowledged_by", "user")
            alert_manager.store.acknowledge_alert(alert_id, acknowledged_by)
            return jsonify({"ok": True, "message": "Alert acknowledged"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/alerts/tool/<tool_name>")
    def api_tool_alerts(tool_name: str):
        """Get alerts for a specific tool."""
        try:
            from webapp.alerting_system import alert_manager
            limit = int(request.args.get("limit", 50))
            alerts = alert_manager.store.get_tool_alerts(tool_name, limit=limit)
            return jsonify({"ok": True, "tool": tool_name, "alerts": alerts})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/reports/daily")
    def api_report_daily():
        """Generate daily report for a tool."""
        try:
            from webapp.reporting_system import report_generator
            tool_name = request.args.get("tool", "")
            if not tool_name:
                return jsonify({"error": "tool parameter required"}), 400
            
            report = report_generator.generate_daily_report(tool_name)
            return jsonify({"ok": True, "report": report})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/reports/weekly")
    def api_report_weekly():
        """Generate weekly report for a tool."""
        try:
            from webapp.reporting_system import report_generator
            tool_name = request.args.get("tool", "")
            if not tool_name:
                return jsonify({"error": "tool parameter required"}), 400
            
            report = report_generator.generate_weekly_report(tool_name)
            return jsonify({"ok": True, "report": report})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/reports/monthly")
    def api_report_monthly():
        """Generate monthly report for a tool."""
        try:
            from webapp.reporting_system import report_generator
            tool_name = request.args.get("tool", "")
            if not tool_name:
                return jsonify({"error": "tool parameter required"}), 400
            
            report = report_generator.generate_monthly_report(tool_name)
            return jsonify({"ok": True, "report": report})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/reports/regressions")
    def api_report_regressions():
        """Get detected regressions."""
        try:
            from webapp.reporting_system import report_store
            tool_name = request.args.get("tool")
            limit = int(request.args.get("limit", 50))
            
            regressions = report_store.get_regressions(tool_name=tool_name, limit=limit)
            return jsonify({"ok": True, "regressions": regressions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/reports/trends")
    def api_report_trends():
        """Get trend history for a tool."""
        try:
            from webapp.reporting_system import report_store
            tool_name = request.args.get("tool", "")
            if not tool_name:
                return jsonify({"error": "tool parameter required"}), 400
            
            days = int(request.args.get("days", 30))
            trends = report_store.get_trend_history(tool_name, days=days)
            return jsonify({"ok": True, "tool": tool_name, "days": days, "trends": trends})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- SSE STREAMING ENDPOINTS ---

    @app.get("/api/sse/alerts")
    def api_sse_alerts():
        """Stream real-time alerts via SSE."""
        from webapp.sse import sse_stream
        return sse_stream("alerts")

    @app.get("/api/sse/health")
    def api_sse_health():
        """Stream real-time health updates via SSE."""
        from webapp.sse import sse_stream
        return sse_stream("health")

    @app.get("/api/sse/metrics")
    def api_sse_metrics():
        """Stream real-time metrics updates via SSE."""
        from webapp.sse import sse_stream
        return sse_stream("cold")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
