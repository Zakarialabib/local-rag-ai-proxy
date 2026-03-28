import json
import os
import secrets
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
from typing import Dict, Any, Optional
import httpx
import requests

# Setup path for parent directory imports FIRST
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger_config import setup_logger, log_api_error, error_to_dict

# Setup centralized logger
logger = setup_logger("webapp", log_file="webapp.log")

from flask import Flask, Response, abort, jsonify, render_template, request, session


from engine import USE_CASE_PROFILES
from shared.ace import ACESessionStore
from shared.agent_state import AgentSessionStore
from shared.operator_state import OperatorState, OperatorStateStore
from shared.preset_service import PresetService
from shared.profile_service import ProfileService
from shared.request_service import RequestService
from shared.runtime_service import RuntimeService, clean_model_id
from shared.workspace_samples import WORKSPACE_PRESETS, WORKSPACE_TESTS


def _default_bridge_base() -> str:
    host = os.getenv("BRIDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("BRIDGE_PORT", "8080").strip() or "8080"
    return f"http://{host}:{port}"


def _default_state() -> OperatorState:
    return OperatorState(
        bridge_base=_default_bridge_base(),
        lmstudio_base=os.getenv("LMSTUDIO_BASE_URL", "http://192.168.1.12:1234").rstrip("/"),
        selected_model=clean_model_id(os.getenv("MAIN_MODEL", "")),
        role_map={
            "main": clean_model_id(os.getenv("MAIN_MODEL", "qwen3.5-4b")),
            "reasoning": clean_model_id(os.getenv("REASONING_MODEL", "qwen3.5-4b")),
            "embed": clean_model_id(os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-4b")),
            "rerank": clean_model_id(os.getenv("RERANK_MODEL", "qwen.qwen3-reranker-4b")),
        },
    )


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
            finally:
                task.finished_at = time.time()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)


def create_app() -> Flask:
    root = ROOT
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.getenv("WEB_CONSOLE_SECRET", secrets.token_hex(24))

    state_store = OperatorStateStore(root / ".gui_state")
    agent_store = AgentSessionStore(root / ".gui_state" / "agent_sessions")
    ace_store = ACESessionStore(root / ".gui_state" / "ace_sessions")
    state_lock = threading.RLock()
    state = state_store.load(_default_state())
    profile_service = ProfileService()
    preset_service = PresetService()
    tasks = TaskRegistry()

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
            role: runtime_service().build_role_choices(models, role, show_all=False)
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
        models: List[Dict[str, Any]] | None = None,
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
            logger.debug("Fetching dashboard state")
            return jsonify(build_dashboard_payload())
        except Exception as e:
            logger.error("State fetch error", extra=error_to_dict(e))
            return jsonify({'error': 'Failed to fetch state', 'details': str(e)}), 500

    @app.get("/api/agent/sessions")
    def api_agent_sessions():
        try:
            logger.info("Fetching agent sessions")
            try:
                remote = bridge_get("/v1/agent/sessions")
                sessions = remote.get("sessions", [])
                logger.info(f"Retrieved {len(sessions)} agent sessions from bridge")
            except Exception as e:
                logger.warning(f"Failed to fetch from bridge, using local store", extra=error_to_dict(e))
                sessions = agent_store.list_sessions(25)
            return jsonify({"sessions": sessions})
        except Exception as e:
            logger.error("Agent sessions fetch error", extra=error_to_dict(e))
            return jsonify({'error': 'Failed to fetch sessions', 'details': str(e)}), 500

    @app.route('/v1/ace/sessions', methods=['GET'])
    def get_ace_sessions():
        """List all ACE sessions."""
        try:
            limit = int(request.args.get('limit') or 20)
            logger.info(f"Fetching ACE sessions with limit={limit}")
            result = bridge_get(f"/v1/ace/sessions?limit={limit}")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error fetching ACE sessions", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except ValueError as e:
            logger.warning(f"Invalid limit parameter for ACE sessions", extra=error_to_dict(e))
            return jsonify({'error': 'Invalid limit parameter'}), 400
        except Exception as e:
            logger.error(f"Unexpected error fetching ACE sessions", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.route('/v1/ace/generate', methods=['POST'])
    def generate_ace_session():
        """Create new ACE session with intelligent prefilling."""
        try:
            data = request.get_json() or {}
            logger.info(f"Generating ACE session with data keys: {list(data.keys())}")
            result = bridge_post("/v1/ace/generate", data)
            session_id = result.get("session_id")
            log_event("ace", "ACE session generated", session_id=session_id)
            logger.info(f"ACE session generated successfully: {session_id}")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error generating ACE session", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error generating ACE session", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.route('/v1/ace/sessions/<session_id>/trace', methods=['GET'])
    def get_ace_trace(session_id):
        """Get ACE session trace."""
        try:
            logger.info(f"Fetching trace for ACE session: {session_id}")
            result = bridge_get(f"/v1/ace/sessions/{session_id}/trace")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error fetching ACE trace", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error fetching ACE trace", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.route('/v1/ace/sessions/<session_id>', methods=['DELETE'])
    def delete_ace_session(session_id):
        """Delete ACE session."""
        try:
            logger.info(f"Deleting ACE session: {session_id}")
            bridge_post(f"/v1/ace/sessions/{session_id}/delete", {})
            log_event("ace", "ACE session deleted", session_id=session_id)
            logger.info(f"ACE session deleted successfully: {session_id}")
            return jsonify({'success': True})
        except RuntimeError as e:
            logger.error(f"Bridge error deleting ACE session", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error deleting ACE session", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.route('/v1/ace/sessions/<session_id>/checkpoint', methods=['POST'])
    def restore_ace_checkpoint(session_id):
        """Restore ACE session checkpoint."""
        try:
            data = request.get_json() or {}
            logger.info(f"Restoring checkpoint for ACE session: {session_id}")
            result = bridge_post(f"/v1/ace/sessions/{session_id}/checkpoint", data)
            log_event("ace", "ACE checkpoint restored", session_id=session_id)
            logger.info(f"ACE checkpoint restored successfully: {session_id}")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error restoring ACE checkpoint", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error restoring ACE checkpoint", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.route('/v1/ace/analyze', methods=['POST'])
    def analyze_agent():
        """Analyze system prompt for agent profile."""
        try:
            data = request.get_json() or {}
            logger.info(f"Analyzing agent with data keys: {list(data.keys())}")
            result = bridge_post("/v1/ace/analyze", data)
            logger.info(f"Agent analysis completed")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error analyzing agent", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error analyzing agent", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.post("/api/ace/select-option")
    def api_ace_select_option():
        try:
            body = request.get_json(force=True) or {}
            logger.info(f"Recording ACE selection for session: {body.get('session_id')}")
            result = bridge_post("/v1/ace/select-option", body)
            log_event("ace", "ACE selection recorded", session_id=body.get("session_id"), option_id=body.get("option_id"))
            logger.info(f"ACE selection recorded successfully")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error recording ACE selection", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error recording ACE selection", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.post("/api/agent/sessions")
    def api_agent_create_session():
        try:
            body = request.get_json(force=True) or {}
            workflow = body.get("workflow") or "coding_sprint"
            logger.info(f"Creating agent session with workflow: {workflow}")
            
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
            log_event("agent", "Agent session created", session_id=session_id, workflow=workflow)
            logger.info(f"Agent session created successfully: {session_id}")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error creating agent session", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except ValueError as e:
            logger.warning(f"Invalid parameter for agent session", extra=error_to_dict(e))
            return jsonify({'error': 'Invalid parameters', 'details': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error creating agent session", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.get("/api/agent/sessions/<session_id>/state")
    def api_agent_session_state(session_id: str):
        try:
            logger.debug(f"Fetching state for agent session: {session_id}")
            return jsonify(bridge_get(f"/v1/agent/sessions/{session_id}/state"))
        except RuntimeError as e:
            logger.error(f"Bridge error fetching agent session state", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error fetching agent session state", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.post("/api/agent/sessions/<session_id>/turn")
    def api_agent_turn(session_id: str):
        try:
            body = request.get_json(force=True) or {}
            logger.info(f"Starting agent turn for session: {session_id}")

            def work():
                try:
                    log_event("agent", "Agent turn started", session_id=session_id)
                    result = bridge_post(f"/v1/agent/sessions/{session_id}/turn", body)
                    log_event("agent", "Agent turn completed", session_id=session_id, next_agent=result.get("state_update", {}).get("next_agent"))
                    logger.info(f"Agent turn completed for session: {session_id}")
                    return result
                except Exception as e:
                    logger.error(f"Error in agent turn task", extra=error_to_dict(e))
                    raise

            increment_request_counter("agent_turn")
            task = tasks.submit("agent_turn", work)
            update_state(last_task_id=task.id)
            logger.info(f"Agent turn task submitted: {task.id}")
            return jsonify({"ok": True, "task_id": task.id})
        except Exception as e:
            logger.error(f"Error submitting agent turn", extra=error_to_dict(e))
            return jsonify({'error': 'Failed to submit agent turn', 'details': str(e)}), 500

    @app.post("/api/agent/sessions/<session_id>/branch")
    def api_agent_branch(session_id: str):
        try:
            body = request.get_json(force=True) or {}
            logger.info(f"Branching agent session: {session_id}")
            result = bridge_post(f"/v1/agent/sessions/{session_id}/branch", body)
            branch_id = result.get("session_id")
            log_event("agent", "Agent session branched", session_id=session_id, branch_session_id=branch_id)
            logger.info(f"Agent session branched successfully: {branch_id}")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error branching agent session", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error branching agent session", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.post("/api/agent/sessions/<session_id>/restore")
    def api_agent_restore(session_id: str):
        try:
            body = request.get_json(force=True) or {}
            checkpoint_id = str(body.get("checkpoint_id") or "")
            logger.info(f"Restoring agent session checkpoint: {session_id}/{checkpoint_id}")
            result = bridge_post(f"/v1/agent/sessions/{session_id}/checkpoint/{checkpoint_id}/restore", {})
            log_event("agent", "Agent checkpoint restored", session_id=session_id, checkpoint_id=checkpoint_id)
            logger.info(f"Agent checkpoint restored successfully")
            return jsonify(result)
        except RuntimeError as e:
            logger.error(f"Bridge error restoring agent checkpoint", extra=error_to_dict(e))
            return jsonify({'error': 'Could not connect to bridge', 'details': str(e)}), 502
        except Exception as e:
            logger.error(f"Unexpected error restoring agent checkpoint", extra=error_to_dict(e))
            return jsonify({'error': 'Server error', 'details': str(e)}), 500

    @app.get("/api/agent/sessions/<session_id>/events")
    def api_agent_events(session_id: str):
        def generate():
            last_sent = 0
            while True:
                records = agent_store.read_events(session_id, 200)
                fresh = [item for item in records if int(item.get("id", 0)) > last_sent]
                if fresh:
                    for item in fresh:
                        last_sent = max(last_sent, int(item.get("id", 0)))
                        yield f"id: {item['id']}\ndata: {json.dumps(item)}\n\n"
                else:
                    yield ": ping\n\n"
                time.sleep(1)

        return Response(generate(), mimetype="text/event-stream")

    @app.post("/api/runtime/update-config")
    def api_update_config():
        try:
            body = request.get_json(force=True) or {}
            bridge_base = str(body.get("bridge_base") or get_state().bridge_base).rstrip("/")
            lmstudio_base = str(body.get("lmstudio_base") or get_state().lmstudio_base).rstrip("/")
            logger.info(f"Updating runtime config: bridge={bridge_base}, lmstudio={lmstudio_base}")
            update_state(bridge_base=bridge_base, lmstudio_base=lmstudio_base)
            log_event("runtime", "Runtime endpoints updated", bridge_base=bridge_base, lmstudio_base=lmstudio_base)
            logger.info(f"Runtime config updated successfully")
            return jsonify({"ok": True, "state": asdict(get_state())})
        except Exception as e:
            logger.error(f"Error updating runtime config", extra=error_to_dict(e))
            return jsonify({'error': 'Failed to update config', 'details': str(e)}), 500

    @app.post("/api/runtime/refresh")
    def api_runtime_refresh():
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

    @app.post("/api/profile/recompute")
    def api_profile_recompute():
        body = request.get_json(force=True) or {}
        selected_model = clean_model_id(body.get("selected_model") or get_state().selected_model or get_state().role_map.get("main"))
        use_case = str(body.get("use_case") or "balanced")
        backend = str(body.get("backend") or "cuda").lower()
        profile = recompute_profile(selected_model, use_case=use_case, backend=backend, persist=True)
        if not profile:
            return jsonify({"error": "Unable to compute profile for selected model"}), 400
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

    @app.post("/api/models/load")
    def api_models_load():
        body = request.get_json(force=True) or {}
        role = str(body.get("role") or "main")
        model_id = clean_model_id(body.get("model") or get_state().role_map.get(role, ""))
        context_length = int(body.get("context_length") or get_state().profile.get("context_length") or 0)

        def work():
            log_event("task", "Model load started", role=role, model=model_id)
            result = runtime_service().load_role_model(role, model_id, context_length=context_length or None)
            report = runtime_service().refresh_runtime_status()
            local_models = runtime_service().list_local_models()
            loaded = runtime_service().list_loaded_models(report, local_models)
            update_state(loaded_models=loaded, runtime_status="Bridge reachable", last_error="")
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
            report = runtime_service().refresh_runtime_status()
            local_models = runtime_service().list_local_models()
            loaded = runtime_service().list_loaded_models(report, local_models)
            update_state(loaded_models=loaded, runtime_status="Bridge reachable", last_error="")
            log_event("task", "Model unload completed", model=model_id)
            return result

        increment_request_counter("model_unload")
        task = tasks.submit("unload:model", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    @app.post("/api/requests/responses/preview")
    def api_responses_preview():
        body = request.get_json(force=True) or {}
        payload = request_service().build_responses_payload(body, get_state().role_map, ensure_profile())
        return jsonify(request_service().preview_payload(payload))

    @app.post("/api/workspace/autotune")
    def api_workspace_autotune():
        current = get_state()
        profile = ensure_profile()
        model_id = clean_model_id(
            (request.get_json(force=True) or {}).get("model")
            or current.selected_model
            or current.role_map.get("main", "")
        )
        use_case = str(profile.get("use_case") or "balanced")
        recommendation = request_service().recommend_workspace(
            model_id=model_id,
            use_case=use_case,
            profile=profile,
            role_map=current.role_map,
            loaded_models=current.loaded_models,
        )
        log_event("workspace", "Workspace auto-tuned", model=model_id, use_case=use_case, mode=recommendation["responses"]["mode"])
        return jsonify({"ok": True, "workspace": recommendation})

    @app.post("/api/requests/responses/run")
    def api_responses_run():
        body = request.get_json(force=True) or {}
        payload = request_service().build_responses_payload(body, get_state().role_map, ensure_profile())

        def work():
            log_event("task", "Responses request started", model=payload.get("model"))
            try:
                result = request_service().run_responses(payload)
                update_state(last_response=result, last_error="", runtime_status=get_state().runtime_status)
                log_event("task", "Responses request completed", model=payload.get("model"))
                return result
            except Exception as exc:
                update_state(last_error=str(exc), runtime_status=f"Bridge issue: {exc}")
                log_event("task", "Responses request failed", model=payload.get("model"), error=str(exc))
                raise

        increment_request_counter("responses")
        task = tasks.submit("responses", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    @app.post("/api/requests/chat/preview")
    def api_chat_preview():
        body = request.get_json(force=True) or {}
        payload = request_service().build_chat_payload(body, get_state().role_map, ensure_profile())
        return jsonify(request_service().preview_payload(payload))

    @app.post("/api/requests/chat/run")
    def api_chat_run():
        body = request.get_json(force=True) or {}
        payload = request_service().build_chat_payload(body, get_state().role_map, ensure_profile())

        def work():
            log_event("task", "Chat request started", model=payload.get("model"), stream=payload.get("stream", False))
            try:
                result = request_service().run_chat(payload)
                update_state(last_chat=result, last_error="", runtime_status=get_state().runtime_status)
                log_event("task", "Chat request completed", model=payload.get("model"), stream=payload.get("stream", False))
                return result
            except Exception as exc:
                update_state(last_error=str(exc), runtime_status=f"Bridge issue: {exc}")
                log_event("task", "Chat request failed", model=payload.get("model"), stream=payload.get("stream", False), error=str(exc))
                raise

        increment_request_counter("chat")
        task = tasks.submit("chat", work)
        update_state(last_task_id=task.id)
        return jsonify({"ok": True, "task_id": task.id})

    @app.post("/api/presets/apply")
    def api_preset_apply():
        body = request.get_json(force=True) or {}
        preset_path = Path(str(body.get("path") or ""))
        if not preset_path.exists():
            return jsonify({"error": "Preset not found"}), 404
        preset = preset_service.read_preset(preset_path)
        current = get_state()
        profile = preset_service.apply_to_profile(
            preset,
            current_profile=current.profile or ensure_profile(),
            role_map=current.role_map,
            selected_model=current.selected_model or current.role_map.get("main", ""),
        )
        update_state(profile=profile)
        log_event("preset", "Preset applied", path=str(preset_path), model=profile.get("model_id"))
        return jsonify({"ok": True, "profile": profile})

    @app.post("/api/presets/export")
    def api_preset_export():
        body = request.get_json(force=True) or {}
        current = get_state()
        selected_model = clean_model_id(body.get("selected_model") or current.selected_model or current.role_map.get("main"))
        use_case = str(body.get("use_case") or current.profile.get("use_case") or "balanced")
        backend = str(body.get("backend") or current.profile.get("backend") or "cuda")
        models = runtime_service().list_local_models()
        model_entry = next((item for item in models if item.get("id") == selected_model), None)
        if not model_entry:
            return jsonify({"error": "Selected model is not available"}), 400
        params_b = profile_service.estimate_params_b(model_entry)
        recs = profile_service.compute_recommendations(selected_model, params_b, use_case=use_case, backend=backend)
        if not recs:
            return jsonify({"error": "No recommendation available for preset export"}), 400
        rec = recs[0]
        profile = dict(current.profile or ensure_profile())
        rec.temperature = profile.get("temperature", rec.temperature)
        rec.top_p = profile.get("top_p", rec.top_p)
        rec.top_k = profile.get("top_k", rec.top_k)
        rec.repeat_penalty = profile.get("repeat_penalty", rec.repeat_penalty)
        rec.max_tokens = profile.get("max_tokens", rec.max_tokens)
        rec.context_length = profile.get("context_length", rec.context_length)
        rec.system_prompt = profile.get("system_prompt", rec.system_prompt)
        target_dir = Path.home() / ".cache" / "lm-studio" / "config-presets"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{selected_model.replace('/', '_')}_agent_console_web.preset.json"
        exported = preset_service.export_preset(rec, target_path, profile, profile_service.get_hardware_dict())
        log_event("preset", "Preset exported", path=str(exported), model=selected_model)
        return jsonify({"ok": True, "path": str(exported)})

    @app.get("/api/tasks/<task_id>")
    def api_task_status(task_id: str):
        task = tasks.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        payload = {
            "id": task.id,
            "name": task.name,
            "status": task.status,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "result": task.result,
            "error": task.error,
        }
        if task.status == "failed":
            update_state(last_error=task.error)
        return jsonify(payload)

    @app.get("/events")
    def events():
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



    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("WEB_CONSOLE_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_CONSOLE_PORT", "8090"))
    app.run(host=host, port=port, debug=False, threaded=True)
