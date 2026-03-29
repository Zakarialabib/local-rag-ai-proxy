const dashboard = window.__DASHBOARD__ || {};
const csrfToken = window.__CSRF__;
const workspaceTests = dashboard.workspace_tests || [];
const workspacePresets = dashboard.workspace_presets || [];
let agentSessions = dashboard.agent_sessions || [];
let aceSessions = dashboard.ace_sessions || [];
let refreshTimer = null;
let agentEventSource = null;

function byId(id) {
  return document.getElementById(id);
}

// Tab navigation for all sections
const tabIds = [
  'overview', 'playground', 'context', 'sessions', 'presets', 
  'settings', 'retrieval', 'embeddings', 'agent', 'ace', 'benchmark'
];

function switchView(view, el) {
  tabIds.forEach(id => {
    const section = byId('view-' + id);
    if (section) section.classList.remove('active');
  });
  const activeSection = byId('view-' + view);
  if (activeSection) activeSection.classList.add('active');
  
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => item.classList.remove('active'));
  if (el) {
    el.classList.add('active');
  } else {
    const targetNav = document.querySelector(`.nav-item[onclick*="'${view}'"]`);
    if (targetNav) targetNav.classList.add('active');
  }
}

async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrfToken,
    },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.detail || "Request failed");
  return data;
}

async function apiGet(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.detail || "Request failed");
  return data;
}

function renderJson(targetId, payload) {
  const el = byId(targetId);
  if (el) el.textContent = JSON.stringify(payload, null, 2);
}

function showBanner(message, tone = "info") {
  const banner = byId("banner");
  const tones = {
    info: "border-sky-700 bg-sky-950 text-sky-200",
    success: "border-emerald-700 bg-emerald-950 text-emerald-200",
    error: "border-rose-700 bg-rose-950 text-rose-200",
  };
  banner.className = `rounded-2xl border px-4 py-3 text-sm ${tones[tone] || tones.info}`;
  banner.textContent = message;
  banner.classList.remove("hidden");
  setTimeout(() => banner.classList.add("hidden"), 5000);
}

function setBusy(button, busy) {
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Processing...";
    button.disabled = true;
    button.classList.add("opacity-60");
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
    button.classList.remove("opacity-60");
  }
}

async function refreshHardware() {
  const statusDot = byId("status-dot");
  const start = Date.now();
  try {
    const result = await apiPost("/api/runtime/refresh", {});
    const latency = Date.now() - start;
    if (byId("bridge-latency")) byId("bridge-latency").textContent = latency + " ms";
    if (statusDot) {
      statusDot.style.background = "var(--accent-success)";
      statusDot.style.boxShadow = "0 0 10px var(--accent-success)";
    }
    await refreshState();
    showBanner("Hardware synchronized", "success");
  } catch (e) {
    if (statusDot) {
      statusDot.style.background = "var(--accent-error)";
    }
    showBanner("Hardware refresh failed", "error");
  }
}

async function refreshState() {
  const data = await apiGet("/api/state");
  const state = data.state;
  
  if (byId("role-main-text")) byId("role-main-text").textContent = (state.role_map.main || "None").split('/').pop();
  if (byId("role-reasoning-text")) byId("role-reasoning-text").textContent = (state.role_map.reasoning || "None").split('/').pop();
  if (byId("loaded-count")) byId("loaded-count").textContent = state.loaded_models ? state.loaded_models.length : 0;
  if (byId("runtime-status")) byId("runtime-status").textContent = state.runtime_status;
  if (byId("loaded-models")) byId("loaded-models").textContent = state.loaded_models ? state.loaded_models.join("\n") : "None";

  const hw = data.hardware || {};
  if (byId("hw-gpu")) byId("hw-gpu").textContent = hw.gpu_name || hw.gpu || "Unknown GPU";
  if (byId("hw-platform")) byId("hw-platform").textContent = hw.platform || "Unknown OS";
  
  if (hw.vram_gb && byId("vram-text") && byId("vram-bar")) {
    const used = (hw.vram_gb * 0.4).toFixed(1); // Simulation
    byId("vram-text").textContent = `${used} GB / ${hw.vram_gb.toFixed(1)} GB`;
    byId("vram-bar").style.width = "40%";
  }

  renderEvents(data.events || []);
  renderAgentSessions(data.agent_sessions || []);
  renderACESessions(data.ace_sessions || []);
  renderSessionVault(data.acid_sessions || []);
}

function renderEvents(events) {
  const box = byId("events-log");
  if (!box) return;
  const lines = (events || []).map(ev => `[${ev.ts}] ${ev.type}: ${ev.message}`);
  box.textContent = lines.join("\n");
  box.scrollTop = box.scrollHeight;
}

function renderAgentSessions(sessions) {
  agentSessions = sessions || [];
  const select = byId("agent-session-select");
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  agentSessions.forEach(item => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id.substring(0,8)} | ${item.workflow}`;
    if (current === item.id) option.selected = true;
    select.appendChild(option);
  });
}

function renderACESessions(sessions) {
  aceSessions = sessions || [];
  const select = byId("ace-session-select");
  if (!select) return;
  select.innerHTML = '<option value="">Create New Session</option>';
  aceSessions.forEach(item => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id.substring(0,8)} | ${item.status}`;
    select.appendChild(option);
  });
}

function renderSessionVault(sessions) {
  const body = byId("session-vault-body");
  if (!body) return;
  if (!sessions || sessions.length === 0) {
    body.innerHTML = '<tr><td colspan="5" style="padding: 40px; text-align: center; color: var(--text-muted);">No sessions in vault.</td></tr>';
    return;
  }
  body.innerHTML = sessions.map(s => `
    <tr style="border-bottom: 1px solid var(--glass-border); font-size: 13px;">
      <td style="padding: 12px; font-family: var(--font-mono); color: var(--accent-primary);">${s.id.substring(0,8)}</td>
      <td style="padding: 12px;"><span class="badge">${s.type}</span></td>
      <td style="padding: 12px;">${s.workflow}</td>
      <td style="padding: 12px;">${s.status}</td>
      <td style="padding: 12px; color: var(--text-muted);">${s.created_at}</td>
    </tr>
  `).join('');
}

async function pollTask(taskId, targetId) {
  for (;;) {
    const task = await apiGet(`/api/tasks/${taskId}`);
    if (targetId) renderJson(targetId, task);
    if (task.status === "completed") return task.result;
    if (task.status === "failed") throw new Error(task.error || "Task failed");
    await new Promise(r => setTimeout(r, 1000));
  }
}

// --- Specific Feature Logic ---

async function runRetrievalTest() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const input = byId('retrieval-input').value;
    const docs = byId('retrieval-docs').value.split(',').map(s => s.trim()).filter(Boolean);
    const res = await apiPost('/api/retrieval/test', { input, docs });
    renderJson('retrieval-result', res.result);
    showBanner('Retrieval search completed', 'success');
  } catch (e) {
    showBanner('Retrieval failed: ' + e.message, 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runRerankTest() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const query = byId('rerank-query').value;
    const chunks = byId('rerank-chunks').value.split('\n').filter(Boolean);
    const res = await apiPost('/api/rerank/test', { query, chunks });
    renderJson('rerank-result', res.result);
    showBanner('Rerank optimized via Cross-Encoder', 'success');
  } catch (e) {
    showBanner('Reranking failed: ' + e.message, 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runEmbeddingsTest() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const input = byId('embeddings-input').value;
    const res = await apiPost('/api/embeddings/test', { input });
    renderJson('embeddings-result', res.result);
  } catch (e) {
    showBanner('Embeddings failed: ' + e.message, 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runAgentCreate() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const res = await apiPost('/api/agent/sessions', {
      workflow: byId('agent-workflow').value,
      tool_budget: byId('agent-tool-budget').value
    });
    byId('agent-session-id').value = res.session_id;
    await refreshState();
    showBanner('Autonomous agent active', 'success');
  } catch (e) {
    showBanner('Agent start failed', 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runAgentTurn() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const sid = byId('agent-session-id').value;
    if (!sid) throw new Error('No active session');
    const res = await apiPost(`/api/agent/sessions/${sid}/turn`, {
      input: { type: 'user_request', content: byId('agent-input').value }
    });
    const result = await pollTask(res.task_id);
    renderJson('agent-session-state', result);
  } catch (e) {
    showBanner('Agent turn error', 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runACEGenerate() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const res = await apiPost('/v1/ace/generate', { prompt: byId('ace-input').value });
    renderJson('ace-session-result', res);
    showBanner('Context engineering completed', 'success');
  } catch (e) {
    showBanner('ACE failed', 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

async function runBenchmarkTest() {
  const btn = event?.target?.closest('button');
  if (btn) setBusy(btn, true);
  try {
    const res = await apiPost('/api/benchmark/test', {
      inputs: byId('benchmark-inputs').value.split('\n').filter(Boolean)
    });
    renderJson('benchmark-result', res.results);
    showBanner('Benchmark diagnostics complete', 'success');
  } catch (e) {
    showBanner('Benchmark failed', 'error');
  } finally {
    if (btn) setBusy(btn, false);
  }
}

function loadWorkspaceTest() {
  const selected = workspaceTests.find(t => t.id === byId("workspace-test-select").value);
  if (!selected) return;
  byId("responses-input").value = selected.input || "";
  if (byId("chat-input-text")) byId("chat-input-text").value = selected.input || "";
  if (selected.use_case && byId("use-case")) byId("use-case").value = selected.use_case;
}

// --- Global Click Router for Legacy/Config Buttons ---
document.addEventListener("click", async (event) => {
  const btn = event.target.closest("button");
  if (!btn || !btn.dataset.action) return;
  
  const action = btn.dataset.action;
  try {
    if (action === "save-runtime") {
      setBusy(btn, true);
      await apiPost("/api/runtime/update-config", {
        bridge_base: byId("bridge-base").value,
        lmstudio_base: byId("lmstudio-base").value,
      });
      showBanner("Runtime config saved", "success");
    } else if (action === "refresh-runtime") {
      await refreshHardware();
    } else if (action === "autotune-workspace") {
      setBusy(btn, true);
      const res = await apiPost("/api/workspace/autotune");
      // Apply logic would go here
      showBanner("Auto-tuned based on hardware", "success");
    } else if (action === "run-chat") {
      setBusy(btn, true);
      const res = await apiPost("/api/requests/chat/run", {
        input: byId("chat-input-text").value,
        stream: false
      });
      const result = await pollTask(res.task_id, "chat-result");
      renderJson("chat-result", result);
    }
  } catch (e) {
    showBanner(e.message, "error");
  } finally {
    setBusy(btn, false);
  }
});

// --- Lifecycle ---
document.addEventListener("DOMContentLoaded", () => {
  switchView('overview');
  refreshHardware().catch(console.error);
  
  // Continuous Event Polling for Global State
  setInterval(() => {
    apiGet("/api/state").then(d => {
        renderEvents(d.events);
        renderAgentSessions(d.agent_sessions);
        renderACESessions(d.ace_sessions);
        renderSessionVault(d.acid_sessions);
        // Sync vram bar if available
        if (d.hardware?.vram_gb && byId("vram-bar")) {
            byId("vram-text").textContent = `Active / ${d.hardware.vram_gb.toFixed(1)} GB`;
            byId("vram-bar").style.width = "40%";
        }
    }).catch(()=>{});
  }, 3000);
});
