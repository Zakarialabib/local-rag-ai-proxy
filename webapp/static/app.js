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
  if (!response.ok) {
    throw new Error(data.error || data.detail || "Request failed");
  }
  return data;
}

async function apiGet(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || data.detail || "Request failed");
  }
  return data;
}

function renderJson(targetId, payload) {
  byId(targetId).textContent = JSON.stringify(payload, null, 2);
}

function setBusy(button, busy) {
  if (!button) return;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Working...";
    button.disabled = true;
    button.classList.add("opacity-60");
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
    button.classList.remove("opacity-60");
  }
}

async function pollTask(taskId, targetId) {
  for (;;) {
    const task = await apiGet(`/api/tasks/${taskId}`);
    byId("last-task-id").textContent = task.id;
    if (targetId) {
      renderJson(targetId, task);
    }
    if (task.status === "completed") {
      showBanner(`${task.name} completed`, "success");
      await refreshState();
      return task.result;
    }
    if (task.status === "failed") {
      showBanner(task.error || `${task.name} failed`, "error");
      throw new Error(task.error || `${task.name} failed`);
    }
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

function responsesForm() {
  return {
    model: byId("selected-model").value,
    instructions: byId("responses-instructions").value,
    input: byId("responses-input").value,
    mode: byId("responses-mode").value,
    reasoning_effort: byId("responses-effort").value,
    max_output_tokens: byId("responses-max-output").value,
    temperature: byId("responses-temperature").value,
    top_p: byId("responses-top-p").value,
    top_k: byId("responses-top-k").value,
    repeat_penalty: byId("responses-repeat-penalty").value,
    retrieval_top_k: byId("responses-retrieval-top-k").value,
    max_context_chars: byId("responses-max-context-chars").value,
    chunk_size: byId("responses-chunk-size").value,
    chunk_overlap: byId("responses-chunk-overlap").value,
  };
}

function chatForm() {
  return {
    model: byId("selected-model").value,
    system_prompt: byId("chat-system-prompt").value,
    input: byId("chat-input").value,
    mode: byId("chat-mode").value,
    max_tokens: byId("chat-max-output").value,
    temperature: byId("chat-temperature").value,
    top_p: byId("chat-top-p").value,
    top_k: byId("chat-top-k").value,
    repeat_penalty: byId("chat-repeat-penalty").value,
    stream: byId("chat-stream").checked,
  };
}

function roleForm() {
  return {
    main: byId("role-main").value,
    reasoning: byId("role-reasoning").value,
    embed: byId("role-embed").value,
    rerank: byId("role-rerank").value,
  };
}

async function refreshState() {
  const state = await apiGet("/api/state");
  byId("runtime-status").textContent = state.state.runtime_status;
  byId("loaded-models").textContent = state.state.loaded_models.length ? state.state.loaded_models.join("\n") : "No loaded models reported";
  renderJson("profile-json", state.state.profile || {});
  renderAgentSessions(state.agent_sessions || []);
  renderACESessions(state.ace_sessions || []);
  renderEvents(state.events || []);
}

function scheduleRefreshState() {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  refreshTimer = setTimeout(() => {
    refreshState().catch(() => {});
    refreshTimer = null;
  }, 250);
}

function renderEvents(events) {
  const box = byId("events-log");
  const lines = (events || []).map((event) => `[${event.ts}] ${event.type}: ${event.message}${Object.keys(event.data || {}).length ? ` ${JSON.stringify(event.data)}` : ""}`);
  box.textContent = lines.join("\n");
  box.scrollTop = box.scrollHeight;
}

function renderAgentSessions(sessions) {
  agentSessions = sessions || [];
  const select = byId("agent-session-select");
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  for (const item of agentSessions) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id} | ${item.workflow} | ${item.current_agent}`;
    if (item.id === current) option.selected = true;
    select.appendChild(option);
  }
}

function renderACESessions(sessions) {
  aceSessions = sessions || [];
  const select = byId("ace-session-select");
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Create from next run";
  select.appendChild(blank);
  for (const item of aceSessions) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id} | ${item.mode} | ${item.status}`;
    if (item.id === current) option.selected = true;
    select.appendChild(option);
  }
}

async function refreshAgentSessions() {
  const data = await apiGet("/api/agent/sessions");
  renderAgentSessions(data.sessions || []);
}

async function refreshACESessions() {
  const data = await apiGet("/api/ace/sessions");
  renderACESessions(data.sessions || []);
}

async function refreshAgentState() {
  const sessionId = byId("agent-session-select").value;
  if (!sessionId) return;
  const data = await apiGet(`/api/agent/sessions/${sessionId}/state`);
  renderJson("agent-state-box", data);
}

async function refreshACETrace() {
  const sessionId = byId("ace-session-select").value;
  if (!sessionId) return;
  const data = await apiGet(`/api/ace/sessions/${sessionId}/trace`);
  renderJson("ace-trace-box", data);
  if (data.final_text || data.reasoning_text) {
    renderJson("ace-final-box", {
      content: data.final_text || "",
      reasoning: data.reasoning_text || "",
      selections: data.selections || [],
    });
  }
}

function startAgentEventStream() {
  const sessionId = byId("agent-session-select").value;
  if (!sessionId) return;
  if (agentEventSource) {
    agentEventSource.close();
  }
  agentEventSource = new EventSource(`/api/agent/sessions/${sessionId}/events`);
  agentEventSource.onmessage = (event) => {
    const box = byId("agent-events-box");
    const parsed = JSON.parse(event.data);
    box.textContent += `[${parsed.ts}] ${parsed.type} ${JSON.stringify(parsed.data)}\n`;
    box.scrollTop = box.scrollHeight;
    scheduleRefreshState();
  };
}

function applyWorkspace(workspace) {
  if (!workspace) return;
  const responses = workspace.responses || {};
  const chat = workspace.chat || {};
  byId("responses-instructions").value = responses.instructions || "";
  byId("responses-mode").value = responses.mode || "fast";
  byId("responses-effort").value = responses.reasoning_effort || "medium";
  byId("responses-max-output").value = responses.max_output_tokens ?? "";
  byId("responses-temperature").value = responses.temperature ?? "";
  byId("responses-top-p").value = responses.top_p ?? "";
  byId("responses-top-k").value = responses.top_k ?? "";
  byId("responses-repeat-penalty").value = responses.repeat_penalty ?? "";
  byId("responses-retrieval-top-k").value = responses.retrieval_top_k ?? "";
  byId("responses-max-context-chars").value = responses.max_context_chars ?? "";
  if (responses.chunk_size !== undefined) byId("responses-chunk-size").value = responses.chunk_size;
  if (responses.chunk_overlap !== undefined) byId("responses-chunk-overlap").value = responses.chunk_overlap;

  byId("chat-system-prompt").value = chat.system_prompt || "";
  byId("chat-mode").value = chat.mode || "fast";
  byId("chat-max-output").value = chat.max_tokens ?? "";
  byId("chat-temperature").value = chat.temperature ?? "";
  byId("chat-top-p").value = chat.top_p ?? "";
  byId("chat-top-k").value = chat.top_k ?? "";
  byId("chat-repeat-penalty").value = chat.repeat_penalty ?? "";
  byId("chat-stream").checked = Boolean(chat.stream);

  const notes = []
    .concat((workspace.notes || []).map((item) => `- ${item}`))
    .concat([`Capabilities: ${JSON.stringify(workspace.capabilities || {}, null, 2)}`]);
  byId("workspace-notes").textContent = notes.join("\n");
}

function loadWorkspaceTest() {
  const selected = workspaceTests.find((item) => item.id === byId("workspace-test-select").value);
  if (!selected) return;
  byId("responses-input").value = selected.input || "";
  byId("chat-input").value = selected.input || "";
  if (selected.use_case) {
    byId("use-case").value = selected.use_case;
  }
  byId("workspace-notes").textContent = `Loaded test input: ${selected.label}\nUse case: ${selected.use_case}`;
}

function applyWorkspacePreset() {
  const selected = workspacePresets.find((item) => item.id === byId("workspace-preset-select").value);
  if (!selected) return;
  applyWorkspace({
    responses: selected.responses || {},
    chat: selected.chat || {},
    capabilities: {},
    notes: selected.notes || [],
  });
  if (selected.target_use_case) {
    byId("use-case").value = selected.target_use_case;
  }
}

function startEventStream() {
  const stream = new EventSource("/events");
  stream.onmessage = async () => {
    scheduleRefreshState();
  };
}

function agentTurnForm() {
  return {
    input: {
      type: byId("agent-input-type").value || "user_request",
      content: byId("agent-input").value,
    },
  };
}

function aceForm() {
  let docs = [];
  const rawDocs = byId("ace-docs").value.trim();
  if (rawDocs) {
    try {
      docs = JSON.parse(rawDocs);
    } catch (error) {
      docs = rawDocs.split("\n").map((item) => item.trim()).filter(Boolean);
    }
  }
  return {
    session_id: byId("ace-session-select").value,
    prompt: byId("ace-prompt").value,
    docs,
    mode: byId("ace-mode").value || "think",
    temperature: byId("ace-temperature").value,
    max_output_tokens: byId("ace-max-output").value,
    top_p: byId("ace-top-p").value,
    top_k: byId("ace-top-k").value,
    repeat_penalty: byId("ace-repeat-penalty").value,
  };
}

async function streamACE() {
  byId("ace-stream-box").textContent = "";
  byId("ace-final-box").textContent = "";
  const response = await fetch("/api/ace/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrfToken,
    },
    body: JSON.stringify(aceForm()),
  });
  if (!response.ok || !response.body) {
    let error = "ACE request failed";
    try {
      const data = await response.json();
      error = data.error || data.detail || error;
    } catch (_) {
      // no-op
    }
    throw new Error(error);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let latestSessionId = byId("ace-session-select").value;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const lines = part.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event: "));
      const dataLine = lines.find((line) => line.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      const eventType = eventLine.slice(7).trim();
      const payload = JSON.parse(dataLine.slice(6));
      if (payload.session_id) {
        latestSessionId = payload.session_id;
        byId("ace-session-select").value = latestSessionId;
      }
      const box = byId("ace-stream-box");
      box.textContent += `[${eventType}] ${JSON.stringify(payload.data)}\n`;
      box.scrollTop = box.scrollHeight;
      if (eventType === "final_output") {
        renderJson("ace-final-box", payload.data);
      }
    }
  }
  await refreshACESessions();
  if (latestSessionId) {
    byId("ace-session-select").value = latestSessionId;
    await refreshACETrace();
  }
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;

  try {
    if (button.dataset.action === "save-runtime") {
      setBusy(button, true);
      await apiPost("/api/runtime/update-config", {
        bridge_base: byId("bridge-base").value,
        lmstudio_base: byId("lmstudio-base").value,
      });
      showBanner("Runtime endpoints saved", "success");
      return;
    }

    if (button.dataset.action === "refresh-runtime") {
      setBusy(button, true);
      const result = await apiPost("/api/runtime/refresh", {});
      byId("runtime-status").textContent = result.runtime_status;
      byId("loaded-models").textContent = result.loaded_models.length ? result.loaded_models.join("\n") : "No loaded models reported";
      showBanner("Runtime refreshed", "success");
      await refreshState();
      return;
    }

    if (button.dataset.action === "refresh-agent-sessions") {
      setBusy(button, true);
      await refreshAgentSessions();
      showBanner("Agent sessions refreshed", "success");
      return;
    }

    if (button.dataset.action === "refresh-ace-sessions") {
      setBusy(button, true);
      await refreshACESessions();
      showBanner("ACE sessions refreshed", "success");
      return;
    }

    if (button.dataset.action === "create-agent-session") {
      setBusy(button, true);
      const result = await apiPost("/api/agent/sessions", {
        workflow: byId("agent-workflow").value,
        tool_budget: byId("agent-tool-budget").value,
      });
      await refreshAgentSessions();
      byId("agent-session-select").value = result.session_id;
      startAgentEventStream();
      await refreshAgentState();
      showBanner("Agent session created", "success");
      return;
    }

    if (button.dataset.action === "refresh-agent-state") {
      setBusy(button, true);
      await refreshAgentState();
      startAgentEventStream();
      showBanner("Agent state refreshed", "success");
      return;
    }

    if (button.dataset.action === "run-agent-turn") {
      setBusy(button, true);
      const sessionId = byId("agent-session-select").value;
      const task = await apiPost(`/api/agent/sessions/${sessionId}/turn`, agentTurnForm());
      const result = await pollTask(task.task_id, "agent-result-box");
      renderJson("agent-result-box", result);
      await refreshAgentState();
      startAgentEventStream();
      return;
    }

    if (button.dataset.action === "branch-agent-session") {
      setBusy(button, true);
      const sessionId = byId("agent-session-select").value;
      const result = await apiPost(`/api/agent/sessions/${sessionId}/branch`, {
        checkpoint_id: byId("agent-checkpoint-id").value,
      });
      await refreshAgentSessions();
      byId("agent-session-select").value = result.session_id;
      startAgentEventStream();
      await refreshAgentState();
      showBanner("Agent session branched", "success");
      return;
    }

    if (button.dataset.action === "restore-agent-checkpoint") {
      setBusy(button, true);
      const sessionId = byId("agent-session-select").value;
      const result = await apiPost(`/api/agent/sessions/${sessionId}/restore`, {
        checkpoint_id: byId("agent-checkpoint-id").value,
      });
      renderJson("agent-state-box", result);
      startAgentEventStream();
      showBanner("Checkpoint restored", "success");
      return;
    }

    if (button.dataset.action === "run-ace") {
      setBusy(button, true);
      await streamACE();
      showBanner("ACE generation completed", "success");
      return;
    }

    if (button.dataset.action === "refresh-ace-trace") {
      setBusy(button, true);
      await refreshACETrace();
      showBanner("ACE trace refreshed", "success");
      return;
    }

    if (button.dataset.action === "ace-select-option") {
      setBusy(button, true);
      const sessionId = byId("ace-session-select").value;
      if (!sessionId) {
        throw new Error("Run or select an ACE session first");
      }
      const result = await apiPost("/api/ace/select-option", {
        session_id: sessionId,
        option_id: byId("ace-option-id").value,
      });
      renderJson("ace-final-box", result);
      await refreshACETrace();
      showBanner("ACE selection recorded", "success");
      return;
    }

    if (button.dataset.action === "recompute-profile") {
      setBusy(button, true);
      const result = await apiPost("/api/profile/recompute", {
        selected_model: byId("selected-model").value,
        use_case: byId("use-case").value,
        backend: byId("backend").value,
      });
      renderJson("profile-json", result.profile);
      showBanner("Profile recomputed", "success");
      return;
    }

    if (button.dataset.action === "save-roles") {
      setBusy(button, true);
      await apiPost("/api/roles/save", roleForm());
      showBanner("Role mapping saved to .env", "success");
      return;
    }

    if (button.dataset.loadRole) {
      setBusy(button, true);
      const task = await apiPost("/api/models/load", {
        role: button.dataset.loadRole,
        model: byId(`role-${button.dataset.loadRole}`).value,
      });
      await pollTask(task.task_id, "preset-result");
      return;
    }

    if (button.dataset.loadDirect) {
      setBusy(button, true);
      const task = await apiPost("/api/models/load", {
        role: "main",
        model: button.dataset.loadDirect,
      });
      await pollTask(task.task_id, "preset-result");
      return;
    }

    if (button.dataset.unloadModel) {
      setBusy(button, true);
      const task = await apiPost("/api/models/unload", {
        model: button.dataset.unloadModel,
      });
      await pollTask(task.task_id, "preset-result");
      return;
    }

    if (button.dataset.action === "preview-responses") {
      setBusy(button, true);
      const result = await apiPost("/api/requests/responses/preview", responsesForm());
      renderJson("responses-preview", result);
      showBanner("Responses preview updated", "success");
      return;
    }

    if (button.dataset.action === "autotune-workspace") {
      setBusy(button, true);
      const result = await apiPost("/api/workspace/autotune", {
        model: byId("selected-model").value,
      });
      applyWorkspace(result.workspace);
      showBanner("Workspace auto-tuned from model, hardware, and role state", "success");
      return;
    }

    if (button.dataset.action === "load-workspace-test") {
      loadWorkspaceTest();
      showBanner("Workspace test input loaded", "success");
      return;
    }

    if (button.dataset.action === "apply-workspace-preset") {
      applyWorkspacePreset();
      showBanner("Workspace preset applied", "success");
      return;
    }

    if (button.dataset.action === "run-responses") {
      setBusy(button, true);
      const task = await apiPost("/api/requests/responses/run", responsesForm());
      const result = await pollTask(task.task_id, "responses-result");
      renderJson("responses-result", result);
      return;
    }

    if (button.dataset.action === "preview-chat") {
      setBusy(button, true);
      const result = await apiPost("/api/requests/chat/preview", chatForm());
      renderJson("chat-preview", result);
      showBanner("Chat preview updated", "success");
      return;
    }

    if (button.dataset.action === "run-chat") {
      setBusy(button, true);
      const task = await apiPost("/api/requests/chat/run", chatForm());
      const result = await pollTask(task.task_id, "chat-result");
      renderJson("chat-result", result);
      return;
    }

    if (button.dataset.action === "apply-preset") {
      setBusy(button, true);
      const result = await apiPost("/api/presets/apply", {
        path: byId("preset-select").value,
      });
      renderJson("preset-result", result);
      renderJson("profile-json", result.profile);
      showBanner("Preset applied", "success");
      return;
    }

    if (button.dataset.action === "export-preset") {
      setBusy(button, true);
      const result = await apiPost("/api/presets/export", {
        selected_model: byId("selected-model").value,
        use_case: byId("use-case").value,
        backend: byId("backend").value,
      });
      renderJson("preset-result", result);
      showBanner("Preset exported to LM Studio config-presets", "success");
      return;
    }
  } catch (error) {
    showBanner(error.message || String(error), "error");
  } finally {
    setBusy(button, false);
  }
});

// ACE Session Creation & Form Generation Functions

let currentAceProfile = null;
let aceFormFields = [];

async function aceAnalyzePrompt(prompt) {
  try {
    const response = await apiPost("/v1/ace/analyze", { system_prompt: prompt });
    return response;
  } catch (error) {
    console.error("Failed to analyze prompt:", error);
    return null;
  }
}

async function aceGenerateDynamicFields(profile) {
  if (!profile) return;
  currentAceProfile = profile;
  
  // Base fields
  aceFormFields = [
    { name: "topic", label: "Topic or Question", type: "text" },
    { name: "constraints", label: "Key Constraints", type: "textarea" },
    { name: "goals", label: "Primary Goals", type: "textarea" },
  ];
  
  // Role-specific fields
  if (profile.role_type === "researcher") {
    aceFormFields.push(
      { name: "source_quality", label: "Source Quality Level", type: "select", options: ["high", "medium", "low"] },
      { name: "depth", label: "Analysis Depth", type: "select", options: ["surface", "moderate", "deep"] }
    );
  } else if (profile.role_type === "coder") {
    aceFormFields.push(
      { name: "language", label: "Programming Language", type: "select", options: ["python", "javascript", "rust", "go", "c++"] },
      { name: "style", label: "Code Style", type: "select", options: ["pep8", "google", "airbnb"] }
    );
  }
  
  // Render fields
  const container = byId("ace-context-form");
  container.innerHTML = aceFormFields.map(field => {
    if (field.type === "select") {
      return `<label class="block"><span class="mb-2 block text-sm text-slate-300">${field.label}</span><select id="ace-field-${field.name}" class="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm">${field.options.map(o => `<option value="${o}">${o}</option>`).join("")}</select></label>`;
    } else if (field.type === "textarea") {
      return `<label class="block"><span class="mb-2 block text-sm text-slate-300">${field.label}</span><textarea id="ace-field-${field.name}" class="min-h-20 rounded-2xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm"></textarea></label>`;
    } else {
      return `<label class="block"><span class="mb-2 block text-sm text-slate-300">${field.label}</span><input type="${field.type}" id="ace-field-${field.name}" class="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm"></label>`;
    }
  }).join("");
  
  container.classList.remove("hidden");
  byId("ace-override-label").classList.remove("hidden");
  byId("ace-intervention-triggers").classList.remove("hidden");
}

function aceShowAnalysis(profile) {
  const analysisDiv = byId("ace-agent-analysis");
  const contentDiv = byId("ace-analysis-content");
  
  analysisDiv.classList.remove("hidden");
  contentDiv.innerHTML = `
    <div class="space-y-2">
      <p><strong>Role:</strong> ${profile.role_type || "General"}</p>
      <p><strong>Domain:</strong> ${profile.domain || "General"}</p>
      <p><strong>Output Format:</strong> ${profile.output_format || "Text"}</p>
      ${profile.goals && profile.goals.length ? `<p><strong>Goals:</strong> ${profile.goals.slice(0, 2).join(", ")}</p>` : ""}
      ${profile.constraints && profile.constraints.length ? `<p><strong>Constraints:</strong> ${profile.constraints.slice(0, 2).join(", ")}</p>` : ""}
    </div>
  `;
}

async function aceCreateSession() {
  const agentName = byId("ace-agent-name").value.trim();
  const systemPrompt = byId("ace-system-prompt").value.trim();
  
  if (!systemPrompt) {
    showBanner("System prompt is required", "error");
    return;
  }
  
  if (!currentAceProfile) {
    showBanner("Analyze system prompt first", "error");
    return;
  }
  
  try {
    const context = {};
    for (const field of aceFormFields) {
      const value = byId(`ace-field-${field.name}`)?.value;
      if (value) context[field.name] = value;
    }
    
    const result = await apiPost("/v1/ace/generate", {
      agent_name: agentName || "ACE Agent",
      system_prompt: systemPrompt,
      context: context,
    });
    
    showBanner(`ACE session created: ${result.session_id}`, "success");
    byId("ace-agent-name").value = "";
    byId("ace-system-prompt").value = "";
    byId("ace-context-form").classList.add("hidden");
    byId("ace-agent-analysis").classList.add("hidden");
    await aceRefreshSessions();
    
  } catch (error) {
    showBanner(error.message || "Failed to create session", "error");
  }
}

async function aceRefreshSessions() {
  try {
    const data = await apiGet("/v1/ace/sessions");
    const sessions = data.sessions || data || [];
    const tbody = byId("ace-sessions-list");
    
    if (!sessions || sessions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="px-3 py-2 text-center text-xs text-slate-400">No sessions</td></tr>';
      return;
    }
    
    tbody.innerHTML = sessions.map(session => `
      <tr>
        <td class="px-3 py-2 text-xs font-mono">${(session.id || "N/A").substring(0, 8)}</td>
        <td class="px-3 py-2 text-xs">${session.agent_name || session.name || "Unknown"}</td>
        <td class="px-3 py-2 text-xs"><span class="rounded px-2 py-1 bg-slate-800 text-slate-300">${session.status || "active"}</span></td>
        <td class="px-3 py-2 text-xs"><button class="rounded px-2 py-1 hover:bg-slate-800" onclick="aceLoadTrace('${session.id}')">View</button></td>
      </tr>
    `).join("");
  } catch (error) {
    console.error("Failed to refresh ACE sessions:", error);
  }
}

async function aceLoadTrace(sessionId) {
  try {
    const data = await apiGet(`/v1/ace/sessions/${sessionId}/trace`);
    byId("ace-trace-box").textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    showBanner("Failed to load trace", "error");
  }
}

// Monitor system prompt changes for real-time analysis
byId("ace-system-prompt")?.addEventListener("blur", async () => {
  const prompt = byId("ace-system-prompt").value.trim();
  if (!prompt) {
    byId("ace-agent-analysis").classList.add("hidden");
    return;
  }
  
  const profile = await aceAnalyzePrompt(prompt);
  if (profile) {
    aceShowAnalysis(profile);
    await aceGenerateDynamicFields(profile);
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  renderEvents(dashboard.events || []);
  renderACESessions(dashboard.ace_sessions || []);
  aceRefreshSessions();
  startEventStream();
  startAgentEventStream();
  await refreshState();
});

document.addEventListener("change", async (event) => {
  if (event.target && event.target.id === "agent-session-select") {
    startAgentEventStream();
    await refreshAgentState();
  }
  if (event.target && event.target.id === "ace-session-select") {
    await refreshACETrace();
  }
});
