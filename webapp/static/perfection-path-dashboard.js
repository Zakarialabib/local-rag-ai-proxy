/* Phase 6 Perfection Path Dashboard - JavaScript */

// Global state
let perfectionCharts = {};
let perfectionState = {
  resilience_mode: 'IDEAL',
  probing_envelope: null,
  residency_info: null,
  prewarming: null,
  truncation_patterns: null,
  reranker: null,
  vram: null,
  presets: null,
  streaming: null,
  vision: null
};

// Initialize perfection path dashboard
function initPerfectionPath() {
  console.log('Initializing Perfection Path Dashboard...');
  
  // Initialize all charts
  initCharts();
  
  // Setup event listeners
  setupEventListeners();
  
  // Load initial data
  loadAllPerfectionData();
  
  // Setup real-time SSE subscriptions
  setupSSESubscriptions();
  
  // Setup refresh intervals (5sec for most, 30sec for slow updates)
  setInterval(() => loadAllPerfectionData(), 5000);
}

// ============================================
// CHART INITIALIZATION
// ============================================

function initCharts() {
  // Chart 1: Probing Envelope (TTFT vs Context)
  const ctxProbing = document.getElementById('chart-probing');
  if (ctxProbing) {
    perfectionCharts.probing = new Chart(ctxProbing, {
      type: 'line',
      data: {
        labels: ['512', '1K', '2K', '4K', '8K', '16K'],
        datasets: [{
          label: 'TTFT (ms)',
          data: [850, 920, 1100, 1450, 2800, 26000],
          borderColor: '#00ff41',
          backgroundColor: 'rgba(0, 255, 65, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: '#00ff41',
          pointBorderColor: '#fff',
          pointBorderWidth: 1
        },
        {
          label: 'Safe Zone',
          data: [850, 920, 1100, 1450, null, null],
          borderColor: 'rgba(0, 200, 100, 0.5)',
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Cliff Edge',
          data: [null, null, null, null, 2800, 26000],
          borderColor: 'rgba(255, 100, 0, 0.6)',
          borderWidth: 2,
          backgroundColor: 'rgba(255, 100, 0, 0.1)',
          fill: false,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            labels: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: { size: 11, family: 'monospace' }
            }
          }
        },
        scales: {
          y: {
            type: 'logarithmic',
            ticks: {
              color: 'rgba(255, 255, 255, 0.5)',
              font: { size: 10 }
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.05)'
            }
          },
          x: {
            ticks: {
              color: 'rgba(255, 255, 255, 0.5)',
              font: { size: 10 }
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.05)'
            }
          }
        }
      }
    });
  }

  // Chart 2: Truncation Patterns (Bar chart)
  const ctxTruncation = document.getElementById('chart-truncation');
  if (ctxTruncation) {
    perfectionCharts.truncation = new Chart(ctxTruncation, {
      type: 'bar',
      data: {
        labels: ['Code', 'Reasoning', 'Retrieval'],
        datasets: [{
          label: 'Truncation Rate (%)',
          data: [51, 5, 0],
          backgroundColor: [
            'rgba(255, 100, 0, 0.6)',
            'rgba(0, 255, 65, 0.6)',
            'rgba(100, 150, 255, 0.6)'
          ],
          borderColor: [
            '#ff6400',
            '#00ff41',
            '#6496ff'
          ],
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            max: 100,
            ticks: {
              color: 'rgba(255, 255, 255, 0.5)',
              font: { size: 9 },
              callback: function(value) { return value + '%'; }
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.05)'
            }
          },
          y: {
            ticks: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: { size: 10 }
            },
            grid: { display: false }
          }
        }
      }
    });
  }

  // Chart 3: Streaming Mode Distribution (Pie chart)
  const ctxStreaming = document.getElementById('chart-streaming');
  if (ctxStreaming) {
    perfectionCharts.streaming = new Chart(ctxStreaming, {
      type: 'doughnut',
      data: {
        labels: ['True Streaming (>10 TPS)', 'Batch-and-Stream (<10 TPS)'],
        datasets: [{
          data: [32, 68],
          backgroundColor: [
            'rgba(0, 255, 65, 0.7)',
            'rgba(255, 150, 0, 0.7)'
          ],
          borderColor: [
            '#00ff41',
            '#ff9600'
          ],
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            labels: {
              color: 'rgba(255, 255, 255, 0.7)',
              font: { size: 10 }
            },
            position: 'bottom'
          }
        }
      }
    });
  }
}

// ============================================
// EVENT LISTENERS
// ============================================

function setupEventListeners() {
  // Refresh buttons
  document.querySelectorAll('.btn-refresh').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      const panelId = this.closest('.perfection-panel').id;
      refreshPanel(panelId.replace('panel-', ''));
    });
  });

  // Action buttons - Pre-warming
  const prewarmBtn = document.querySelector('button[onclick="triggerPrewarm()"]');
  if (prewarmBtn) {
    prewarmBtn.addEventListener('click', triggerPrewarm);
  }

  // Action buttons - VRAM Grooming
  const groomBtn = document.querySelector('button[onclick="triggerGrooming()"]');
  if (groomBtn) {
    groomBtn.addEventListener('click', triggerGrooming);
  }

  // Action buttons - Vision
  const visionBtn = document.querySelector('button[onclick="enableVision()"]');
  if (visionBtn) {
    visionBtn.addEventListener('click', enableVision);
  }
}

// ============================================
// PANEL REFRESH
// ============================================

function refreshPanel(panelType) {
  const panelMap = {
    'probing': 'panel-probing',
    'sharding': 'panel-sharding',
    'prewarming': 'panel-prewarming',
    'truncation': 'panel-truncation',
    'reranker': 'panel-reranker',
    'vram': 'panel-grooming',
    'presets': 'panel-evolution',
    'streaming': 'panel-streaming',
    'vision': 'panel-vision',
    'resilience': 'panel-resilience'
  };

  const panelEl = document.getElementById(panelMap[panelType]);
  if (panelEl) {
    panelEl.classList.add('loading');
    
    loadPerfectionData(panelType).then(() => {
      panelEl.classList.remove('loading');
    }).catch(err => {
      console.error('Failed to load panel:', panelType, err);
      panelEl.classList.remove('loading');
    });
  }
}

// ============================================
// DATA LOADING
// ============================================

async function loadAllPerfectionData() {
  try {
    // Load all panel data in parallel
    await Promise.all([
      loadPerfectionData('probing'),
      loadPerfectionData('sharding'),
      loadPerfectionData('prewarming'),
      loadPerfectionData('truncation'),
      loadPerfectionData('reranker'),
      loadPerfectionData('vram'),
      loadPerfectionData('presets'),
      loadPerfectionData('streaming'),
      loadPerfectionData('vision'),
      loadPerfectionData('resilience')
    ]);
  } catch (err) {
    console.error('Failed to load perfection data:', err);
  }
}

async function loadPerfectionData(panelType) {
  try {
    let endpoint = '';
    switch(panelType) {
      case 'probing':
        endpoint = '/api/perfection/probing/envelope';
        break;
      case 'sharding':
        endpoint = '/api/perfection/sharding/residency';
        break;
      case 'prewarming':
        endpoint = '/api/perfection/prewarming/metrics';
        break;
      case 'truncation':
        endpoint = '/api/perfection/truncation/patterns';
        break;
      case 'reranker':
        endpoint = '/api/perfection/reranker/stats';
        break;
      case 'vram':
        endpoint = '/api/perfection/vram/fragmentation';
        break;
      case 'presets':
        endpoint = '/api/perfection/presets/lineage';
        break;
      case 'streaming':
        endpoint = '/api/perfection/streaming/mode-distribution';
        break;
      case 'vision':
        endpoint = '/api/perfection/vision/capability-status';
        break;
      case 'resilience':
        endpoint = '/api/perfection/resilience/status';
        break;
    }

    if (!endpoint) return;

    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    updatePerfectionPanel(panelType, data);
  } catch (err) {
    console.error(`Failed to load ${panelType}:`, err);
  }
}

function updatePerfectionPanel(panelType, data) {
  switch(panelType) {
    case 'probing':
      updateProbingPanel(data);
      break;
    case 'sharding':
      updateShardingPanel(data);
      break;
    case 'prewarming':
      updatePrewarmingPanel(data);
      break;
    case 'truncation':
      updateTruncationPanel(data);
      break;
    case 'reranker':
      updateRerankerPanel(data);
      break;
    case 'vram':
      updateVramPanel(data);
      break;
    case 'presets':
      updatePresetsPanel(data);
      break;
    case 'streaming':
      updateStreamingPanel(data);
      break;
    case 'vision':
      updateVisionPanel(data);
      break;
    case 'resilience':
      updateResiliencePanel(data);
      break;
  }
}

function updateProbingPanel(data) {
  if (data.safe_zone_max) {
    document.getElementById('stat-safe-zone').textContent = `0-${data.safe_zone_max} tokens`;
  }
  if (data.cliff_edge) {
    document.getElementById('stat-cliff').textContent = `${data.cliff_edge} tokens`;
  }
  if (data.trend) {
    document.getElementById('stat-trend').textContent = data.trend;
  }
}

function updateShardingPanel(data) {
  if (data.resident_vram_gb !== undefined) {
    const total = 8;
    const used = data.resident_vram_gb + (data.systemram_gb || 0) + (data.disk_gb || 0);
    const info = document.getElementById('residency-info');
    if (info) {
      info.innerHTML = `
        <p>Resident (VRAM): <strong>${data.resident_vram_gb.toFixed(1)}GB</strong></p>
        <p>System RAM: <strong>${(data.systemram_gb || 0).toFixed(1)}GB</strong></p>
        <p>Total Used: <strong>${used.toFixed(1)}GB / ${total}GB</strong></p>
        <p>Free VRAM: <strong>${(total - used).toFixed(1)}GB</strong></p>
      `;
    }
  }
}

function updatePrewarmingPanel(data) {
  if (data.cold_ttft_ms) {
    const coldBox = document.querySelector('.metric-box.cold value');
    if (coldBox) coldBox.textContent = `${data.cold_ttft_ms.toFixed(0)}s`;
  }
  if (data.hot_ttft_ms) {
    const hotBox = document.querySelector('.metric-box.hot value');
    if (hotBox) hotBox.textContent = `${(data.hot_ttft_ms / 1000).toFixed(1)}s`;
  }
  if (data.last_idle_seconds) {
    document.getElementById('idle-time').textContent = `${data.last_idle_seconds}s`;
  }
  if (data.prediction) {
    document.getElementById('warm-status').textContent = data.prediction;
  }
}

function updateTruncationPanel(data) {
  if (data.patterns && perfectionCharts.truncation) {
    const labels = [];
    const rates = [];
    Object.entries(data.patterns).forEach(([taskType, pattern]) => {
      labels.push(taskType.charAt(0).toUpperCase() + taskType.slice(1));
      rates.push(pattern.truncation_rate * 100);
    });
    perfectionCharts.truncation.data.labels = labels;
    perfectionCharts.truncation.data.datasets[0].data = rates;
    perfectionCharts.truncation.update();
  }
}

function updateRerankerPanel(data) {
  if (data.fast_model_usage_percent) {
    const fastBox = document.querySelector('.reranker-options .option.active');
    if (fastBox) {
      fastBox.querySelector('p:nth-child(3)').textContent = `Used: ${data.fast_model_usage_percent}% of time`;
    }
  }
}

function updateVramPanel(data) {
  if (data.fragmentation_mb) {
    const alert = document.querySelector('.fragmentation-alert p:nth-child(3)');
    if (alert) {
      const percentage = ((data.fragmentation_mb / 7500) * 100).toFixed(0);
      alert.textContent = `Loss: ${data.fragmentation_mb}MB (${percentage}%)`;
    }
  }
  if (data.next_groom_seconds) {
    const schedule = document.querySelector('.groom-schedule p');
    if (schedule) {
      const eta = new Date(Date.now() + data.next_groom_seconds * 1000).toLocaleTimeString();
      schedule.textContent = `Next: ${eta}`;
    }
  }
}

function updatePresetsPanel(data) {
  // Update will be from mock tree - static for now
}

function updateStreamingPanel(data) {
  if (data.true_streaming_percent !== undefined && perfectionCharts.streaming) {
    perfectionCharts.streaming.data.datasets[0].data = [
      data.true_streaming_percent,
      100 - data.true_streaming_percent
    ];
    perfectionCharts.streaming.update();
  }
}

function updateVisionPanel(data) {
  if (data.runtime_masked !== undefined) {
    const status = document.querySelector('.capability.runtime p:nth-child(2)');
    if (status) {
      status.textContent = `vision: ${data.runtime_masked ? '❌ MASKED' : '✓ ENABLED'}`;
    }
  }
}

function updateResiliencePanel(data) {
  const banner = document.querySelector('.resilience-mode-display');
  if (banner && data.current_mode) {
    const emoji = data.emoji || '🟢';
    const mode = data.current_mode;
    banner.innerHTML = `
      <span class="mode-emoji">${emoji}</span>
      <div>
        <span class="mode-name">${mode}</span>
        <small>${getModelDescription(mode)}</small>
      </div>
    `;
  }
}

function getModelDescription(mode) {
  const descriptions = {
    'IDEAL': 'Qwen3.5-4B Q4_K_M + Embed4B + Rerank0.6B',
    'PRESSURE': 'Qwen3.5-4B Q3_K_M + Embed4B (reduced quantization)',
    'EMERGENCY': 'Lfm2.5-1.2B + Embed4B (smaller model)',
    'RETRIEVAL': 'Embed4B + Rerank0.6B only (no generation)',
    'CIRCUIT_BREAK': 'System offline - manual restart required'
  };
  return descriptions[mode] || '';
}

// ============================================
// ACTION HANDLERS
// ============================================

async function triggerPrewarm() {
  try {
    const response = await fetch('/api/perfection/prewarming/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force: true })
    });
    if (response.ok) {
      const data = await response.json();
      console.log('Pre-warming triggered:', data);
      updatePrewarmingPanel(data);
      showNotification('✓ Pre-warming cycle started', 'success');
    }
  } catch (err) {
    console.error('Pre-warming failed:', err);
    showNotification('✗ Pre-warming failed', 'error');
  }
}

async function triggerGrooming() {
  try {
    const response = await fetch('/api/perfection/vram/groom-now', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (response.ok) {
      const data = await response.json();
      console.log('VRAM grooming triggered:', data);
      showNotification('✓ VRAM grooming started (32s estimated)', 'success');
      
      // Refresh after grooming completes
      setTimeout(() => {
        loadPerfectionData('vram');
      }, 35000);
    }
  } catch (err) {
    console.error('Grooming failed:', err);
    showNotification('✗ Grooming failed', 'error');
  }
}

async function enableVision() {
  try {
    const response = await fetch('/api/perfection/vision/enable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (response.ok) {
      const data = await response.json();
      console.log('Vision enabled:', data);
      updateVisionPanel(data);
      showNotification('✓ Vision capability enabled', 'success');
    }
  } catch (err) {
    console.error('Vision enable failed:', err);
    showNotification('✗ Vision enable failed', 'error');
  }
}

// ============================================
// SSE SUBSCRIPTIONS
// ============================================

function setupSSESubscriptions() {
  // Subscribe to hot/warm/cold events for real-time updates
  if (window.setupSSEChannel) {
    window.setupSSEChannel('perfection-metrics', (event) => {
      const data = JSON.parse(event.data);
      console.log('Perfection metrics update:', data);
      
      // Update resilience mode in real-time
      if (data.mode) {
        const emoji = data.emoji || '🟢';
        const banner = document.querySelector('.resilience-mode-display .mode-emoji');
        if (banner) banner.textContent = emoji;
        
        const name = document.querySelector('.resilience-mode-display .mode-name');
        if (name) name.textContent = data.mode;
      }
    });
  }
}

// ============================================
// UTILITIES
// ============================================

function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 12px 18px;
    background: ${type === 'success' ? 'rgba(0, 255, 65, 0.2)' : 'rgba(255, 100, 0, 0.2)'};
    border: 1px solid ${type === 'success' ? 'rgba(0, 255, 65, 0.4)' : 'rgba(255, 100, 0, 0.4)'};
    color: ${type === 'success' ? '#00ff41' : '#ffb366'};
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
  `;
  notification.textContent = message;
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-in';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Auto-initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
  // Check if perfection path view should be shown
  const perfectionView = document.getElementById('perfection-path');
  if (perfectionView) {
    initPerfectionPath();
  }
});

// Export for use in dashboard
window.initPerfectionPath = initPerfectionPath;
window.refreshPanel = refreshPanel;
window.triggerPrewarm = triggerPrewarm;
window.triggerGrooming = triggerGrooming;
window.enableVision = enableVision;
