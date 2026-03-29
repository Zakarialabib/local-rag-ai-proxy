/**
 * Perfection Index & Tool Health Dashboard UI
 * Real-time monitoring and analytics visualization
 */

class PerfectionDashboard {
    constructor() {
        this.refreshInterval = 5000; // 5 seconds
        this.anomaliesChart = null;
        this.healthChart = null;
        this.init();
    }

    init() {
        console.log("PerfectionDashboard initialized");
        this.setupEventListeners();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Refresh buttons
        document.querySelectorAll('[data-action="refresh-health"]').forEach(btn => {
            btn.addEventListener('click', () => this.refreshHealth());
        });

        document.querySelectorAll('[data-action="refresh-perfection"]').forEach(btn => {
            btn.addEventListener('click', () => this.refreshPerfection());
        });

        document.querySelectorAll('[data-action="detect-anomalies"]').forEach(btn => {
            btn.addEventListener('click', () => this.detectAnomalies());
        });

        document.querySelectorAll('[data-action="view-remediation"]').forEach(btn => {
            btn.addEventListener('click', () => this.viewRemediationActions());
        });
    }

    startAutoRefresh() {
        setInterval(() => {
            this.refreshHealth();
            this.refreshPerfection();
        }, this.refreshInterval);
    }

    async refreshHealth() {
        try {
            const response = await fetch('/api/tools/health');
            const data = await response.json();
            
            if (data.ok) {
                this.renderHealthWidget(data);
                this.updateHealthTimeline();
            }
        } catch (error) {
            console.error("Health refresh error:", error);
        }
    }

    async refreshPerfection() {
        try {
            const response = await fetch('/api/tools/perfection');
            const data = await response.json();
            
            if (data.ok) {
                this.renderPerfectionWidget(data);
            }
        } catch (error) {
            console.error("Perfection refresh error:", error);
        }
    }

    async detectAnomalies() {
        try {
            const response = await fetch('/api/tools/analytics/detect-anomalies', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await response.json();
            
            if (data.ok) {
                this.displayAnomalies(data.anomalies);
            }
        } catch (error) {
            console.error("Anomaly detection error:", error);
        }
    }

    async viewRemediationActions() {
        try {
            const response = await fetch('/api/tools/remediation/pending?limit=20');
            const data = await response.json();
            
            if (data.ok) {
                this.displayRemediationActions(data.pending_actions);
            }
        } catch (error) {
            console.error("Remediation fetch error:", error);
        }
    }

    async updateHealthTimeline() {
        try {
            const response = await fetch('/api/tools/health/timeline');
            const data = await response.json();
            
            if (data.ok) {
                this.renderTimeline(data.timeline);
            }
        } catch (error) {
            console.error("Timeline fetch error:", error);
        }
    }

    renderHealthWidget(data) {
        const container = document.querySelector('[data-widget="health-status"]');
        if (!container) return;

        const overallHealth = data.tools ? 
            (Object.values(data.tools).filter(t => t.status === 'healthy').length / 
             Object.values(data.tools).length * 100) : 0;

        const html = `
            <div class="health-widget">
                <h3>Tool Health Status</h3>
                <div class="health-gauge" style="width: ${overallHealth}%; background: ${this.getHealthColor(overallHealth)};">
                    <span>${overallHealth.toFixed(1)}%</span>
                </div>
                <div class="tool-statuses">
                    ${Object.entries(data.tools || {}).map(([name, status]) => `
                        <div class="tool-status ${status.status}">
                            <span class="tool-name">${name}</span>
                            <span class="status-badge">${status.status}</span>
                            <span class="error-rate">${(status.error_rate * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
                <div class="timestamp">Last updated: ${new Date(data.overall_health * 1000).toLocaleTimeString()}</div>
            </div>
        `;
        container.innerHTML = html;
    }

    renderPerfectionWidget(data) {
        const container = document.querySelector('[data-widget="perfection-index"]');
        if (!container) return;

        const globalIndex = (data.global_index || 0).toFixed(3);
        const qualityScore = (data.quality_score || 0).toFixed(1);
        const reliabilityIndex = (data.reliability_index || 0).toFixed(3);

        const html = `
            <div class="perfection-widget">
                <h3>Perfection Index</h3>
                <div class="perfection-metrics">
                    <div class="metric-card">
                        <h4>Global Index</h4>
                        <div class="metric-value" style="color: ${this.getIndexColor(globalIndex)};">
                            ${globalIndex}
                        </div>
                        <div class="metric-scale">
                            <div class="scale-fill" style="width: ${globalIndex * 100}%;"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <h4>Quality Score</h4>
                        <div class="metric-value">${qualityScore}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Reliability</h4>
                        <div class="metric-value">${reliabilityIndex}</div>
                    </div>
                </div>
                <div class="tool-perfection">
                    <h4>Tool Metrics</h4>
                    ${Object.entries(data.tool_metrics || {}).map(([tool, metrics]) => `
                        <div class="tool-metric-row">
                            <span class="tool-name">${tool}</span>
                            <span class="perfection-score" style="color: ${this.getIndexColor(metrics.perfection_index)};">
                                ${metrics.perfection_index.toFixed(3)}
                            </span>
                            <span class="quality">${metrics.quality_score.toFixed(0)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    displayAnomalies(anomalies) {
        const container = document.querySelector('[data-widget="anomalies"]');
        if (!container) return;

        const html = `
            <div class="anomalies-widget">
                <h3>Detected Anomalies (${anomalies.length})</h3>
                <div class="anomalies-list">
                    ${anomalies.length === 0 ? '<p>No anomalies detected</p>' : anomalies.map(a => `
                        <div class="anomaly-item severity-${Math.round(a.severity * 3)}">
                            <div class="anomaly-header">
                                <span class="tool-name">${a.tool_name}</span>
                                <span class="anomaly-type">${a.anomaly_type}</span>
                                <span class="severity-badge" style="opacity: ${a.severity};">
                                    ${(a.severity * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div class="anomaly-details">
                                ${JSON.stringify(a.details).substring(0, 100)}...
                            </div>
                            <div class="suggested-fixes">
                                ${a.suggested_fixes.map(f => `<span class="fix-tag">${f}</span>`).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    displayRemediationActions(actions) {
        const container = document.querySelector('[data-widget="remediation"]');
        if (!container) return;

        const html = `
            <div class="remediation-widget">
                <h3>Recent Remediation Actions (${actions.length})</h3>
                <div class="actions-list">
                    ${actions.map(a => `
                        <div class="action-item result-${a.result}">
                            <div class="action-header">
                                <span class="tool-name">${a.tool_name}</span>
                                <span class="action-type">${a.action}</span>
                                <span class="result-badge">${a.result}</span>
                            </div>
                            <div class="action-reason">${a.reason}</div>
                            <div class="action-time">${new Date(a.timestamp * 1000).toLocaleString()}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    renderTimeline(timeline) {
        const container = document.querySelector('[data-widget="health-timeline"]');
        if (!container) return;

        const html = `
            <div class="timeline-widget">
                <h3>Health Timeline</h3>
                <div class="timeline">
                    ${timeline.slice(-20).map((event, idx) => `
                        <div class="timeline-event event-type-${event.event_type}">
                            <div class="timeline-marker"></div>
                            <div class="timeline-content">
                                <span class="event-tool">${event.tool}</span>
                                <span class="event-type">${event.event_type}</span>
                                <span class="event-time">${new Date(event.timestamp * 1000).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    getHealthColor(percentage) {
        if (percentage >= 80) return '#4CAF50'; // Green
        if (percentage >= 60) return '#FFC107'; // Yellow
        if (percentage >= 40) return '#FF9800'; // Orange
        return '#F44336'; // Red
    }

    getIndexColor(value) {
        const numValue = parseFloat(value);
        if (numValue >= 0.8) return '#4CAF50';
        if (numValue >= 0.6) return '#FFC107';
        if (numValue >= 0.4) return '#FF9800';
        return '#F44336';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new PerfectionDashboard();
});
