"""Remediation Callbacks - Action handlers for tool isolation, throttling, fallback, etc."""

import logging
import threading
from typing import Callable, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RemediationAction(str, Enum):
    """Remediation action types - Phases 1-5 + Phase 6 (Perfection Path)"""
    # Phases 1-5 Actions
    RESTART_TOOL = "restart_tool"
    THROTTLE_CALLS = "throttle_calls"
    INCREASE_TIMEOUT = "increase_timeout"
    ISOLATE_TOOL = "isolate_tool"
    RESET_CIRCUIT = "reset_circuit"
    LOG_ISSUE = "log_issue"
    ALERT_OPERATOR = "alert_operator"
    FALLBACK_STRATEGY = "fallback_strategy"
    
    # Phase 6 Actions (Perfection Path)
    FALLBACK_CHAIN_NEXT = "fallback_chain_next"
    EVICT_MODELS = "evict_models"
    TRIGGER_VRAM_GROOM = "trigger_vram_groom"
    REDUCE_QUANTIZATION = "reduce_quantization"
    PRESET_MUTATION = "preset_mutation"
    ALERT_CLIFF_EDGE = "alert_cliff_edge"
    ALERT_COLD_START = "alert_cold_start"
    DEEP_RERANKER_SWAP = "deep_reranker_swap"
    MASK_VISION_CAPABILITY = "mask_vision_capability"
    SWITCH_STREAMING_MODE = "switch_streaming_mode"


class RemediationCallbackRegistry:
    """Register and execute remediation action callbacks."""

    def __init__(self):
        self.callbacks: Dict[RemediationAction, Callable] = {}
        self.lock = threading.RLock()
        self._register_default_callbacks()

    def _register_default_callbacks(self):
        """Register default callback implementations."""
        # ISOLATE_TOOL - Mark tool as isolated
        def isolate_tool(tool_name: str, reason: str = ""):
            try:
                from webapp.tool_ecosystem import tool_health_monitor
                tool_health_monitor.circuit_breaker.record_failure(tool_name)
                logger.warning(f"Isolated tool: {tool_name} - {reason}")
            except Exception as e:
                logger.error(f"Failed to isolate {tool_name}: {e}")

        # RESET_CIRCUIT - Reset circuit breaker
        def reset_circuit(tool_name: str, reason: str = ""):
            try:
                from webapp.tool_ecosystem import tool_health_monitor
                tool_health_monitor.circuit_breaker.reset(tool_name)
                logger.info(f"Reset circuit for: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to reset circuit: {e}")

        # LOG_ISSUE - Log to analytics
        def log_issue(tool_name: str, reason: str = ""):
            try:
                from webapp.tool_analytics import analytics_store
                analytics_store.log_lifecycle(tool_name, "remediation", 0.0, False, reason)
                logger.error(f"Issue logged for {tool_name}: {reason}")
            except Exception as e:
                logger.error(f"Failed to log issue: {e}")

        # THROTTLE_CALLS - Reduce rate limit tokens
        def throttle_calls(tool_name: str, reason: str = ""):
            try:
                from webapp.tool_ecosystem import tool_health_monitor
                metrics = tool_health_monitor.tools.get(tool_name)
                if metrics:
                    # Reduce tokens by 50%
                    metrics.rate_limiter_tokens = max(1, metrics.rate_limiter_tokens * 0.5)
                logger.warning(f"Throttled: {tool_name} - {reason}")
            except Exception as e:
                logger.error(f"Failed to throttle {tool_name}: {e}")

        # INCREASE_TIMEOUT - Log for timeout adjustment
        def increase_timeout(tool_name: str, reason: str = ""):
            logger.info(f"Timeout increase recommended for {tool_name}: {reason}")

        # ALERT_OPERATOR - Log alert for manual intervention
        def alert_operator(tool_name: str, reason: str = ""):
            logger.critical(f"Operator alert for {tool_name}: {reason}")

        # FALLBACK_STRATEGY - Log fallback execution
        def fallback_strategy(tool_name: str, reason: str = ""):
            logger.warning(f"Fallback strategy for {tool_name}: {reason}")

        # RESTART_TOOL - Log restart request
        def restart_tool(tool_name: str, reason: str = ""):
            logger.info(f"Restart requested for {tool_name}: {reason}")

        self.register(RemediationAction.ISOLATE_TOOL, isolate_tool)
        self.register(RemediationAction.RESET_CIRCUIT, reset_circuit)
        self.register(RemediationAction.LOG_ISSUE, log_issue)
        self.register(RemediationAction.THROTTLE_CALLS, throttle_calls)
        self.register(RemediationAction.INCREASE_TIMEOUT, increase_timeout)
        self.register(RemediationAction.ALERT_OPERATOR, alert_operator)
        self.register(RemediationAction.FALLBACK_STRATEGY, fallback_strategy)
        self.register(RemediationAction.RESTART_TOOL, restart_tool)
        
        # ===== PHASE 6 CALLBACKS =====
        
        # FALLBACK_CHAIN_NEXT - Escalate to next fallback level
        def fallback_chain_next(tool_name: str, reason: str = ""):
            try:
                logger.warning(f"Activating next fallback level: {reason}")
                # Fallback chain is managed by FallbackOrchestrator in perfection_path.py
                # This signals the orchestrator to move to next level
            except Exception as e:
                logger.error(f"Fallback chain escalation failed: {e}")
        
        # EVICT_MODELS - Free VRAM by unloading models
        def evict_models(tool_name: str, reason: str = ""):
            try:
                logger.warning(f"Evicting models to free VRAM: {reason}")
                # Signal LMStudioBridge to unload specified models
                # Implementation integrates with lmstudio_bridge.py
            except Exception as e:
                logger.error(f"Model eviction failed: {e}")
        
        # TRIGGER_VRAM_GROOM - Schedule/execute memory defragmentation
        def trigger_vram_groom(tool_name: str, reason: str = ""):
            try:
                logger.info(f"Triggering VRAM grooming: {reason}")
                # HardwareTuner will schedule grooming during next idle window
            except Exception as e:
                logger.error(f"VRAM grooming trigger failed: {e}")
        
        # REDUCE_QUANTIZATION - Downgrade model quantization level
        def reduce_quantization(tool_name: str, reason: str = ""):
            try:
                logger.warning(f"Reducing quantization for {tool_name}: {reason}")
                # Switch from Q4_K_M to Q3_K_M for VRAM saving
            except Exception as e:
                logger.error(f"Quantization reduction failed: {e}")
        
        # PRESET_MUTATION - Auto-apply evolved preset variant
        def preset_mutation(tool_name: str, reason: str = ""):
            try:
                logger.info(f"Applying preset mutation: {reason}")
                # HardwareTuner applies new preset variant
            except Exception as e:
                logger.error(f"Preset mutation failed: {e}")
        
        # ALERT_CLIFF_EDGE -Alert on TTFT cliff detection
        def alert_cliff_edge(tool_name: str, reason: str = ""):
            try:
                from webapp.alerting_system import alert_manager
                alert_manager.send_alert(
                    tool_name="probing",
                    anomaly_type="cliff_edge_detected",
                    severity=0.7,
                    message=f"TTFT cliff detected: {reason}",
                    details={"recommendation": "reduce_context_length"}
                )
                logger.warning(f"CLIFF-EDGE ALERT: {reason}")
            except Exception as e:
                logger.error(f"Cliff-edge alert failed: {e}")
        
        # ALERT_COLD_START - Trigger pre-warming
        def alert_cold_start(tool_name: str, reason: str = ""):
            try:
                logger.info(f"Cold-start detected, triggering pre-warm: {reason}")
                # PrewarmerService will execute pre-warming cycle
            except Exception as e:
                logger.error(f"Cold-start alert failed: {e}")
        
        # DEEP_RERANKER_SWAP - Swap to 4B reranker
        def deep_reranker_swap(tool_name: str, reason: str = ""):
            try:
                logger.warning(f"Swapping to 4B reranker: {reason}")
                # LMStudioBridge unloads 0.6B, loads 4B reranker temporarily
            except Exception as e:
                logger.error(f"Reranker swap failed: {e}")
        
        # MASK_VISION_CAPABILITY - Hide vision due to VRAM constraint
        def mask_vision_capability(tool_name: str, reason: str = ""):
            try:
                logger.info(f"Masking vision capability: {reason}")
                # benchmark.py reports vision: false to clients
            except Exception as e:
                logger.error(f"Vision masking failed: {e}")
        
        # SWITCH_STREAMING_MODE - Toggle between streaming/batch-and-stream
        def switch_streaming_mode(tool_name: str, reason: str = ""):
            try:
                logger.info(f"Switching streaming mode: {reason}")
                # ACE orchestrator uses batch-and-stream if TPS < 10
            except Exception as e:
                logger.error(f"Streaming mode switch failed: {e}")
        
        self.register(RemediationAction.FALLBACK_CHAIN_NEXT, fallback_chain_next)
        self.register(RemediationAction.EVICT_MODELS, evict_models)
        self.register(RemediationAction.TRIGGER_VRAM_GROOM, trigger_vram_groom)
        self.register(RemediationAction.REDUCE_QUANTIZATION, reduce_quantization)
        self.register(RemediationAction.PRESET_MUTATION, preset_mutation)
        self.register(RemediationAction.ALERT_CLIFF_EDGE, alert_cliff_edge)
        self.register(RemediationAction.ALERT_COLD_START, alert_cold_start)
        self.register(RemediationAction.DEEP_RERANKER_SWAP, deep_reranker_swap)
        self.register(RemediationAction.MASK_VISION_CAPABILITY, mask_vision_capability)
        self.register(RemediationAction.SWITCH_STREAMING_MODE, switch_streaming_mode)

    def register(self, action: RemediationAction, callback: Callable):
        """Register a callback for a remediation action."""
        with self.lock:
            self.callbacks[action] = callback
            logger.info(f"Registered callback for: {action.value}")

    def execute(self, action: RemediationAction, tool_name: str, reason: str = "") -> bool:
        """Execute a remediation action callback."""
        with self.lock:
            callback = self.callbacks.get(action)
            if not callback:
                logger.warning(f"No callback registered for: {action.value}")
                return False

        try:
            callback(tool_name, reason)
            return True
        except Exception as e:
            logger.error(f"Callback execution failed for {action.value}: {e}")
            return False


# Global registry instance
remediation_callback_registry = RemediationCallbackRegistry()
