#!/usr/bin/env python
"""Test analytics, anomaly detection, and remediation."""

from webapp.tool_analytics import anomaly_detector, analytics_store, remediation_engine, RemediationAction

def test_analytics():
    print("Testing analytics, anomaly detection, and remediation...")
    
    # Simulate tool executions
    print("\n1. Recording tool executions:")
    tool_name = "problematic_tool"
    
    # Good executions
    for i in range(5):
        anomaly_detector.record_execution(tool_name, 50.0, error=False)
        print(f"   ✓ Normal execution #{i+1} (50ms)")
    
    # Add some slow executions
    for i in range(3):
        anomaly_detector.record_execution(tool_name, 800.0, error=False)
        print(f"   ⚠ Slow execution #{i+1} (800ms)")
    
    # Add errors
    for i in range(2):
        anomaly_detector.record_execution(tool_name, 100.0, error=True)
        print(f"   ✗ Error execution #{i+1}")
    
    # Detect anomalies
    print("\n2. Detecting anomalies:")
    anomalies = anomaly_detector.detect_anomalies(tool_name)
    print(f"   Found {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"     - {anomaly.anomaly_type.value}: severity={anomaly.severity:.2f}")
        print(f"       Details: {anomaly.details}")
        print(f"       Suggested fixes: {[a.value for a in anomaly.suggested_fixes]}")
        
        # Log the anomaly
        analytics_store.log_anomaly(anomaly)
    
    # Get tool history
    print("\n3. Tool history:")
    history = analytics_store.get_tool_history(tool_name, limit=20)
    print(f"   Lifecycle events: {len(history['lifecycle'])}")
    print(f"   Anomalies logged: {len(history['anomalies'])}")
    print(f"   Remediation actions: {len(history['remediation'])}")
    
    # Trigger remediation
    print("\n4. Triggering remediation:")
    def handle_isolate(tool):
        return f"Isolated {tool}"
    
    def handle_log_issue(tool):
        return f"Logged issue for {tool}"
    
    remediation_engine.register_action_callback(RemediationAction.ISOLATE_TOOL, handle_isolate)
    remediation_engine.register_action_callback(RemediationAction.LOG_ISSUE, handle_log_issue)
    
    actions = remediation_engine.trigger_remediation(tool_name, anomalies)
    print(f"   Triggered {len(actions)} remediation actions:")
    for action in actions:
        print(f"     - {action.action.value}: {action.result}")
    
    # Get pending actions
    print("\n5. Pending remediation actions:")
    pending = remediation_engine.get_pending_actions(limit=10)
    for action in pending:
        print(f"   - {action['action']}: {action['result']}")
    
    print("\n✓ Analytics test completed!")

if __name__ == '__main__':
    print("\n=== ANALYTICS & REMEDIATION TEST ===\n")
    test_analytics()
    print("\n=== TEST COMPLETE ===\n")
