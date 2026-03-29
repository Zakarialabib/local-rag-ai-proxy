#!/usr/bin/env python
"""Test perfection index tracking with simulated tool execution."""

from webapp.perfection_index import perfection_tracker
import time

def test_perfection_tracking():
    # Simulate some tool executions
    test_cases = [
        ("grep_search", 12.5, True),
        ("file_read", 45.2, True),
        ("file_write", 38.1, True),
        ("grep_search", 14.1, True),
        ("grep_search", 11.9, True),
        ("file_read", 200.0, False),  # Slow/failed
        ("grep_search", 13.5, True),
        ("code_analysis", 89.4, True),
        ("code_analysis", 91.2, True),
        ("file_write", 42.0, False),  # Failed
    ]
    
    print("Recording tool executions...")
    for tool, latency, success in test_cases:
        perfection_tracker.record_tool_execution(
            tool_name=tool,
            latency_ms=latency,
            success=success,
            session_id="test_session_001",
            agent="test_agent",
            error="timeout" if not success else None
        )
        print(f"  ✓ {tool:20} {latency:6.1f}ms {'✓' if success else '✗'}")
    
    print("\n" + "="*60)
    print("Calculating indices...")
    print("="*60)
    
    # Get indices for each tool
    for tool in ["grep_search", "file_read", "file_write", "code_analysis"]:
        indices = perfection_tracker.calculate_tool_indices(tool, window_minutes=60)
        print(f"\n{tool}:")
        print(f"  Quality Score:       {indices['quality_score']:.2f}")
        print(f"  Reliability Index:   {indices['reliability_index']:.4f}")
        print(f"  Perfection Index:    {indices['perfection_index']:.4f}")
        print(f"  Calls:               {indices['calls']}")
        print(f"  Successes:           {indices['successes']}")
        print(f"  Failures:            {indices['failures']}")
        print(f"  Avg Latency:         {indices['avg_latency_ms']:.2f}ms")
    
    print("\n" + "="*60)
    print("System Health Summary:")
    print("="*60)
    summary = perfection_tracker.get_system_health_summary()
    print(f"Global Perfection Index: {summary['perfection_index']:.4f}")
    print(f"Average Quality Score:   {summary['quality_score']:.2f}")
    print(f"Average Reliability:     {summary['reliability_index']:.4f}")
    print(f"Total Velocity:          {summary['velocity']:.4f} calls/min")
    print(f"Tools tracked:           {len(summary['tools'])}")
    
    print("\n✓ Perfection tracking test completed!")

if __name__ == '__main__':
    print("\n=== PERFECTION INDEX TRACKING TEST ===\n")
    test_perfection_tracking()
    print("\n=== TEST COMPLETE ===\n")
