#!/usr/bin/env python
"""Test circuit breaker functionality."""

from webapp.tool_ecosystem import tool_health_monitor

def test_circuit_breaker():
    print("Testing circuit breaker implementation...")
    
    # Get circuit breaker
    cb = tool_health_monitor.circuit_breaker
    
    # Test 1: Tool starts healthy
    print("\n1. Initial state - healthy tool:")
    health = tool_health_monitor.get_tool_status("test_tool_1")
    print(f"   Status: {health.get('status')}")
    print(f"   Circuit open: {cb.is_open('test_tool_1')}")
    
    # Test 2: Record failures to trigger circuit
    print("\n2. Recording 5 failures to trigger circuit:")
    for i in range(5):
        cb.record_failure("test_tool_1")
        is_open = cb.is_open("test_tool_1")
        print(f"   Failure #{i+1}: circuit_open={is_open}")
    
    # Test 3: Check tool is now in isolated state
    print("\n3. Tool is now isolated:")
    health = tool_health_monitor.get_tool_status("test_tool_1")
    print(f"   Status: {health.get('status')}")
    print(f"   Circuit open: {cb.is_open('test_tool_1')}")
    
    # Test 4: Record a success to reset
    print("\n4. Recording success resets circuit:")
    cb.record_success("test_tool_1")
    is_open = cb.is_open("test_tool_1")
    health = tool_health_monitor.get_tool_status("test_tool_1")
    print(f"   Circuit open: {is_open}")
    print(f"   Status: {health.get('status')}")
    
    # Test 5: Test timeout recovery
    print("\n5. Testing timeout recovery:")
    cb_timeout = tool_health_monitor.circuit_breaker
    # Manually trigger isolation
    for _ in range(5):
        cb_timeout.record_failure("test_tool_2")
    print(f"   Circuit open before timeout: {cb_timeout.is_open('test_tool_2')}")
    
    # Simulate time passing by directly modifying opened_at
    import time
    cb_timeout.opened_at["test_tool_2"] = time.time() - 120  # 2 minutes ago
    print(f"   Circuit open after simulated timeout: {cb_timeout.is_open('test_tool_2')}")
    
    print("\n✓ Circuit breaker test completed!")

if __name__ == '__main__':
    print("\n=== CIRCUIT BREAKER TEST ===\n")
    test_circuit_breaker()
    print("\n=== TEST COMPLETE ===\n")
