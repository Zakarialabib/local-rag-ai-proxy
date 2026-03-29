#!/usr/bin/env python
"""Test rate limiter functionality."""

from webapp.tool_ecosystem import tool_health_monitor, ToolMetrics, ToolCategory
import time

def test_rate_limiter():
    print("Testing rate limiter implementation...")
    
    # Get rate limiter
    rl = tool_health_monitor.rate_limiter
    
    # Create test metrics
    metrics = ToolMetrics(name="test_tool", category=ToolCategory.EXECUTION)
    
    # Test 1: Initial state - should have tokens
    print("\n1. Initial state:")
    print(f"   Tokens: {metrics.rate_limiter_tokens:.2f}")
    print(f"   Threshold: {rl.tokens_per_minute}")
    
    # Test 2: Consume tokens
    print("\n2. Consuming 1 token at a time:")
    for i in range(5):
        result = rl.consume(metrics, required=1.0)
        print(f"   Token #{i+1}: consumed={result}, remaining={metrics.rate_limiter_tokens:.2f}")
    
    # Test 3: Try to consume more than available
    print("\n3. Attempt to consume when low on tokens:")
    # Drain tokens
    metrics.rate_limiter_tokens = 0.5
    for i in range(3):
        result = rl.consume(metrics, required=1.0)
        print(f"   Attempt #{i+1}: consumed={result}, tokens={metrics.rate_limiter_tokens:.2f}")
    
    # Test 4: Token refill over time
    print("\n4. Token refill over simulated time:")
    metrics.rate_limiter_tokens = 0.0
    metrics.rate_limiter_last_refill = time.time() - 30  # 30 seconds ago
    print(f"   Before refill: tokens={metrics.rate_limiter_tokens:.2f}")
    rl.refill(metrics)
    print(f"   After refill: tokens={metrics.rate_limiter_tokens:.2f} (30 sec passed)")
    
    # Test 5: Max token cap
    print("\n5. Token cap enforcement:")
    metrics.rate_limiter_tokens = 15.0
    metrics.rate_limiter_last_refill = time.time() - 10
    print(f"   Before refill: tokens={metrics.rate_limiter_tokens:.2f}")
    rl.refill(metrics)
    print(f"   After refill: tokens={metrics.rate_limiter_tokens:.2f} (capped at {rl.tokens_per_minute})")
    
    print("\n✓ Rate limiter test completed!")

if __name__ == '__main__':
    print("\n=== RATE LIMITER TEST ===\n")
    test_rate_limiter()
    print("\n=== TEST COMPLETE ===\n")
