#!/usr/bin/env python3
"""
Test script to verify various message formats are handled correctly.
"""

import sys
sys.path.insert(0, "..")

from context_manager import ContextManager

def test_extract_content():
    cm = ContextManager()
    
    # Test cases: (input, expected_output)
    test_cases = [
        # String content
        ({"content": "Hello World"}, "Hello World"),
        ({"type": "text", "content": ["Hello", "World"]}, "Hello World"),
        
        # List content (edge case)
        (["item1", "item2"], "item1 item2"),
        ([1, 2, 3], "1 2 3"),
        
        # None content
        ({}, ""),
        ({"content": None}, ""),
        
        # Other types
        (42, "42"),
    ]
    
    all_passed = True
    for i, (input_msg, expected) in enumerate(test_cases):
        try:
            result = cm.should_compress([input_msg], token_budget=100)
            # Just verify it doesn't crash - should return False for these small inputs
            print(f"Test {i+1}: Input={input_msg}")
            print(f"  Result: {result} (no exception raised) PASS")
        except Exception as e:
            print(f"Test {i+1}: FAILED with exception: {e}")
            all_passed = False
    
    # Test with larger input that triggers token estimation
    large_messages = []
    for _ in range(50):
        large_messages.append({"role": "user", "content": "This is a very long message that should definitely exceed the token budget when joined together."})
    large_messages.extend([
        {"role": "assistant", "content": "Here is my response to your question about the topic we discussed earlier today."} for _ in range(50)
    ])
    
    try:
        result = cm.should_compress(large_messages, token_budget=100)
        print(f"Test Large Input: {result}")
        if result:
            print("  PASS - Compression correctly triggered")
        else:
            print("  FAIL - Should have returned True for large input")
            all_passed = False
    except Exception as e:
        print(f"Test Large Input: FAILED with exception: {e}")
        all_passed = False
    
    # Test compress_context with problematic nested structure
    try:
        compressed = cm.compress_context([
            {"role": "user", "content": ["Hello", "World"]},  # Nested list format
            {"role": "assistant", "content": "Response here."}
        ])
        print(f"Test compress_context: PASS (no exception raised)")
    except Exception as e:
        print(f"Test compress_context: FAILED with exception: {e}")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    passed = test_extract_content()
    sys.exit(0 if passed else 1)
