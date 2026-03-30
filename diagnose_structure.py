#!/usr/bin/env python3
"""
Diagnostic tool to inspect actual message structure from API requests.
Run this first to confirm the real data format before applying fixes.
"""

import asyncio
import json
import httpx
from typing import Any, Dict

def extract_content(content: Any) -> str:
    """Extract text content from various nested structures."""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        # Extract content from nested structure (e.g., {"type": "text", "content": [...]})
        inner_content = content.get("content")
        if isinstance(inner_content, str):
            return inner_content
        else:
            # Handle list or other types of content
            items = inner_content if inner_content is not None else []
            return " ".join(str(item) for item in items)
    elif content is None:
        return ""
    else:
        # Fallback for any other format
        return str(content)

def analyze_messages(messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze message structure and detect problematic formats."""
    issues = []
    sample_formats = {}
    
    for i, msg in enumerate(messages):
        content = msg.get("content")
        role = msg.get("role", "unknown")
        
        # Check if content is a string
        if isinstance(content, str):
            continue  # Normal case
        elif isinstance(content, dict):
            inner_content = content.get("content")
            sample_formats[f"msg_{i}_dict"] = {
                "role": role,
                "outer_type": type(content).__name__,
                "inner_type": type(inner_content).__name__ if inner_content is not None else None,
                "raw_inner": str(inner_content)[:200] if inner_content is not None else None
            }
        elif isinstance(content, list):
            sample_formats[f"msg_{i}_list"] = {
                "role": role,
                "outer_type": type(content).__name__,
                "inner_items": content[:5]  # First few items
            }
        else:
            issues.append({
                "index": i,
                "role": role,
                "content_type": type(content).__name__,
                "raw_content": str(content)[:200]
            })
    
    return {
        "total_messages": len(messages),
        "issues": issues,
        "sample_formats": sample_formats
    }

async def main():
    # Connect to LM Studio directly (bypassing proxy)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get("http://192.168.1.12:1234/v1/models")
            if response.status_code == 200:
                models = response.json()
                print(f"Connected to LM Studio, found {len(models)} models\n")
                
                # Try a simple chat request
                body = {
                    "model": "qwen3.5-4b",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "stream": False
                }
                
                response = await client.post("http://192.168.1.12:1234/v1/chat/completions", json=body)
                if response.status_code == 200:
                    result = response.json()
                    messages = result.get("choices", [{}])[0].get("message", {})
                    content = messages.get("content", "")
                    print(f"Response content type: {type(content).__name__}")
                    print(f"Content preview: {str(content)[:200]}\n")
                
                # Try with longer context to trigger compression
                body["messages"] = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello, how are you?"}
                ] + [{"role": "assistant", "content": f"This is message {i}."} for i in range(20)]
                
                response = await client.post("http://192.168.1.12:1234/v1/chat/completions", json=body)
                if response.status_code == 200:
                    result = response.json()
                    messages = result.get("choices", [{}])[0].get("message", {})
                    content = messages.get("content", "")
                    print(f"Response content type: {type(content).__name__}")
                    print(f"Content preview: {str(content)[:200]}\n")
                
            else:
                print(f"Failed to connect: status {response.status_code}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # If we can't reach LM Studio, show expected formats
    print("=" * 60)
    print("EXPECTED MESSAGE FORMATS THAT CAUSE ISSUES:")
    print("=" * 60)
    print(json.dumps([
        {"type": "text", "content": ["Hello", "World"]},
        {"content": "Plain string"},
        "Plain string"
    ], indent=2))

async def main_async():
    # If we can't reach LM Studio, show expected formats
    print("=" * 60)
    print("EXPECTED MESSAGE FORMATS THAT CAUSE ISSUES:")
    print("=" * 60)
    print(json.dumps([
        {"type": "text", "content": ["Hello", "World"]},
        {"content": "Plain string"},
        "Plain string"
    ], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
