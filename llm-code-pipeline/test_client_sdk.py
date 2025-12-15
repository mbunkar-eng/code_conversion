#!/usr/bin/env python3
"""
Test the Python Client SDK against the running server.
"""

import sys
sys.path.insert(0, '.')

from clients.python.client import LLMClient, ChatMessage

def main():
    print("=" * 60)
    print("Testing Python Client SDK")
    print("=" * 60)
    print()

    # Create client
    client = LLMClient(base_url="http://127.0.0.1:8000")

    # Test 1: Health check
    print("[Test 1] Health Check...")
    try:
        resp = client._request("GET", "/health")
        print(f"  Status: {resp.get('status')}")
        print(f"  Model: {resp.get('model_name')}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 2: List models
    print("[Test 2] List Models...")
    try:
        models = client.models.list()
        print(f"  Found {len(models)} models")
        print(f"  First model: {models[0]['id'] if models else 'None'}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 3: Chat completion
    print("[Test 3] Chat Completion...")
    try:
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )
        print(f"  Response: {response.choices[0].message.content[:50]}...")
        print(f"  Tokens: {response.usage.total_tokens}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 4: Chat with ChatMessage objects
    print("[Test 4] Chat with ChatMessage objects...")
    try:
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[
                ChatMessage(role="system", content="You are a code expert."),
                ChatMessage(role="user", content="Write a hello function")
            ],
            temperature=0.5,
            max_tokens=100
        )
        print(f"  Response: {response.choices[0].message.content[:50]}...")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 5: Code conversion
    print("[Test 5] Code Conversion (Python -> Java)...")
    try:
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[
                {"role": "system", "content": "You are a code converter."},
                {"role": "user", "content": "Convert Python to Java: def greet(name): print(f'Hello {name}')"}
            ],
            temperature=0.2
        )
        print(f"  Response: {response.choices[0].message.content[:60]}...")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 6: JSON mode
    print("[Test 6] JSON Mode...")
    try:
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[
                {"role": "user", "content": "Return JSON with a message field"}
            ],
            response_format={"type": "json_object"}
        )
        print(f"  Response: {response.choices[0].message.content[:50]}...")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    # Test 7: Legacy completions
    print("[Test 7] Legacy Completions API...")
    try:
        response = client.completions.create(
            model="qwen2.5-coder-7b",
            prompt="def fibonacci(n):",
            max_tokens=100,
            temperature=0.3
        )
        print(f"  Response: {response.choices[0].text[:50]}...")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
    print()

    print("=" * 60)
    print("Client SDK Tests Complete!")
    print("=" * 60)

    client.close()

if __name__ == "__main__":
    main()
