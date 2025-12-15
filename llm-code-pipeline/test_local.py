#!/usr/bin/env python3
"""
Local testing script for LLM Code Pipeline API.
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def print_test(name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")

def test_health():
    """Test health endpoint."""
    try:
        resp = requests.get(f"{BASE_URL}/health")
        data = resp.json()
        passed = resp.status_code == 200 and data["status"] == "healthy"
        print_test("Health Check", passed, f"Status: {data.get('status')}, Model: {data.get('model_name')}")
        return passed
    except Exception as e:
        print_test("Health Check", False, str(e))
        return False

def test_list_models():
    """Test listing models."""
    try:
        resp = requests.get(f"{BASE_URL}/v1/models")
        data = resp.json()
        passed = resp.status_code == 200 and len(data.get("data", [])) > 0
        print_test("List Models", passed, f"Found {len(data.get('data', []))} models")
        return passed
    except Exception as e:
        print_test("List Models", False, str(e))
        return False

def test_chat_completion():
    """Test basic chat completion."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ]
            }
        )
        data = resp.json()
        passed = resp.status_code == 200 and len(data.get("choices", [])) > 0
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
        print_test("Chat Completion", passed, f"Response: {content}...")
        return passed
    except Exception as e:
        print_test("Chat Completion", False, str(e))
        return False

def test_code_conversion():
    """Test code conversion (Python to Java)."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "system", "content": "You are a code conversion expert."},
                    {"role": "user", "content": "Convert this Python code to Java: def hello(): print('Hello')"}
                ],
                "temperature": 0.2
            }
        )
        data = resp.json()
        passed = resp.status_code == 200 and len(data.get("choices", [])) > 0
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80]
        print_test("Code Conversion (Python->Java)", passed, f"Response: {content}...")
        return passed
    except Exception as e:
        print_test("Code Conversion", False, str(e))
        return False

def test_json_mode():
    """Test JSON mode response format."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Return a JSON object with a message field"}
                ],
                "response_format": {"type": "json_object"}
            }
        )
        data = resp.json()
        passed = resp.status_code == 200
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
        print_test("JSON Mode", passed, f"Response: {content}...")
        return passed
    except Exception as e:
        print_test("JSON Mode", False, str(e))
        return False

def test_completions():
    """Test legacy completions API."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "prompt": "def hello():",
                "max_tokens": 100
            }
        )
        data = resp.json()
        passed = resp.status_code == 200 and len(data.get("choices", [])) > 0
        text = data.get("choices", [{}])[0].get("text", "")[:50]
        print_test("Completions API", passed, f"Response: {text}...")
        return passed
    except Exception as e:
        print_test("Completions API", False, str(e))
        return False

def test_with_parameters():
    """Test chat with various parameters."""
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Write code"}
                ],
                "temperature": 0.5,
                "max_tokens": 50,
                "top_p": 0.9
            }
        )
        data = resp.json()
        passed = resp.status_code == 200
        usage = data.get("usage", {})
        print_test("Chat with Parameters", passed, f"Tokens: {usage.get('total_tokens', 0)}")
        return passed
    except Exception as e:
        print_test("Chat with Parameters", False, str(e))
        return False

def test_error_handling():
    """Test error handling."""
    try:
        # Test missing model
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        passed = resp.status_code == 422  # Validation error
        print_test("Error Handling (missing model)", passed, f"Status: {resp.status_code}")
        return passed
    except Exception as e:
        print_test("Error Handling", False, str(e))
        return False

def main():
    print("=" * 60)
    print("LLM Code Pipeline - Local API Tests")
    print("=" * 60)
    print()

    tests = [
        test_health,
        test_list_models,
        test_chat_completion,
        test_code_conversion,
        test_json_mode,
        test_completions,
        test_with_parameters,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_test(test.__name__, False, str(e))
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
