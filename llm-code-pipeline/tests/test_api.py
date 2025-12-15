"""
Tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import os

# Set mock mode before importing app
os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "gpu_available" in data

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/live")
        assert response.status_code == 200

        data = response.json()
        assert data["alive"] is True

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "LLM Code Pipeline API"
        assert "version" in data


class TestModelsEndpoints:
    """Tests for model listing endpoints."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_model(self, client):
        """Test getting a specific model."""
        response = client.get("/v1/models/qwen2.5-coder-7b")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "qwen2.5-coder-7b"
        assert data["object"] == "model"

    def test_get_nonexistent_model(self, client):
        """Test getting a non-existent model."""
        response = client.get("/v1/models/nonexistent-model")
        assert response.status_code == 404


class TestChatCompletions:
    """Tests for chat completions endpoint."""

    def test_chat_completion_basic(self, client):
        """Test basic chat completion."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ]
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "usage" in data

    def test_chat_completion_with_system_message(self, client):
        """Test chat completion with system message."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        assert response.status_code == 200

    def test_chat_completion_with_parameters(self, client):
        """Test chat completion with generation parameters."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Write code"}
                ],
                "temperature": 0.5,
                "max_tokens": 100,
                "top_p": 0.9
            }
        )
        assert response.status_code == 200

    def test_chat_completion_json_mode(self, client):
        """Test chat completion with JSON mode."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Return JSON with 'message' field"}
                ],
                "response_format": {"type": "json_object"}
            }
        )
        assert response.status_code == 200

    def test_chat_completion_empty_messages(self, client):
        """Test chat completion with empty messages."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": []
            }
        )
        assert response.status_code == 400

    def test_chat_completion_missing_model(self, client):
        """Test chat completion without model."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        assert response.status_code == 422  # Validation error

    def test_chat_completion_invalid_temperature(self, client):
        """Test chat completion with invalid temperature."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 3.0  # Invalid: > 2.0
            }
        )
        assert response.status_code == 422


class TestCompletions:
    """Tests for completions endpoint."""

    def test_completion_basic(self, client):
        """Test basic completion."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "prompt": "def hello():"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "text_completion"
        assert "choices" in data
        assert "text" in data["choices"][0]
        assert "usage" in data

    def test_completion_with_parameters(self, client):
        """Test completion with parameters."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "prompt": "function add(a, b) {",
                "max_tokens": 50,
                "temperature": 0.3,
                "stop": ["}"]
            }
        )
        assert response.status_code == 200

    def test_completion_batch(self, client):
        """Test batch completion."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "prompt": ["def foo():", "def bar():"],
                "max_tokens": 50
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 2


class TestCodeConversion:
    """Tests for code conversion use cases."""

    def test_python_to_java_conversion(self, client):
        """Test Python to Java code conversion."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code conversion expert. Convert Python code to Java."
                    },
                    {
                        "role": "user",
                        "content": "Convert this Python code to Java:\ndef greet(name):\n    print(f'Hello, {name}!')"
                    }
                ],
                "temperature": 0.2
            }
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        # Mock should return Java-like code
        assert len(content) > 0

    def test_javascript_to_go_conversion(self, client):
        """Test JavaScript to Go code conversion."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Convert JavaScript to Go:\nfunction sum(a, b) { return a + b; }"
                    }
                ]
            }
        )
        assert response.status_code == 200


class TestAuthentication:
    """Tests for API authentication."""

    def test_no_auth_required_in_mock(self, client):
        """Test that no auth is required in mock mode without keys."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_auth_with_bearer_token(self, client):
        """Test authentication with bearer token."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        # Should work in mock mode
        assert response.status_code == 200

    def test_auth_with_x_api_key(self, client):
        """Test authentication with X-API-Key header."""
        response = client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "test-key"},
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/v1/chat/completions",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/v1/chat/completions",
            json={}
        )
        assert response.status_code == 422

    def test_error_response_format(self, client):
        """Test that errors follow OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []}  # Missing model
        )
        assert response.status_code == 422

        # Should have error object
        data = response.json()
        assert "detail" in data or "error" in data
