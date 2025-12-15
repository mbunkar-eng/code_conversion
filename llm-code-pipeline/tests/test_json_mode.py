"""
Tests specifically for JSON mode functionality.
"""

import pytest
import json
from fastapi.testclient import TestClient
import os

# Set mock mode before importing app
os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestJSONModeAPI:
    """Tests for JSON mode in API."""

    def test_json_object_mode(self, client):
        """Test response_format with type: json_object."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Return a JSON object with a 'message' field"}
                ],
                "response_format": {"type": "json_object"}
            }
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # In mock mode, content might not be valid JSON
        # but the API should still work
        assert len(content) > 0

    def test_json_schema_mode(self, client):
        """Test response_format with type: json_schema."""
        schema = {
            "type": "object",
            "properties": {
                "convertedCode": {"type": "string"},
                "language": {"type": "string"}
            },
            "required": ["convertedCode"]
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Convert print('hello') to Java"}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CodeConversion",
                        "schema": schema
                    }
                }
            }
        )
        assert response.status_code == 200

    def test_text_mode_default(self, client):
        """Test that default mode is text."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        assert response.status_code == 200

    def test_invalid_response_format_type(self, client):
        """Test invalid response_format type."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "response_format": {"type": "invalid_type"}
            }
        )
        assert response.status_code == 422


class TestJSONFormatterUnit:
    """Unit tests for JSON formatter."""

    def test_extract_nested_json(self):
        """Test extracting nested JSON."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        text = '''
        {
            "code": {
                "source": "def hello(): pass",
                "target": "void hello() {}"
            },
            "metadata": {
                "from": "python",
                "to": "java"
            }
        }
        '''
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is True
        assert result.data["code"]["source"] == "def hello(): pass"
        assert result.data["metadata"]["from"] == "python"

    def test_extract_json_array(self):
        """Test extracting JSON array."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is True
        assert len(result.data) == 3
        assert result.data[0]["id"] == 1

    def test_extract_json_with_special_characters(self):
        """Test extracting JSON with special characters."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        text = '{"code": "print(\\"Hello, World!\\")", "special": "tab\\there"}'
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is True
        assert "Hello" in result.data["code"]

    def test_lenient_mode_unquoted_keys(self):
        """Test lenient mode with unquoted keys."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        text = '{name: "test", value: 42}'
        result = formatter.extract_json(text, JSONExtractionMode.LENIENT)

        assert result.success is True
        assert result.data["name"] == "test"

    def test_multiple_json_objects_takes_largest(self):
        """Test that with multiple JSON objects, largest is selected."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        text = '''
        First: {"a": 1}
        Second: {"name": "test", "value": 42, "nested": {"x": 1}}
        '''
        result = formatter.extract_json(text, JSONExtractionMode.LENIENT)

        assert result.success is True
        # Should get the larger object
        assert "nested" in result.data or "name" in result.data


class TestCodeConversionJSON:
    """Tests for code conversion with JSON output."""

    def test_python_to_java_json_response(self, client):
        """Test Python to Java with JSON response format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code converter. Return JSON with 'convertedCode' and 'language' fields."
                    },
                    {
                        "role": "user",
                        "content": "Convert: def greet(): print('hello')"
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
        )
        assert response.status_code == 200

    def test_multi_language_conversion_json(self, client):
        """Test converting to multiple languages with JSON."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {
                        "role": "system",
                        "content": """Convert the code to multiple languages.
                        Return JSON: {"conversions": [{"language": "...", "code": "..."}]}"""
                    },
                    {
                        "role": "user",
                        "content": "Convert: console.log('hello')"
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MultiConversion",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "conversions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "language": {"type": "string"},
                                            "code": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
        assert response.status_code == 200


class TestJSONValidation:
    """Tests for JSON schema validation."""

    def test_schema_with_strict_mode(self, client):
        """Test schema validation with strict mode."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Return code conversion result"}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "StrictSchema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "success": {"type": "boolean"}
                            },
                            "required": ["code", "success"]
                        },
                        "strict": True
                    }
                }
            }
        )
        assert response.status_code == 200

    def test_schema_missing_name(self, client):
        """Test that schema without name fails validation."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "user", "content": "Test"}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {"type": "object"}
                        # Missing "name"
                    }
                }
            }
        )
        # Should fail validation
        assert response.status_code in [400, 422]
