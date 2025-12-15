"""
Tests for the inference module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestVLLMRunner:
    """Tests for VLLMRunner class."""

    def test_mock_runner_initialization(self):
        """Test mock runner initialization."""
        from inference.vllm_runner import MockVLLMRunner

        runner = MockVLLMRunner(model_path="test-model")
        runner.initialize()

        assert runner._initialized is True

    def test_mock_runner_generate(self):
        """Test mock runner generation."""
        from inference.vllm_runner import MockVLLMRunner, GenerationConfig

        runner = MockVLLMRunner(model_path="test-model")
        runner.initialize()

        config = GenerationConfig(max_tokens=100, temperature=0.7)
        result = runner.generate("Write a function", config)

        assert result.text is not None
        assert result.finish_reason == "stop"
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0

    def test_mock_runner_batch_generate(self):
        """Test mock runner batch generation."""
        from inference.vllm_runner import MockVLLMRunner

        runner = MockVLLMRunner(model_path="test-model")
        runner.initialize()

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = runner.generate(prompts)

        assert len(results) == 3
        for result in results:
            assert result.text is not None

    def test_mock_runner_code_detection(self):
        """Test mock runner detects code conversion requests."""
        from inference.vllm_runner import MockVLLMRunner

        runner = MockVLLMRunner(model_path="test-model")
        runner.initialize()

        # Python to Java
        result = runner.generate("Convert Python code to Java")
        assert "java" in result.text.lower() or "class" in result.text.lower()

        # JavaScript/TypeScript
        result = runner.generate("Convert to TypeScript")
        assert "function" in result.text.lower() or "export" in result.text.lower()

    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        from inference.vllm_runner import GenerationConfig

        config = GenerationConfig()

        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.stream is False

    def test_generation_config_custom(self):
        """Test GenerationConfig custom values."""
        from inference.vllm_runner import GenerationConfig

        config = GenerationConfig(
            max_tokens=500,
            temperature=0.2,
            top_p=0.9,
            stop=["```", "\n\n"]
        )

        assert config.max_tokens == 500
        assert config.temperature == 0.2
        assert config.stop == ["```", "\n\n"]

    def test_runner_get_model_info_not_initialized(self):
        """Test getting model info before initialization."""
        from inference.vllm_runner import MockVLLMRunner

        runner = MockVLLMRunner(model_path="test-model")
        info = runner.get_model_info()

        assert info["status"] == "not_initialized"

    def test_runner_get_model_info_initialized(self):
        """Test getting model info after initialization."""
        from inference.vllm_runner import MockVLLMRunner

        runner = MockVLLMRunner(model_path="test-model")
        runner.initialize()
        info = runner.get_model_info()

        assert info["status"] == "initialized"
        assert info["model_path"] == "test-model"


class TestTokenizerService:
    """Tests for TokenizerService class."""

    def test_mock_tokenizer_initialization(self):
        """Test mock tokenizer initialization."""
        from inference.tokenizer_service import MockTokenizerService

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        assert tokenizer._initialized is True

    def test_mock_tokenizer_encode(self):
        """Test mock tokenizer encoding."""
        from inference.tokenizer_service import MockTokenizerService

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        result = tokenizer.encode("Hello, world!")

        assert result.tokens is not None
        assert result.token_count > 0
        assert result.text == "Hello, world!"

    def test_mock_tokenizer_decode(self):
        """Test mock tokenizer decoding."""
        from inference.tokenizer_service import MockTokenizerService

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        result = tokenizer.decode([1, 2, 3, 4, 5])

        assert "5 tokens" in result

    def test_mock_tokenizer_count_tokens(self):
        """Test mock tokenizer token counting."""
        from inference.tokenizer_service import MockTokenizerService

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        count = tokenizer.count_tokens("This is a test sentence.")

        assert count > 0

    def test_mock_tokenizer_chat_template(self):
        """Test mock tokenizer chat template."""
        from inference.tokenizer_service import MockTokenizerService, ChatMessage

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello")
        ]

        result = tokenizer.apply_chat_template(messages)

        assert "System:" in result
        assert "User:" in result
        assert "You are helpful." in result
        assert "Hello" in result

    def test_mock_tokenizer_special_tokens(self):
        """Test mock tokenizer special tokens."""
        from inference.tokenizer_service import MockTokenizerService

        tokenizer = MockTokenizerService(model_path="test-model")
        tokenizer.initialize()

        tokens = tokenizer.get_special_tokens()

        assert "bos_token" in tokens
        assert "eos_token" in tokens
        assert "vocab_size" in tokens


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_extract_json_from_text(self):
        """Test extracting JSON from plain text."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = '{"name": "test", "value": 42}'
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is True
        assert result.data["name"] == "test"
        assert result.data["value"] == 42

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = '''Here is the JSON:
```json
{"code": "print('hello')", "language": "python"}
```
'''
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is True
        assert result.data["language"] == "python"

    def test_extract_json_from_tags(self):
        """Test extracting JSON from custom tags."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = '''Some text before
<json_output>
{"result": "success", "count": 10}
</json_output>
Some text after'''

        result = formatter.extract_json(text, JSONExtractionMode.TAGGED)

        assert result.success is True
        assert result.data["result"] == "success"

    def test_lenient_mode_fixes_trailing_commas(self):
        """Test lenient mode fixes trailing commas."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = '{"name": "test", "items": [1, 2, 3,],}'  # Trailing commas
        result = formatter.extract_json(text, JSONExtractionMode.LENIENT)

        assert result.success is True
        assert result.data["name"] == "test"

    def test_lenient_mode_fixes_python_booleans(self):
        """Test lenient mode fixes Python-style booleans."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = '{"active": True, "deleted": False, "value": None}'
        result = formatter.extract_json(text, JSONExtractionMode.LENIENT)

        assert result.success is True
        assert result.data["active"] is True
        assert result.data["deleted"] is False
        assert result.data["value"] is None

    def test_strict_mode_fails_invalid_json(self):
        """Test strict mode fails on invalid JSON."""
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()

        text = "This is not JSON at all"
        result = formatter.extract_json(text, JSONExtractionMode.STRICT)

        assert result.success is False
        assert result.error is not None

    def test_format_for_output(self):
        """Test formatting data for output."""
        from inference.json_formatter import JSONFormatter

        formatter = JSONFormatter()
        data = {"code": "hello", "lang": "python"}

        result = formatter.format_for_output(data)
        assert '"code"' in result
        assert '"python"' in result

    def test_format_for_output_with_tags(self):
        """Test formatting data with tags."""
        from inference.json_formatter import JSONFormatter

        formatter = JSONFormatter()
        data = {"test": True}

        result = formatter.format_for_output(data, wrap_in_tags=True)
        assert "<json_output>" in result
        assert "</json_output>" in result

    def test_create_json_prompt_suffix(self):
        """Test creating JSON prompt suffix."""
        from inference.json_formatter import JSONFormatter

        formatter = JSONFormatter()
        suffix = formatter.create_json_prompt_suffix()

        assert "JSON" in suffix
        assert "json_output" in suffix

    def test_create_json_prompt_suffix_with_schema(self):
        """Test creating JSON prompt suffix with schema."""
        from inference.json_formatter import JSONFormatter

        formatter = JSONFormatter()
        schema = {
            "type": "object",
            "properties": {
                "code": {"type": "string"}
            }
        }

        suffix = formatter.create_json_prompt_suffix(schema=schema)

        assert "schema" in suffix.lower()
        assert "code" in suffix


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_generation_result_creation(self):
        """Test creating GenerationResult."""
        from inference.vllm_runner import GenerationResult

        result = GenerationResult(
            text="Generated text",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time_ms=100.0,
            tokens_per_second=200.0
        )

        assert result.text == "Generated text"
        assert result.total_tokens == 30
        assert result.tokens_per_second == 200.0
