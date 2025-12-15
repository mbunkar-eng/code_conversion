"""
Pytest configuration and fixtures.
"""

import os
import pytest
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment
os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture(scope="session")
def mock_mode():
    """Ensure mock mode is enabled for all tests."""
    os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"
    yield
    # Cleanup if needed


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

def main():
    nums = [1, 2, 3, 4, 5]
    result = calculate_sum(nums)
    print(f"Sum: {result}")

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
function calculateSum(numbers) {
    let total = 0;
    for (const num of numbers) {
        total += num;
    }
    return total;
}

function main() {
    const nums = [1, 2, 3, 4, 5];
    const result = calculateSum(nums);
    console.log(`Sum: ${result}`);
}

main();
'''


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful code assistant."},
        {"role": "user", "content": "Write a hello world function."},
        {"role": "assistant", "content": "Here's a hello world function in Python:\n```python\ndef hello():\n    print('Hello, World!')\n```"},
        {"role": "user", "content": "Now convert it to Java."}
    ]


@pytest.fixture
def code_conversion_schema():
    """JSON schema for code conversion responses."""
    return {
        "type": "object",
        "properties": {
            "convertedCode": {
                "type": "string",
                "description": "The converted source code"
            },
            "sourceLanguage": {
                "type": "string",
                "description": "Original programming language"
            },
            "targetLanguage": {
                "type": "string",
                "description": "Target programming language"
            },
            "explanation": {
                "type": "string",
                "description": "Explanation of the conversion"
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any warnings about the conversion"
            }
        },
        "required": ["convertedCode", "targetLanguage"]
    }
