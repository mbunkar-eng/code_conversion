# LLM Code Pipeline

Internal LLM Pipeline for Code Generation & Code Conversion with OpenAI-compatible API.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API
- **Code Generation**: Generate code in multiple programming languages
- **Code Conversion**: Convert code between languages (Python → Java, JS → Go, etc.)
- **JSON Mode**: Structured JSON output with schema validation
- **Streaming**: Server-sent events for real-time responses
- **Multiple Models**: Support for DeepSeek Coder, StarCoder2, CodeLlama, Qwen2.5-Coder
- **GPU Acceleration**: vLLM-powered inference with tensor parallelism
- **Quantization**: INT4 AWQ/GPTQ support for reduced memory usage
- **Web UI**: Streamlit-based interface for easy model management and chatting

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd llm-code-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU inference with vLLM
pip install -r requirements-inference.txt
```

### 2. Run in Mock Mode (No GPU Required)

```bash
# Start the API server in mock mode
LLM_PIPELINE_MOCK_MODE=true python -m api.main

# Or use the CLI
python -m api.main --mock
```

### 3. Run with GPU

```bash
# Set model path and start server
export MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct
python -m api.main --host 0.0.0.0 --port 8000
```

### 4. Using Docker

```bash
# Development (mock mode)
docker-compose --profile mock up

# Production (GPU)
MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct docker-compose --profile gpu up
```

## API Usage

### Chat Completions

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen2.5-coder-7b",
        "messages": [
            {"role": "system", "content": "You are a code conversion expert."},
            {"role": "user", "content": "Convert this Python to Java: def hello(): print('Hello')"}
        ],
        "temperature": 0.2
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### JSON Mode

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen2.5-coder-7b",
        "messages": [
            {"role": "user", "content": "Convert print('hello') to Java"}
        ],
        "response_format": {"type": "json_object"}
    }
)
```

### Using the Python SDK

```python
from clients.python import LLMClient

client = LLMClient(base_url="http://localhost:8000")

# Chat completion
response = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[
        {"role": "user", "content": "Write a hello world in Rust"}
    ]
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Explain quicksort"}],
    stream=True
):
    print(chunk.delta_content, end="")
```

### Using the JavaScript SDK

```javascript
import { LLMClient } from '@llm-pipeline/client';

const client = new LLMClient({
    baseUrl: 'http://localhost:8000'
});

// Chat completion
const response = await client.chat.completions.create({
    model: 'qwen2.5-coder-7b',
    messages: [
        { role: 'user', content: 'Convert Python to TypeScript: def add(a, b): return a + b' }
    ]
});
console.log(response.choices[0].message.content);
```

## Project Structure

```
llm-code-pipeline/
├── api/                    # FastAPI server
│   ├── main.py            # Application entry point
│   ├── routes/            # API endpoints
│   ├── schemas/           # Pydantic models
│   └── utils/             # Utilities
├── models/                 # Model management
│   ├── download.py        # Model downloader
│   ├── convert.py         # Model converter
│   └── registry.json      # Model registry
├── inference/              # Inference engine
│   ├── vllm_runner.py     # vLLM wrapper
│   ├── tokenizer_service.py
│   └── json_formatter.py  # JSON extraction
├── clients/                # Client SDKs
│   ├── python/            # Python client
│   └── js/                # JavaScript client
├── deployment/             # Deployment configs
│   ├── docker/            # Dockerfiles
│   └── kubernetes/        # K8s manifests
├── tests/                  # Test suite
└── config/                 # Configuration
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `MODEL_PATH`: HuggingFace model ID or local path
- `LLM_API_KEY`: API key for authentication
- `TENSOR_PARALLEL_SIZE`: Number of GPUs
- `GPU_MEMORY_UTILIZATION`: GPU memory fraction (0.0-1.0)

## Available Models

| Model | Parameters | Context Length | GPU Memory |
|-------|-----------|----------------|------------|
| qwen2.5-coder-7b | 7B | 131K | 14 GB |
| qwen2.5-coder-14b | 14B | 131K | 28 GB |
| qwen2.5-coder-32b | 32B | 131K | 64 GB |
| deepseek-coder-6.7b | 6.7B | 16K | 14 GB |
| codellama-7b | 7B | 16K | 14 GB |
| starcoder2-7b | 7B | 16K | 14 GB |

## Downloading Models

```bash
# Download a model
python -m models.download qwen2.5-coder-7b -o ./downloaded_models

# List available models
python -m models.download --list
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI docs |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov=inference --cov=models

# Run specific tests
pytest tests/test_api.py -v
```

## Deployment

### Kubernetes

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/api.yaml
kubectl apply -f deployment/kubernetes/inference.yaml
```

### Docker Compose

```bash
# Start with GPU
docker-compose --profile gpu up -d

# Start mock mode
docker-compose --profile mock up -d

# Start with monitoring
docker-compose --profile gpu --profile monitoring up -d
```

## Streamlit Web UI

The project includes a beautiful Streamlit web interface for easy model management and chatting.

### Features
- **Model Management**: Download models from HuggingFace
- **Chat Interface**: Interactive chat with downloaded models
- **API Integration**: Uses the OpenAI-compatible API
- **Real-time Responses**: Streaming responses from models

### Running the UI

#### Option 1: Docker (Recommended)
```bash
# Start API + UI
cd deployment
docker-compose --profile ui up

# Access UI at: http://localhost:8501
# Access API at: http://localhost:8000
```

#### Option 2: Local Development
```bash
# Install Streamlit
pip install streamlit

# Start API server first
python -m api.main

# Start UI in another terminal
./run_streamlit.sh  # Linux/Mac
# or: run_streamlit.bat  # Windows

# Access UI at: http://localhost:8501
```

### UI Features

#### 💬 Chat Tab
- Select from downloaded models
- Real-time chat interface
- Message history
- Streaming responses

#### 📦 Models Tab
- Download popular models from HuggingFace
- View available local models
- Model registry integration

#### 📊 Analytics Tab
- Usage statistics (coming soon)
- Model performance metrics

## License

MIT
