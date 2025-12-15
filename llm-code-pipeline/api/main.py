"""
Main FastAPI application for LLM Code Pipeline.
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import chat_router, completions_router, health_router, models_router
from .utils.logging import setup_logging, log_request
from .utils.error_handler import error_handler, LLMPipelineError

logger = logging.getLogger(__name__)

# Global instances
_inference_runner = None
_tokenizer_service = None


def get_inference_runner():
    """Get the global inference runner instance."""
    return _inference_runner


def get_tokenizer_service():
    """Get the global tokenizer service instance."""
    return _tokenizer_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _inference_runner, _tokenizer_service

    # Setup logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    json_logs = os.environ.get("JSON_LOGS", "false").lower() == "true"
    setup_logging(level=log_level, json_format=json_logs)

    logger.info("Starting LLM Code Pipeline API...")

    # Initialize model if configured
    model_path = os.environ.get("MODEL_PATH")
    mock_mode = os.environ.get("LLM_PIPELINE_MOCK_MODE", "false").lower() == "true"

    if mock_mode:
        logger.info("Running in mock mode - no GPU required")
        from inference.vllm_runner import MockVLLMRunner
        from inference.tokenizer_service import MockTokenizerService

        _inference_runner = MockVLLMRunner(model_path or "mock-model")
        _inference_runner.initialize()

        _tokenizer_service = MockTokenizerService(model_path or "mock-model")
        _tokenizer_service.initialize()

    elif model_path:
        logger.info(f"Loading model from: {model_path}")

        try:
            from inference.vllm_runner import VLLMRunner
            from inference.tokenizer_service import TokenizerService

            # Get configuration from environment
            tensor_parallel = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
            gpu_memory_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
            max_model_len = os.environ.get("MAX_MODEL_LEN")
            max_model_len = int(max_model_len) if max_model_len else None
            dtype = os.environ.get("DTYPE", "float16")
            quantization = os.environ.get("QUANTIZATION")

            _inference_runner = VLLMRunner(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=max_model_len,
                dtype=dtype,
                quantization=quantization
            )
            _inference_runner.initialize()

            _tokenizer_service = TokenizerService(model_path)
            _tokenizer_service.initialize()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Continue without model - health endpoints will report not ready
    else:
        logger.warning("No MODEL_PATH configured - API will start without a model")

    yield

    # Cleanup
    logger.info("Shutting down LLM Code Pipeline API...")
    if _inference_runner:
        _inference_runner.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Code Pipeline API",
        description="""
        OpenAI-compatible API for code generation and conversion using self-hosted LLMs.

        ## Features

        - **Chat Completions** (`/v1/chat/completions`): OpenAI-compatible chat API
        - **Completions** (`/v1/completions`): Legacy completions API
        - **JSON Mode**: Structured JSON output support
        - **Streaming**: Server-sent events for streaming responses
        - **Model Management**: List and query available models

        ## Authentication

        Include your API key in requests:
        - Header: `Authorization: Bearer YOUR_API_KEY`
        - Header: `X-API-Key: YOUR_API_KEY`

        ## Code Conversion Example

        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            json={
                "model": "qwen2.5-coder-7b",
                "messages": [
                    {"role": "system", "content": "You are a code conversion expert."},
                    {"role": "user", "content": "Convert this Python to Java: def hello(): print('Hello')"}
                ],
                "response_format": {"type": "json_object"}
            }
        )
        print(response.json())
        ```
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Add CORS middleware
    cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Add request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        return await log_request(request, call_next)

    # Add exception handlers
    @app.exception_handler(LLMPipelineError)
    async def llm_error_handler(request: Request, exc: LLMPipelineError):
        return await error_handler(request, exc)

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        return await error_handler(request, exc)

    # Include routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(completions_router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "LLM Code Pipeline API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    return app


# Create default app instance
app = create_app()


def main():
    """Run the API server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LLM Code Pipeline API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model", help="Model path or HuggingFace repo")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")

    args = parser.parse_args()

    # Set environment variables from args
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.mock:
        os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
