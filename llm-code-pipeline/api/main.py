"""
Main FastAPI application for LLM Code Pipeline.
"""

import os
import sys
import logging
from typing import Optional
from contextlib import asynccontextmanager

# Add the project root to the path when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import chat_router, completions_router, health_router, models_router
from api.utils.logging import setup_logging, log_request
from api.utils.error_handler import error_handler, LLMPipelineError

logger = logging.getLogger(__name__)

# Global instances
_model_manager = None

# Initialize model manager on import
_init_done = False
if not _init_done:
    _init_done = True
    print("DEBUG: Initializing model manager")
    try:
        from inference.model_manager import ModelManager
        _model_manager = ModelManager()
        print("DEBUG: Model manager initialized successfully")
    except Exception as e:
        print(f"DEBUG: Failed to initialize model manager: {e}")
        import traceback
        traceback.print_exc()
        _model_manager = None


def get_model_manager():
    """Get the global model manager instance."""
    return _model_manager


def get_inference_runner(model_id: str):
    """Get the inference runner for a specific model."""
    if _model_manager is None:
        return None
    try:
        runner, _ = _model_manager.get_or_load_model(model_id)
        return runner
    except Exception as e:
        logger.error(f"Failed to get inference runner for model {model_id}: {e}")
        return None


def get_tokenizer_service(model_id: str):
    """Get the tokenizer service for a specific model."""
    if _model_manager is None:
        return None
    try:
        _, tokenizer = _model_manager.get_or_load_model(model_id)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to get tokenizer service for model {model_id}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Model is already loaded at import time
    yield


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
