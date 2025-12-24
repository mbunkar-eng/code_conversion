#!/usr/bin/env python3
"""
LLM Code Pipeline - Main Entry Point

Run the LLM Code Pipeline locally for code generation and code conversion.

Usage:
    # Run with default settings (mock mode for testing)
    python main.py

    # Run with real model on Mac (MPS)
    python main.py --model Qwen/Qwen2-0.5B-Instruct

    # Run with specific host/port
    python main.py --host 0.0.0.0 --port 8080

    # Run with larger model for better results
    python main.py --model Qwen/Qwen2.5-Coder-7B-Instruct

Examples:
    # Quick test (mock mode)
    python main.py --mock

    # Production mode with real model
    python main.py --model Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0
"""

import argparse
import os
import sys
from pathlib import Path
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Code Pipeline - OpenAI-compatible API for code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Mock mode (testing)
  python main.py --model Qwen/Qwen2-0.5B-Instruct  # Real model
  python main.py --host 0.0.0.0 --port 8080        # Custom host/port

Supported Models:
  - Qwen/Qwen2-0.5B-Instruct      (~1GB, fastest)
  - Qwen/Qwen2.5-Coder-7B-Instruct (~14GB, best quality)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2GB)
  - microsoft/phi-2               (~5GB)
        """
    )

    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000)"
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (HuggingFace model ID or local path)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no real model, for testing)"
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for model (default: float16)"
    )

    # Server options
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    return parser.parse_args()


def setup_environment(args):
    """Configure environment variables based on arguments."""

    # Enable MPS fallback for Mac
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Check for available models
    models_dir = Path("./downloaded_models")
    available_models = []
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                available_models.append(item.name)

    # Mock mode logic
    if args.mock:
        os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"
        print("Running in MOCK MODE (forced by --mock flag)")
    elif args.model:
        os.environ["LLM_PIPELINE_MOCK_MODE"] = "false"
        os.environ["LLM_PIPELINE_MODEL"] = args.model
        print(f"Using specified model: {args.model}")
    elif available_models:
        # Use the first available model (prefer smaller models for speed)
        preferred_order = ["Qwen--Qwen2-0.5B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct", "deepseek-ai--deepseek-coder-6.7b-instruct"]
        selected_model = None
        for pref in preferred_order:
            if pref in available_models:
                selected_model = pref
                break
        if not selected_model:
            selected_model = available_models[0]
        
        os.environ["LLM_PIPELINE_MOCK_MODE"] = "false"
        os.environ["LLM_PIPELINE_MODEL"] = selected_model
        print(f"Using available model: {selected_model}")
    else:
        os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"
        print("Running in MOCK MODE (no models available)")

    # Device configuration
    if args.device != "auto":
        os.environ["LLM_PIPELINE_DEVICE"] = args.device

    # Data type
    os.environ["LLM_PIPELINE_DTYPE"] = args.dtype


def print_banner(args):
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  LLM Code Pipeline")
    print("  OpenAI-compatible API for Code Generation & Conversion")
    print("=" * 60)
    print()
    print(f"  Server:  http://{args.host}:{args.port}")
    print(f"  Docs:    http://{args.host}:{args.port}/docs")
    print(f"  Health:  http://{args.host}:{args.port}/health")
    print()

    mock_mode = os.environ.get("LLM_PIPELINE_MOCK_MODE", "false").lower() == "true"
    
    if mock_mode:
        print("  Mode:    MOCK (for testing)")
        if args.mock:
            print("  Reason:  Forced by --mock flag")
        else:
            print("  Reason:  No models available")
    else:
        model = os.environ.get("LLM_PIPELINE_MODEL", "unknown")
        print(f"  Mode:    REAL INFERENCE")
        print(f"  Model:   {model}")
        print(f"  Device:  {args.device}")
        print(f"  Dtype:   {args.dtype}")

    print()
    print("  Endpoints:")
    print("    POST /v1/chat/completions  - Chat completion")
    print("    POST /v1/completions       - Text completion")
    print("    GET  /v1/models            - List models")
    print()
    print("=" * 60)
    print()


def main():
    args = parse_args()

    # Setup environment
    setup_environment(args)

    # Print banner
    print_banner(args)

    # Run server
    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
