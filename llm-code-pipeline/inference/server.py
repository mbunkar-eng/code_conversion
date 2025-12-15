"""
Inference Server - Standalone vLLM inference server launcher.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def launch_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    dtype: str = "float16",
    quantization: str = None,
    api_key: str = None,
    enable_prefix_caching: bool = True,
    max_num_seqs: int = 256
):
    """
    Launch vLLM OpenAI-compatible server.

    Args:
        model_path: Path to model or HuggingFace repo
        host: Server host
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction to use
        max_model_len: Maximum sequence length
        dtype: Data type (float16, bfloat16, float32)
        quantization: Quantization method (awq, gptq)
        api_key: API key for authentication
        enable_prefix_caching: Enable KV cache prefix caching
        max_num_seqs: Maximum concurrent sequences
    """
    try:
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)

    # Build arguments
    args = [
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code"
    ]

    if max_model_len:
        args.extend(["--max-model-len", str(max_model_len)])

    if quantization:
        args.extend(["--quantization", quantization])

    if api_key:
        args.extend(["--api-key", api_key])

    if enable_prefix_caching:
        args.append("--enable-prefix-caching")

    logger.info(f"Launching vLLM server with model: {model_path}")
    logger.info(f"Server will be available at http://{host}:{port}")

    # Parse and run
    parser = make_arg_parser()
    parsed_args = parser.parse_args(args)
    run_server(parsed_args)


def launch_mock_server(
    host: str = "0.0.0.0",
    port: int = 8000
):
    """
    Launch mock inference server for testing.
    Uses the API module directly with mock runner.
    """
    import uvicorn

    # Set environment variable to use mock mode
    os.environ["LLM_PIPELINE_MOCK_MODE"] = "true"

    logger.info(f"Launching mock server at http://{host}:{port}")

    # Import and run the FastAPI app
    from api.main import app
    uvicorn.run(app, host=host, port=port)


def main():
    """CLI entry point for inference server."""
    parser = argparse.ArgumentParser(
        description="Launch LLM inference server"
    )

    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to model or HuggingFace repo"
    )
    parser.add_argument(
        "--host", "-H",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model data type"
    )
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", None],
        default=None,
        help="Quantization method"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock server for testing"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.mock:
        launch_mock_server(args.host, args.port)
    else:
        if not args.model_path:
            parser.error("model_path is required unless using --mock")

        launch_vllm_server(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            quantization=args.quantization,
            api_key=args.api_key
        )


if __name__ == "__main__":
    main()
