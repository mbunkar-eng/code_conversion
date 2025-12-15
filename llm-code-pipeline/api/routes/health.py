"""
Health check and status endpoints.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends
import torch

from ..schemas.common import HealthResponse
from ..utils.auth import verify_api_key

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)

# Track startup time
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and basic information.
    No authentication required.
    """
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    # Get model status from app state (set by main.py)
    from ..main import get_inference_runner
    runner = get_inference_runner()
    model_loaded = runner is not None and runner._initialized
    model_name = runner.model_path if model_loaded else None

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=model_name,
        gpu_available=gpu_available,
        version="1.0.0"
    )


@router.get("/health/detailed")
async def detailed_health_check(api_key: str = Depends(verify_api_key)):
    """
    Detailed health check with system information.

    Requires authentication.
    """
    gpu_available = torch.cuda.is_available()

    gpu_info = []
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            memory_total = props.total_memory / (1024 ** 3)

            gpu_info.append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(memory_total, 2),
                "allocated_memory_gb": round(memory_allocated, 2),
                "reserved_memory_gb": round(memory_reserved, 2),
                "utilization_percent": round((memory_allocated / memory_total) * 100, 1)
            })

    from ..main import get_inference_runner
    runner = get_inference_runner()

    model_info = None
    if runner and runner._initialized:
        model_info = runner.get_model_info()

    uptime_seconds = time.time() - _startup_time

    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 2),
        "gpu": {
            "available": gpu_available,
            "count": len(gpu_info),
            "devices": gpu_info
        },
        "model": model_info,
        "python_version": __import__("sys").version,
        "torch_version": torch.__version__
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.

    Returns 200 if service is ready to accept requests.
    """
    from ..main import get_inference_runner
    runner = get_inference_runner()

    if runner is None or not runner._initialized:
        return {"ready": False, "reason": "Model not loaded"}

    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.

    Returns 200 if service is alive.
    """
    return {"alive": True}
