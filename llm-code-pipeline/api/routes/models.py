"""
Model listing and information endpoints.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.common import ModelInfo, ModelList
from ..utils.auth import verify_api_key

router = APIRouter(prefix="/v1", tags=["Models"])
logger = logging.getLogger(__name__)


@router.get("/models", response_model=ModelList)
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List available models.

    Returns a list of models available for inference.
    """
    from models.registry import ModelRegistry

    registry = ModelRegistry()
    model_ids = registry.list_models()

    models = []
    for model_id in model_ids:
        model_info = registry.get_model(model_id)
        if model_info:
            models.append(ModelInfo(
                id=model_id,
                object="model",
                created=int(time.time()),
                owned_by="organization",
                permission=[],
                root=model_id,
                parent=model_info.base_model
            ))

    return ModelList(object="list", data=models)


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about a specific model.
    """
    from models.registry import ModelRegistry

    registry = ModelRegistry()
    model_info = registry.get_model(model_id)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    return ModelInfo(
        id=model_id,
        object="model",
        created=int(time.time()),
        owned_by="organization",
        permission=[],
        root=model_id,
        parent=model_info.base_model
    )


@router.get("/models/{model_id}/details")
async def get_model_details(
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get detailed information about a model including requirements.
    """
    from models.registry import ModelRegistry

    registry = ModelRegistry()
    model_info = registry.get_model(model_id)

    if not model_info:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    return {
        "id": model_id,
        "name": model_info.name,
        "hf_repo": model_info.hf_repo,
        "type": model_info.model_type,
        "parameters": model_info.parameters,
        "context_length": model_info.context_length,
        "supported_formats": model_info.supported_formats,
        "default_format": model_info.default_format,
        "gpu_memory_required_gb": model_info.gpu_memory_required_gb,
        "recommended_gpu": model_info.recommended_gpu,
        "quantization": model_info.quantization,
        "bits": model_info.bits
    }
