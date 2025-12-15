"""
Utility modules for the API.
"""

from .auth import verify_api_key, get_api_key_from_header
from .logging import setup_logging, get_logger, log_request
from .error_handler import (
    LLMPipelineError,
    InvalidRequestError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    InferenceError,
    error_handler
)
from .validators import (
    validate_model_id,
    validate_messages,
    validate_generation_params
)

__all__ = [
    "verify_api_key",
    "get_api_key_from_header",
    "setup_logging",
    "get_logger",
    "log_request",
    "LLMPipelineError",
    "InvalidRequestError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "InferenceError",
    "error_handler",
    "validate_model_id",
    "validate_messages",
    "validate_generation_params"
]
