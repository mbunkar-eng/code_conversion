"""
Error handling utilities for the LLM Pipeline API.
"""

import logging
import traceback
from typing import Optional, Any
from functools import wraps

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class LLMPipelineError(Exception):
    """Base exception for LLM Pipeline errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "internal_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
        status_code: int = 500
    ):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to OpenAI-compatible error response."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
                "param": self.param
            }
        }


class InvalidRequestError(LLMPipelineError):
    """Invalid request parameters."""

    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
        code: str = "invalid_request"
    ):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            code=code,
            param=param,
            status_code=400
        )


class AuthenticationError(LLMPipelineError):
    """Authentication failure."""

    def __init__(
        self,
        message: str = "Invalid API key",
        code: str = "invalid_api_key"
    ):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            code=code,
            status_code=401
        )


class RateLimitError(LLMPipelineError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429
        )
        self.retry_after = retry_after


class ModelNotFoundError(LLMPipelineError):
    """Model not found."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' not found",
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
            status_code=404
        )


class InferenceError(LLMPipelineError):
    """Inference execution error."""

    def __init__(self, message: str, details: Optional[str] = None):
        full_message = message
        if details:
            full_message = f"{message}: {details}"
        super().__init__(
            message=full_message,
            error_type="server_error",
            code="inference_error",
            status_code=500
        )


class ContextLengthExceededError(LLMPipelineError):
    """Context length exceeded."""

    def __init__(self, max_length: int, requested_length: int):
        super().__init__(
            message=f"Context length {requested_length} exceeds maximum {max_length}",
            error_type="invalid_request_error",
            code="context_length_exceeded",
            status_code=400
        )


class ContentFilterError(LLMPipelineError):
    """Content filtered due to policy."""

    def __init__(self, message: str = "Content filtered due to policy violation"):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            code="content_filter",
            status_code=400
        )


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler for the API.

    Converts exceptions to OpenAI-compatible error responses.
    """
    # Handle our custom exceptions
    if isinstance(exc, LLMPipelineError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )

    # Handle FastAPI HTTP exceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error",
                    "code": None,
                    "param": None
                }
            }
        )

    # Handle Pydantic validation errors
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            first_error = errors[0]
            param = ".".join(str(loc) for loc in first_error.get("loc", []))
            message = first_error.get("msg", "Validation error")
        else:
            param = None
            message = "Request validation failed"

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "validation_error",
                    "param": param
                }
            }
        )

    # Handle unexpected errors
    logger.error(
        f"Unexpected error: {exc}",
        exc_info=True,
        extra={"traceback": traceback.format_exc()}
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "type": "server_error",
                "code": "internal_error",
                "param": None
            }
        }
    )


def handle_errors(func):
    """Decorator to handle errors in route handlers."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except LLMPipelineError:
            raise
        except HTTPException:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise InferenceError(
                message="An error occurred during processing",
                details=str(e)
            )
    return wrapper
