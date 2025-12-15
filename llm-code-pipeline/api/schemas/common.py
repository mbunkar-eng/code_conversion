"""
Common schema models used across the API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ErrorDetail(BaseModel):
    """Error detail information."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format (OpenAI-compatible)."""
    error: ErrorDetail

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    owned_by: str = Field(default="organization", description="Owner")
    permission: list = Field(default_factory=list, description="Permissions")
    root: str = Field(..., description="Root model")
    parent: Optional[str] = Field(default=None, description="Parent model")


class ModelList(BaseModel):
    """List of available models."""
    object: str = Field(default="list")
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    gpu_available: bool = Field(..., description="GPU availability")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "qwen2.5-coder-7b",
                "gpu_available": True,
                "version": "1.0.0"
            }
        }


class TokenizeRequest(BaseModel):
    """Request for tokenization."""
    text: str = Field(..., description="Text to tokenize")
    model: Optional[str] = Field(None, description="Model for tokenization")


class TokenizeResponse(BaseModel):
    """Response from tokenization."""
    tokens: list[int] = Field(..., description="Token IDs")
    token_count: int = Field(..., description="Number of tokens")


class CountTokensRequest(BaseModel):
    """Request to count tokens."""
    text: str = Field(..., description="Text to count tokens for")
    model: Optional[str] = Field(None, description="Model for tokenization")


class CountTokensResponse(BaseModel):
    """Response with token count."""
    token_count: int = Field(..., description="Number of tokens")
