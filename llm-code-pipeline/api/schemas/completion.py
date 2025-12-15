"""
Completion schemas (OpenAI-compatible legacy completions).
"""

from typing import Optional, Literal, Union, Any
from pydantic import BaseModel, Field
import time

from .common import Usage
from .response_format import ResponseFormat


class CompletionRequest(BaseModel):
    """
    Completion request (OpenAI-compatible).

    Legacy completions API - for simpler prompt-in, text-out use cases.
    """
    model: str = Field(..., description="Model to use for completion")
    prompt: Union[str, list[str]] = Field(
        ...,
        description="Prompt(s) to complete"
    )
    suffix: Optional[str] = Field(
        default=None,
        description="Suffix to append after completion"
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions"
    )
    stream: bool = Field(
        default=False,
        description="Stream responses"
    )
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Include log probabilities"
    )
    echo: bool = Field(
        default=False,
        description="Echo prompt in response"
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Stop sequences"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    best_of: Optional[int] = Field(
        default=None,
        ge=1,
        description="Generate best_of completions and return best"
    )
    logit_bias: Optional[dict[str, float]] = Field(
        default=None,
        description="Token logit biases"
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen2.5-coder-7b",
                "prompt": "Convert this Python code to Java:\ndef hello():\n    print('Hello')\n\n// Java code:",
                "max_tokens": 300,
                "temperature": 0.2,
                "stop": ["\n\n", "```"]
            }
        }


class CompletionLogprobs(BaseModel):
    """Log probability information."""
    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: Optional[list[dict[str, float]]] = None
    text_offset: list[int]


class CompletionChoice(BaseModel):
    """Single completion choice."""
    text: str = Field(..., description="Generated text")
    index: int = Field(..., description="Choice index")
    logprobs: Optional[CompletionLogprobs] = Field(
        default=None,
        description="Log probabilities"
    )
    finish_reason: Literal["stop", "length", "content_filter"] = Field(
        ...,
        description="Reason for completion"
    )


class CompletionResponse(BaseModel):
    """
    Completion response (OpenAI-compatible).
    """
    id: str = Field(..., description="Unique completion ID")
    object: Literal["text_completion"] = Field(default="text_completion")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp"
    )
    model: str = Field(..., description="Model used")
    choices: list[CompletionChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="System fingerprint"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "cmpl-abc123",
                "object": "text_completion",
                "created": 1699000000,
                "model": "qwen2.5-coder-7b",
                "choices": [{
                    "text": "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 35,
                    "total_tokens": 75
                }
            }
        }


class CompletionChunk(BaseModel):
    """
    Streaming completion chunk.
    """
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[dict]  # Simplified for streaming
    system_fingerprint: Optional[str] = None
