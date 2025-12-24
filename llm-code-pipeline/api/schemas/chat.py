"""
Chat completion schemas (OpenAI-compatible).
"""

from typing import Optional, Literal, Union, Any
from pydantic import BaseModel, Field
import time

from .common import Usage
from .response_format import ResponseFormat


class ChatMessage(BaseModel):
    """Chat message structure."""
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the message sender"
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the sender")

    class Config:
        json_schema_extra = {
            "examples": [
                {"role": "system", "content": "You are a code conversion expert."},
                {"role": "user", "content": "Convert this Python code to Java..."},
                {"role": "assistant", "content": "Here is the converted code..."}
            ]
        }


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request (OpenAI-compatible or simplified).

    Supports both full OpenAI format and simplified format with just content.
    """
    model: Optional[str] = Field(None, description="Model to use for completion (optional, uses default if not specified)")
    messages: Optional[list[ChatMessage]] = Field(
        None,
        description="List of messages in the conversation (OpenAI format)"
    )
    content: Optional[str] = Field(
        None,
        description="Simplified content field (alternative to messages)"
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
        description="Number of completions to generate"
    )
    stream: bool = Field(
        default=False,
        description="Stream responses"
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Stop sequences"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
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
    logit_bias: Optional[dict[str, float]] = Field(
        default=None,
        description="Token logit biases"
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier"
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="Response format specification"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "content": "Convert this Python to Java:\ndef hello():\n    print('Hello')",
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                {
                    "model": "qwen2.5-coder-7b",
                    "messages": [
                        {"role": "system", "content": "You are a code conversion expert."},
                        {"role": "user", "content": "Convert this Python to Java:\ndef hello():\n    print('Hello')"}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
            ]
        }

    def normalize_messages(self) -> list[ChatMessage]:
        """
        Normalize input to standard messages format.
        
        Converts simplified content format to full messages format.
        """
        if self.messages:
            return self.messages
        elif self.content:
            # Convert simplified format to messages
            return [
                ChatMessage(role="user", content=self.content)
            ]
        else:
            raise ValueError("Either 'messages' or 'content' must be provided")


class ChatChoice(BaseModel):
    """Single chat completion choice."""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: Literal["stop", "length", "content_filter"] = Field(
        ...,
        description="Reason for completion"
    )
    logprobs: Optional[Any] = Field(default=None, description="Log probabilities")


class ChatCompletionResponse(BaseModel):
    """
    Chat completion response (OpenAI-compatible).
    """
    id: str = Field(..., description="Unique completion ID")
    object: Literal["chat.completion"] = Field(default="chat.completion")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp"
    )
    model: str = Field(..., description="Model used")
    choices: list[ChatChoice] = Field(..., description="Completion choices")
    usage: Usage = Field(..., description="Token usage")
    system_fingerprint: Optional[str] = Field(
        default=None,
        description="System fingerprint"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1699000000,
                "model": "qwen2.5-coder-7b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 30,
                    "total_tokens": 80
                }
            }
        }


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response."""
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice in streaming response."""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    """
    Streaming chat completion chunk (OpenAI-compatible).
    """
    id: str = Field(..., description="Completion ID")
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk"
    )
    created: int = Field(
        default_factory=lambda: int(time.time())
    )
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion.chunk",
                "created": 1699000000,
                "model": "qwen2.5-coder-7b",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "public "},
                    "finish_reason": None
                }]
            }
        }
