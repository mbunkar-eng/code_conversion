"""
Chat completions endpoint (OpenAI-compatible).
"""

import uuid
import time
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    Usage
)
from ..utils.auth import verify_api_key
from ..utils.validators import validate_messages, validate_response_format
from ..utils.error_handler import handle_errors, InvalidRequestError, InferenceError
from ..utils.logging import request_logger

router = APIRouter(prefix="/v1", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@handle_errors
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a chat completion.

    This endpoint is compatible with the OpenAI Chat Completions API.
    It accepts a list of messages and returns a model-generated response.

    Supports:
    - Standard text responses
    - JSON mode (response_format: {"type": "json_object"})
    - JSON schema mode (response_format: {"type": "json_schema", "json_schema": {...}})
    - Streaming responses (stream: true)
    """
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])

    # Normalize messages (handles both full and simplified formats)
    messages = request.normalize_messages()
    
    # Validate inputs
    validate_messages(messages)
    validate_response_format(request.response_format)

    # Get inference runner for the requested model (or default)
    from ..main import get_inference_runner, get_tokenizer_service, get_model_manager
    
    # Use provided model or get default
    model_id = request.model
    if model_id is None:
        model_manager = get_model_manager()
        model_id = model_manager.get_default_model()
        logger.info(f"Using default model: {model_id}")
    
    runner = get_inference_runner(model_id)
    tokenizer = get_tokenizer_service(model_id)

    if runner is None:
        raise InferenceError(f"Failed to load model: {model_id}")

    if tokenizer is None:
        raise InferenceError(f"Failed to load tokenizer for model: {model_id}")

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, runner, tokenizer, request_id, model_id, messages),
            media_type="text/event-stream"
        )

    # Build prompt from messages
    from inference.tokenizer_service import ChatMessage as TokenizerChatMessage

    tokenizer_messages = [
        TokenizerChatMessage(role=msg.role, content=msg.content)
        for msg in messages
    ]

    # Add JSON instruction if needed
    prompt = tokenizer.apply_chat_template(tokenizer_messages, add_generation_prompt=True)

    if request.response_format:
        if request.response_format.type == "json_object":
            prompt += "\n\nRespond with valid JSON only."
        elif request.response_format.type == "json_schema" and request.response_format.json_schema:
            schema_str = json.dumps(request.response_format.json_schema.schema_, indent=2)
            prompt += f"\n\nRespond with valid JSON matching this schema:\n{schema_str}"

    # Configure generation
    from inference.vllm_runner import GenerationConfig

    stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else []

    config = GenerationConfig(
        max_tokens=request.max_tokens or 2048,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=stop_sequences,
        n=request.n
    )

    # Run inference
    start_time = time.perf_counter()
    result = runner.generate(prompt, config)
    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Process response for JSON mode
    response_text = result.text

    if request.response_format and request.response_format.type in ["json_object", "json_schema"]:
        from inference.json_formatter import JSONFormatter, JSONExtractionMode

        formatter = JSONFormatter()
        json_result = formatter.extract_json(response_text, JSONExtractionMode.LENIENT)

        if json_result.success:
            response_text = json.dumps(json_result.data)
        else:
            logger.warning(f"Failed to extract JSON: {json_result.error}")

    # Log inference metrics
    request_logger.log_inference(
        request_id=request_id,
        model=request.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        duration_ms=generation_time_ms
    )

    # Get actual model name from loaded runner
    model_info = runner.get_model_info()
    actual_model = model_info.get("model_path", model_id)
    if actual_model and "/" in actual_model:
        actual_model = actual_model.split("/")[-1]
    elif actual_model and "--" in actual_model:
        actual_model = actual_model.split("--")[-1]

    # Build response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(time.time()),
        model=actual_model or model_id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response_text
                ),
                finish_reason=result.finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens
        ),
        system_fingerprint=f"fp_{uuid.uuid4().hex[:12]}"
    )


async def stream_chat_completion(
    request: ChatCompletionRequest,
    runner,
    tokenizer,
    request_id: str,
    model_id: str,
    messages: list
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    from inference.tokenizer_service import ChatMessage as TokenizerChatMessage

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Build prompt
    tokenizer_messages = [
        TokenizerChatMessage(role=msg.role, content=msg.content)
        for msg in messages
    ]
    prompt = tokenizer.apply_chat_template(tokenizer_messages, add_generation_prompt=True)

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant"),
                finish_reason=None
            )
        ]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Generate content chunks
    from inference.vllm_runner import GenerationConfig

    config = GenerationConfig(
        max_tokens=request.max_tokens or 2048,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=True
    )

    async for text_chunk in runner.generate_stream(prompt, config):
        chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=text_chunk),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
