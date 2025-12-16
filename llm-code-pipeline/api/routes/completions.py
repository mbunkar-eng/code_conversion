"""
Completions endpoint (OpenAI-compatible legacy API).
"""

import uuid
import time
import logging
from typing import Union, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    Usage
)
from ..utils.auth import verify_api_key
from ..utils.validators import validate_prompt
from ..utils.error_handler import handle_errors, InferenceError
from ..utils.logging import request_logger

router = APIRouter(prefix="/v1", tags=["Completions"])
logger = logging.getLogger(__name__)


@router.post("/completions", response_model=CompletionResponse)
@handle_errors
async def create_completion(
    request: CompletionRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a completion.

    This endpoint is compatible with the OpenAI Completions API (legacy).
    It accepts a prompt and returns a model-generated completion.

    Note: For chat-based interactions, use /v1/chat/completions instead.
    """
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4())[:8])

    # Handle single or batch prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    # Validate prompts
    for i, prompt in enumerate(prompts):
        validate_prompt(prompt)

    # Get inference runner for the requested model
    from ..main import get_inference_runner
    runner = get_inference_runner(request.model)

    if runner is None:
        raise InferenceError(f"Failed to load model: {request.model}")

    # Handle streaming
    if request.stream:
        if len(prompts) > 1:
            raise InferenceError("Streaming not supported for batch prompts")

        return StreamingResponse(
            stream_completion(request, runner, prompts[0], request_id),
            media_type="text/event-stream"
        )

    # Configure generation
    from inference.vllm_runner import GenerationConfig

    stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else []

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stop=stop_sequences,
        n=request.n,
        best_of=request.best_of,
        logprobs=request.logprobs,
        echo=request.echo
    )

    # Run inference
    start_time = time.perf_counter()

    if len(prompts) == 1:
        results = [runner.generate(prompts[0], config)]
    else:
        results = runner.generate(prompts, config)

    generation_time_ms = (time.perf_counter() - start_time) * 1000

    # Build choices
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, result in enumerate(results):
        text = result.text
        if request.echo:
            text = prompts[i if i < len(prompts) else 0] + text

        choices.append(CompletionChoice(
            text=text,
            index=i,
            logprobs=None,  # TODO: Add logprobs support
            finish_reason=result.finish_reason
        ))

        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens

    # Log inference metrics
    request_logger.log_inference(
        request_id=request_id,
        model=request.model,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        duration_ms=generation_time_ms
    )

    # Get actual model name from loaded runner
    model_info = runner.get_model_info()
    actual_model = model_info.get("model_path", request.model)
    if actual_model and "/" in actual_model:
        actual_model = actual_model.split("/")[-1]
    elif actual_model and "--" in actual_model:
        actual_model = actual_model.split("--")[-1]

    # Build response
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    return CompletionResponse(
        id=completion_id,
        object="text_completion",
        created=int(time.time()),
        model=actual_model or request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens
        ),
        system_fingerprint=f"fp_{uuid.uuid4().hex[:12]}"
    )


async def stream_completion(
    request: CompletionRequest,
    runner,
    prompt: str,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Generate streaming completion response."""
    from inference.vllm_runner import GenerationConfig

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    config = GenerationConfig(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=True
    )

    # Echo prompt if requested
    if request.echo:
        chunk_data = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": prompt,
                "index": 0,
                "logprobs": None,
                "finish_reason": None
            }]
        }
        yield f"data: {__import__('json').dumps(chunk_data)}\n\n"

    # Generate content chunks
    async for text_chunk in runner.generate_stream(prompt, config):
        chunk_data = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": text_chunk,
                "index": 0,
                "logprobs": None,
                "finish_reason": None
            }]
        }
        yield f"data: {__import__('json').dumps(chunk_data)}\n\n"

    # Final chunk
    final_chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "text": "",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {__import__('json').dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/tokenize")
@handle_errors
async def tokenize_text(
    text: str,
    model: str = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Tokenize text and return token IDs.

    Useful for counting tokens before making API calls.
    """
    from ..main import get_tokenizer_service
    tokenizer = get_tokenizer_service()

    if tokenizer is None:
        raise InferenceError("Tokenizer not initialized")

    result = tokenizer.encode(text)

    return {
        "tokens": result.tokens,
        "token_count": result.token_count
    }


@router.post("/count_tokens")
@handle_errors
async def count_tokens(
    text: str,
    model: str = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Count tokens in text.

    Returns the number of tokens without the full token list.
    """
    from ..main import get_tokenizer_service
    tokenizer = get_tokenizer_service()

    if tokenizer is None:
        raise InferenceError("Tokenizer not initialized")

    count = tokenizer.count_tokens(text)

    return {"token_count": count}
