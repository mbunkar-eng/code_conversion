"""
Request validation utilities.
"""

import re
from typing import Optional

from ..schemas import ChatMessage, ResponseFormat
from .error_handler import InvalidRequestError


def validate_model_id(
    model_id: str,
    available_models: Optional[list[str]] = None
) -> str:
    """
    Validate model identifier.

    Args:
        model_id: Model ID to validate
        available_models: List of available models (if None, any ID is valid)

    Returns:
        Validated model ID

    Raises:
        InvalidRequestError if model ID is invalid
    """
    if not model_id or not model_id.strip():
        raise InvalidRequestError(
            message="Model ID cannot be empty",
            param="model"
        )

    model_id = model_id.strip()

    # Check against available models if provided
    if available_models and model_id not in available_models:
        raise InvalidRequestError(
            message=f"Model '{model_id}' not found. Available models: {', '.join(available_models)}",
            param="model",
            code="model_not_found"
        )

    return model_id


def validate_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Validate chat messages.

    Args:
        messages: List of chat messages

    Returns:
        Validated messages

    Raises:
        InvalidRequestError if messages are invalid
    """
    if not messages:
        raise InvalidRequestError(
            message="Messages cannot be empty",
            param="messages"
        )

    valid_roles = {"system", "user", "assistant"}

    for i, msg in enumerate(messages):
        if msg.role not in valid_roles:
            raise InvalidRequestError(
                message=f"Invalid role '{msg.role}' at message {i}. Must be one of: {', '.join(valid_roles)}",
                param=f"messages[{i}].role"
            )

        if not msg.content and msg.content != "":
            raise InvalidRequestError(
                message=f"Message content at index {i} cannot be null",
                param=f"messages[{i}].content"
            )

    # Check for at least one user message
    has_user_message = any(msg.role == "user" for msg in messages)
    if not has_user_message:
        raise InvalidRequestError(
            message="Messages must contain at least one user message",
            param="messages"
        )

    return messages


def validate_generation_params(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    n: Optional[int] = None,
    stop: Optional[list[str]] = None
) -> dict:
    """
    Validate generation parameters.

    Returns validated parameters dict.

    Raises:
        InvalidRequestError if parameters are invalid
    """
    params = {}

    if temperature is not None:
        if not 0 <= temperature <= 2:
            raise InvalidRequestError(
                message="Temperature must be between 0 and 2",
                param="temperature"
            )
        params["temperature"] = temperature

    if top_p is not None:
        if not 0 <= top_p <= 1:
            raise InvalidRequestError(
                message="top_p must be between 0 and 1",
                param="top_p"
            )
        params["top_p"] = top_p

    if max_tokens is not None:
        if max_tokens < 1:
            raise InvalidRequestError(
                message="max_tokens must be at least 1",
                param="max_tokens"
            )
        if max_tokens > 128000:  # Reasonable upper limit
            raise InvalidRequestError(
                message="max_tokens cannot exceed 128000",
                param="max_tokens"
            )
        params["max_tokens"] = max_tokens

    if frequency_penalty is not None:
        if not -2 <= frequency_penalty <= 2:
            raise InvalidRequestError(
                message="frequency_penalty must be between -2 and 2",
                param="frequency_penalty"
            )
        params["frequency_penalty"] = frequency_penalty

    if presence_penalty is not None:
        if not -2 <= presence_penalty <= 2:
            raise InvalidRequestError(
                message="presence_penalty must be between -2 and 2",
                param="presence_penalty"
            )
        params["presence_penalty"] = presence_penalty

    if n is not None:
        if not 1 <= n <= 10:
            raise InvalidRequestError(
                message="n must be between 1 and 10",
                param="n"
            )
        params["n"] = n

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        if len(stop) > 4:
            raise InvalidRequestError(
                message="Maximum 4 stop sequences allowed",
                param="stop"
            )
        params["stop"] = stop

    return params


def validate_response_format(
    response_format: Optional[ResponseFormat]
) -> Optional[ResponseFormat]:
    """
    Validate response format specification.

    Args:
        response_format: Response format to validate

    Returns:
        Validated response format

    Raises:
        InvalidRequestError if format is invalid
    """
    if response_format is None:
        return None

    valid_types = {"text", "json_object", "json_schema"}

    if response_format.type not in valid_types:
        raise InvalidRequestError(
            message=f"Invalid response_format type '{response_format.type}'. Must be one of: {', '.join(valid_types)}",
            param="response_format.type"
        )

    if response_format.type == "json_schema":
        if not response_format.json_schema:
            raise InvalidRequestError(
                message="json_schema must be provided when type is 'json_schema'",
                param="response_format.json_schema"
            )

        schema = response_format.json_schema
        if not schema.name:
            raise InvalidRequestError(
                message="json_schema.name is required",
                param="response_format.json_schema.name"
            )

    return response_format


def validate_prompt(
    prompt: str,
    max_length: Optional[int] = None
) -> str:
    """
    Validate completion prompt.

    Args:
        prompt: Prompt text
        max_length: Maximum character length

    Returns:
        Validated prompt

    Raises:
        InvalidRequestError if prompt is invalid
    """
    if not prompt and prompt != "":
        raise InvalidRequestError(
            message="Prompt cannot be null",
            param="prompt"
        )

    if max_length and len(prompt) > max_length:
        raise InvalidRequestError(
            message=f"Prompt exceeds maximum length of {max_length} characters",
            param="prompt"
        )

    return prompt


def sanitize_input(text: str) -> str:
    """
    Sanitize input text by removing potentially harmful characters.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove other control characters except newlines and tabs
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text
