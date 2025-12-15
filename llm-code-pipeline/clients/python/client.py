"""
Python Client SDK for LLM Code Pipeline.

OpenAI-compatible client that can be used as a drop-in replacement.
"""

import os
import json
import logging
from typing import Optional, Union, Iterator, AsyncIterator, Any
from dataclasses import dataclass, field
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice:
    """Completion choice."""
    index: int
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    finish_reason: str = "stop"


@dataclass
class ChatCompletion:
    """Chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletion":
        choices = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            message = ChatMessage(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content", "")
            )
            choices.append(Choice(
                index=c.get("index", 0),
                message=message,
                finish_reason=c.get("finish_reason", "stop")
            ))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint")
        )


@dataclass
class Completion:
    """Completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Completion":
        choices = []
        for c in data.get("choices", []):
            choices.append(Choice(
                index=c.get("index", 0),
                text=c.get("text", ""),
                finish_reason=c.get("finish_reason", "stop")
            ))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

        return cls(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint")
        )


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    id: str
    object: str
    created: int
    model: str
    delta_content: Optional[str] = None
    delta_role: Optional[str] = None
    finish_reason: Optional[str] = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class AuthenticationError(LLMClientError):
    """Authentication failure."""
    pass


class RateLimitError(LLMClientError):
    """Rate limit exceeded."""
    pass


class APIError(LLMClientError):
    """General API error."""
    pass


class LLMClient:
    """
    Synchronous client for LLM Code Pipeline API.

    Compatible with OpenAI's API interface for easy migration.

    Example:
        client = LLMClient(
            base_url="http://localhost:8000",
            api_key="your-api-key"
        )

        # Chat completion
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[
                {"role": "user", "content": "Convert Python to Java: print('hello')"}
            ]
        )
        print(response.choices[0].message.content)

        # With JSON mode
        response = client.chat.completions.create(
            model="qwen2.5-coder-7b",
            messages=[{"role": "user", "content": "Return JSON with 'code' field"}],
            response_format={"type": "json_object"}
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: API base URL (default: http://localhost:8000)
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = (base_url or os.environ.get("LLM_API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._get_headers()
        )

        # Namespaced interfaces (OpenAI-style)
        self.chat = self._ChatNamespace(self)
        self.completions = self._CompletionsNamespace(self)
        self.models = self._ModelsNamespace(self)

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        stream: bool = False
    ) -> Union[dict, Iterator[dict]]:
        """Make HTTP request."""
        url = endpoint if endpoint.startswith("/") else f"/{endpoint}"

        try:
            if stream:
                return self._stream_request(method, url, json_data)

            response = self._client.request(method, url, json=json_data)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            self._handle_error(e.response)
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def _stream_request(
        self,
        method: str,
        url: str,
        json_data: Optional[dict]
    ) -> Iterator[dict]:
        """Make streaming HTTP request."""
        with self._client.stream(method, url, json=json_data) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error response."""
        try:
            body = response.json()
            error = body.get("error", {})
            message = error.get("message", response.text)
        except:
            body = None
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code, body)
        elif response.status_code == 429:
            raise RateLimitError(message, response.status_code, body)
        else:
            raise APIError(message, response.status_code, body)

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    class _ChatNamespace:
        """Chat API namespace."""

        def __init__(self, client: "LLMClient"):
            self._client = client
            self.completions = self._CompletionsNamespace(client)

        class _CompletionsNamespace:
            """Chat completions namespace."""

            def __init__(self, client: "LLMClient"):
                self._client = client

            def create(
                self,
                model: str,
                messages: list[Union[dict, ChatMessage]],
                temperature: float = 0.7,
                top_p: float = 1.0,
                n: int = 1,
                stream: bool = False,
                stop: Optional[Union[str, list[str]]] = None,
                max_tokens: Optional[int] = None,
                presence_penalty: float = 0.0,
                frequency_penalty: float = 0.0,
                response_format: Optional[dict] = None,
                **kwargs
            ) -> Union[ChatCompletion, Iterator[StreamChunk]]:
                """
                Create a chat completion.

                Args:
                    model: Model to use
                    messages: Conversation messages
                    temperature: Sampling temperature
                    top_p: Nucleus sampling parameter
                    n: Number of completions
                    stream: Stream responses
                    stop: Stop sequences
                    max_tokens: Maximum tokens to generate
                    presence_penalty: Presence penalty
                    frequency_penalty: Frequency penalty
                    response_format: Response format (e.g., {"type": "json_object"})

                Returns:
                    ChatCompletion or iterator of StreamChunks if streaming
                """
                # Convert messages to dict format
                msg_list = []
                for msg in messages:
                    if isinstance(msg, ChatMessage):
                        msg_list.append(msg.to_dict())
                    else:
                        msg_list.append(msg)

                payload = {
                    "model": model,
                    "messages": msg_list,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                    "stream": stream,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                }

                if stop:
                    payload["stop"] = stop
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                if response_format:
                    payload["response_format"] = response_format

                payload.update(kwargs)

                if stream:
                    return self._stream_completion(payload)

                response = self._client._request("POST", "/v1/chat/completions", payload)
                return ChatCompletion.from_dict(response)

            def _stream_completion(self, payload: dict) -> Iterator[StreamChunk]:
                """Stream chat completion."""
                for chunk_data in self._client._request("POST", "/v1/chat/completions", payload, stream=True):
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        yield StreamChunk(
                            id=chunk_data.get("id", ""),
                            object=chunk_data.get("object", ""),
                            created=chunk_data.get("created", 0),
                            model=chunk_data.get("model", ""),
                            delta_content=delta.get("content"),
                            delta_role=delta.get("role"),
                            finish_reason=choices[0].get("finish_reason")
                        )

    class _CompletionsNamespace:
        """Completions API namespace."""

        def __init__(self, client: "LLMClient"):
            self._client = client

        def create(
            self,
            model: str,
            prompt: Union[str, list[str]],
            max_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 1.0,
            n: int = 1,
            stream: bool = False,
            stop: Optional[Union[str, list[str]]] = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            echo: bool = False,
            **kwargs
        ) -> Union[Completion, Iterator[dict]]:
            """
            Create a completion.

            Args:
                model: Model to use
                prompt: Prompt(s) to complete
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature
                top_p: Nucleus sampling parameter
                n: Number of completions
                stream: Stream responses
                stop: Stop sequences
                presence_penalty: Presence penalty
                frequency_penalty: Frequency penalty
                echo: Echo prompt in response

            Returns:
                Completion or iterator if streaming
            """
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream": stream,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "echo": echo
            }

            if stop:
                payload["stop"] = stop

            payload.update(kwargs)

            if stream:
                return self._client._request("POST", "/v1/completions", payload, stream=True)

            response = self._client._request("POST", "/v1/completions", payload)
            return Completion.from_dict(response)

    class _ModelsNamespace:
        """Models API namespace."""

        def __init__(self, client: "LLMClient"):
            self._client = client

        def list(self) -> list[dict]:
            """List available models."""
            response = self._client._request("GET", "/v1/models")
            return response.get("data", [])

        def retrieve(self, model_id: str) -> dict:
            """Get model information."""
            return self._client._request("GET", f"/v1/models/{model_id}")


class AsyncLLMClient:
    """
    Asynchronous client for LLM Code Pipeline API.

    Same interface as LLMClient but with async/await support.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2
    ):
        self.base_url = (base_url or os.environ.get("LLM_API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._get_headers()
        )

        self.chat = self._ChatNamespace(self)
        self.completions = self._CompletionsNamespace(self)
        self.models = self._ModelsNamespace(self)

    def _get_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        stream: bool = False
    ) -> Union[dict, AsyncIterator[dict]]:
        url = endpoint if endpoint.startswith("/") else f"/{endpoint}"

        try:
            if stream:
                return self._stream_request(method, url, json_data)

            response = await self._client.request(method, url, json=json_data)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            self._handle_error(e.response)
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    async def _stream_request(
        self,
        method: str,
        url: str,
        json_data: Optional[dict]
    ) -> AsyncIterator[dict]:
        async with self._client.stream(method, url, json=json_data) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def _handle_error(self, response: httpx.Response) -> None:
        try:
            body = response.json()
            error = body.get("error", {})
            message = error.get("message", response.text)
        except:
            body = None
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code, body)
        elif response.status_code == 429:
            raise RateLimitError(message, response.status_code, body)
        else:
            raise APIError(message, response.status_code, body)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    class _ChatNamespace:
        def __init__(self, client: "AsyncLLMClient"):
            self._client = client
            self.completions = self._CompletionsNamespace(client)

        class _CompletionsNamespace:
            def __init__(self, client: "AsyncLLMClient"):
                self._client = client

            async def create(
                self,
                model: str,
                messages: list[Union[dict, ChatMessage]],
                stream: bool = False,
                **kwargs
            ) -> Union[ChatCompletion, AsyncIterator[StreamChunk]]:
                msg_list = [
                    msg.to_dict() if isinstance(msg, ChatMessage) else msg
                    for msg in messages
                ]

                payload = {"model": model, "messages": msg_list, "stream": stream, **kwargs}

                if stream:
                    return self._stream_completion(payload)

                response = await self._client._request("POST", "/v1/chat/completions", payload)
                return ChatCompletion.from_dict(response)

            async def _stream_completion(self, payload: dict) -> AsyncIterator[StreamChunk]:
                async for chunk_data in await self._client._request("POST", "/v1/chat/completions", payload, stream=True):
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        yield StreamChunk(
                            id=chunk_data.get("id", ""),
                            object=chunk_data.get("object", ""),
                            created=chunk_data.get("created", 0),
                            model=chunk_data.get("model", ""),
                            delta_content=delta.get("content"),
                            delta_role=delta.get("role"),
                            finish_reason=choices[0].get("finish_reason")
                        )

    class _CompletionsNamespace:
        def __init__(self, client: "AsyncLLMClient"):
            self._client = client

        async def create(self, model: str, prompt: Union[str, list[str]], stream: bool = False, **kwargs):
            payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

            if stream:
                return await self._client._request("POST", "/v1/completions", payload, stream=True)

            response = await self._client._request("POST", "/v1/completions", payload)
            return Completion.from_dict(response)

    class _ModelsNamespace:
        def __init__(self, client: "AsyncLLMClient"):
            self._client = client

        async def list(self) -> list[dict]:
            response = await self._client._request("GET", "/v1/models")
            return response.get("data", [])

        async def retrieve(self, model_id: str) -> dict:
            return await self._client._request("GET", f"/v1/models/{model_id}")


# Convenience function for quick usage
def create_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_client: bool = False
) -> Union[LLMClient, AsyncLLMClient]:
    """
    Create an LLM client.

    Args:
        base_url: API base URL
        api_key: API key
        async_client: Return async client if True

    Returns:
        LLMClient or AsyncLLMClient
    """
    if async_client:
        return AsyncLLMClient(base_url=base_url, api_key=api_key)
    return LLMClient(base_url=base_url, api_key=api_key)
