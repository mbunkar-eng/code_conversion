"""
vLLM Runner - Manages vLLM inference engine.
"""

import asyncio
import logging
import time
from typing import Optional, AsyncGenerator, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    n: int = 1
    best_of: Optional[int] = None
    logprobs: Optional[int] = None
    echo: bool = False


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time_ms: float
    tokens_per_second: float


class VLLMRunner:
    """
    Manages vLLM inference engine for code generation models.

    Supports both synchronous and asynchronous generation,
    with streaming capabilities.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        seed: int = 42
    ):
        """
        Initialize vLLM runner.

        Args:
            model_path: Path to the model or HuggingFace repo
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            dtype: Data type (float16, bfloat16, float32)
            quantization: Quantization method (awq, gptq, None)
            trust_remote_code: Trust remote code in model
            seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.seed = seed

        self._engine = None
        self._sampling_params_class = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the vLLM engine."""
        if self._initialized:
            return

        try:
            from vllm import LLM, SamplingParams
            self._sampling_params_class = SamplingParams

            logger.info(f"Initializing vLLM with model: {self.model_path}")

            self._engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                quantization=self.quantization,
                trust_remote_code=self.trust_remote_code,
                seed=self.seed
            )

            self._initialized = True
            logger.info("vLLM engine initialized successfully")

        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def generate(
        self,
        prompt: Union[str, list[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, list[GenerationResult]]:
        """
        Generate text synchronously.

        Args:
            prompt: Input prompt(s)
            config: Generation configuration

        Returns:
            GenerationResult or list of GenerationResults
        """
        if not self._initialized:
            self.initialize()

        config = config or GenerationConfig()
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        sampling_params = self._sampling_params_class(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop if config.stop else None,
            n=config.n,
            best_of=config.best_of,
            logprobs=config.logprobs
        )

        start_time = time.perf_counter()
        outputs = self._engine.generate(prompts, sampling_params)
        generation_time_ms = (time.perf_counter() - start_time) * 1000

        results = []
        for output in outputs:
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = (completion_tokens / generation_time_ms) * 1000

            results.append(GenerationResult(
                text=output.outputs[0].text,
                finish_reason=output.outputs[0].finish_reason or "stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time_ms,
                tokens_per_second=tokens_per_second
            ))

        return results if is_batch else results[0]

    async def generate_async(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text asynchronously.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            GenerationResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, config)

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Generated text chunks
        """
        if not self._initialized:
            self.initialize()

        config = config or GenerationConfig()

        try:
            from vllm import SamplingParams
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            # For streaming, we need async engine
            # This is a simplified implementation
            # In production, use AsyncLLMEngine directly

            sampling_params = SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=config.stop if config.stop else None
            )

            # Generate full response and simulate streaming
            result = await self.generate_async(prompt, config)

            # Yield in chunks
            chunk_size = 10
            text = result.text
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "status": "initialized"
        }

    def shutdown(self) -> None:
        """Shutdown the vLLM engine."""
        if self._engine is not None:
            del self._engine
            self._engine = None
            self._initialized = False
            logger.info("vLLM engine shutdown complete")


class MockVLLMRunner(VLLMRunner):
    """
    Mock vLLM runner for testing without GPU.
    Returns simulated responses.
    """

    def initialize(self) -> None:
        """Initialize mock runner."""
        self._initialized = True
        logger.info("Mock vLLM runner initialized")

    def generate(
        self,
        prompt: Union[str, list[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, list[GenerationResult]]:
        """Generate mock response."""
        config = config or GenerationConfig()
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        results = []
        for p in prompts:
            # Generate mock code response
            mock_response = self._generate_mock_code(p)

            results.append(GenerationResult(
                text=mock_response,
                finish_reason="stop",
                prompt_tokens=len(p.split()),
                completion_tokens=len(mock_response.split()),
                total_tokens=len(p.split()) + len(mock_response.split()),
                generation_time_ms=100.0,
                tokens_per_second=50.0
            ))

        return results if is_batch else results[0]

    def _generate_mock_code(self, prompt: str) -> str:
        """Generate mock code based on prompt."""
        prompt_lower = prompt.lower()

        if "python" in prompt_lower and "java" in prompt_lower:
            return '''public class Solution {
    public static void main(String[] args) {
        System.out.println("Converted from Python");
    }
}'''
        elif "javascript" in prompt_lower or "typescript" in prompt_lower:
            return '''function solution(): void {
    console.log("Converted code");
}

export { solution };'''
        elif "go" in prompt_lower:
            return '''package main

import "fmt"

func main() {
    fmt.Println("Converted to Go")
}'''
        else:
            return '''// Generated code
function example() {
    return "Hello, World!";
}'''
