"""
Transformers Runner - CPU/MPS inference using HuggingFace Transformers.

This runner is for Mac/CPU inference where vLLM is not available.
It's slower but works on any machine.
"""

import logging
import time
from typing import Optional, Union, AsyncGenerator
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .vllm_runner import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class TransformersRunner:
    """
    Inference runner using HuggingFace Transformers.

    Works on CPU and MPS (Apple Silicon) devices.
    Slower than vLLM but more portable.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        trust_remote_code: bool = True,
        max_memory: Optional[dict] = None
    ):
        """
        Initialize Transformers runner.

        Args:
            model_path: Path to model or HuggingFace repo
            device: Device to use ('cpu', 'mps', 'cuda', or None for auto)
            torch_dtype: Data type (float16, bfloat16, float32)
            trust_remote_code: Trust remote code in model
            max_memory: Maximum memory per device
        """
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Set dtype based on device
        if torch_dtype == "float16":
            if self.device == "cpu":
                self.torch_dtype = torch.float32  # CPU doesn't support float16 well
            else:
                self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.max_memory = max_memory
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the model and tokenizer."""
        if self._initialized:
            return

        logger.info(f"Loading model {self.model_path} on {self.device}...")
        start_time = time.perf_counter()

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": self.torch_dtype,
            }

            if self.device != "cpu":
                model_kwargs["device_map"] = self.device

            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            if self.device == "cpu":
                self._model = self._model.to("cpu")

            # Create pipeline for easier generation
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=None if self.device != "cpu" else -1  # -1 for CPU
            )

            load_time = time.perf_counter() - start_time
            self._initialized = True
            logger.info(f"Model loaded in {load_time:.2f}s on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: Union[str, list[str]],
        config: Optional[GenerationConfig] = None
    ) -> Union[GenerationResult, list[GenerationResult]]:
        """
        Generate text from prompt(s).

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

        results = []

        for p in prompts:
            start_time = time.perf_counter()

            # Tokenize for token counting
            input_ids = self._tokenizer.encode(p, return_tensors="pt")
            prompt_tokens = len(input_ids[0])

            # Generate
            generation_kwargs = {
                "max_new_tokens": config.max_tokens,
                "temperature": max(config.temperature, 0.01),  # Avoid 0
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "do_sample": config.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            if config.stop:
                # Handle stop sequences (basic implementation)
                pass

            outputs = self._pipeline(
                p,
                **generation_kwargs,
                return_full_text=False
            )

            generated_text = outputs[0]["generated_text"]
            generation_time_ms = (time.perf_counter() - start_time) * 1000

            # Count completion tokens
            completion_ids = self._tokenizer.encode(generated_text, add_special_tokens=False)
            completion_tokens = len(completion_ids)

            tokens_per_second = (completion_tokens / generation_time_ms) * 1000 if generation_time_ms > 0 else 0

            results.append(GenerationResult(
                text=generated_text,
                finish_reason="stop",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                generation_time_ms=generation_time_ms,
                tokens_per_second=tokens_per_second
            ))

        return results if is_batch else results[0]

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming (simulated).

        Note: True streaming requires TextIteratorStreamer which is more complex.
        This implementation generates full response then yields in chunks.
        """
        result = self.generate(prompt, config)

        # Yield in chunks
        chunk_size = 5
        text = result.text
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "status": "initialized",
            "backend": "transformers"
        }

    def shutdown(self) -> None:
        """Shutdown and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._initialized = False

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Transformers runner shutdown complete")


def test_transformers_runner():
    """Quick test of the transformers runner."""
    # Use a very small model for testing
    print("Testing Transformers Runner...")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")

    # This would require downloading a model - skip in test
    print("Note: Full test requires downloading a model.")
    print("To test, run:")
    print('  runner = TransformersRunner("microsoft/phi-2")')
    print('  runner.initialize()')
    print('  result = runner.generate("def hello():", GenerationConfig(max_tokens=50))')
    print('  print(result.text)')


if __name__ == "__main__":
    test_transformers_runner()
