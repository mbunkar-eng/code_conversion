"""
Model Manager - Handles dynamic loading and management of multiple models.
"""

import logging
import os
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from models.download import ModelDownloader
from models.registry import ModelRegistry
from inference.transformers_runner import TransformersRunner
from inference.tokenizer_service import TokenizerService
from inference.vllm_runner import GenerationResult

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple models dynamically - downloads and loads them on demand.
    """

    def __init__(
        self,
        models_dir: str = "./downloaded_models",
        hf_token: Optional[str] = None
    ):
        self.models_dir = Path(models_dir)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Initialize components
        self.registry = ModelRegistry()
        self.downloader = ModelDownloader(
            models_dir=str(self.models_dir),
            hf_token=self.hf_token,
            registry=self.registry
        )

        # Model cache: model_id -> (runner, tokenizer_service)
        self._loaded_models: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    @property
    def mock_mode(self) -> bool:
        """Check if running in mock mode (dynamic check)"""
        return os.environ.get("LLM_PIPELINE_MOCK_MODE", "false").lower() == "true"

    def get_or_load_model(self, model_id: str) -> tuple:
        """
        Get a loaded model from local storage only (no downloading).

        Args:
            model_id: Model identifier (registry name or local path)

        Returns:
            Tuple of (inference_runner, tokenizer_service)
        """
        with self._lock:
            # Check if already loaded
            if model_id in self._loaded_models:
                logger.info(f"Using cached model: {model_id}")
                return self._loaded_models[model_id]

            # Determine model path
            model_path = self._get_model_path(model_id)

            # Check if model exists locally and is complete
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise RuntimeError(f"Model {model_id} not found locally at {model_path}")
            
            # Skip completeness check in mock mode
            if not self.mock_mode and not self.downloader._is_model_complete(model_path_obj):
                raise RuntimeError(f"Model {model_id} exists but is incomplete at {model_path}")

            # Load the model
            logger.info(f"Loading model: {model_id} from {model_path}")
            try:
                # Check if in mock mode - return mock objects
                if self.mock_mode:
                    logger.info(f"Mock mode: Creating mock runner and tokenizer for {model_id}")
                    # Create mock objects that behave like the real ones
                    class MockRunner:
                        def generate(self, prompt, config=None):
                            import random
                            import time
                            mock_responses = [
                                "This is a mock response from the LLM. The model is running in mock mode for testing purposes.",
                                "Mock mode activated! This response is generated without loading the actual model.",
                                "Hello! I'm responding in mock mode. The real model would provide more detailed and accurate responses.",
                                "Mock response: Your query has been processed successfully. In a real deployment, this would be answered by the actual language model."
                            ]
                            mock_text = random.choice(mock_responses)
                            prompt_tokens = len(prompt.split()) if isinstance(prompt, str) else sum(len(p.split()) for p in prompt)
                            completion_tokens = len(mock_text.split())
                            total_tokens = prompt_tokens + completion_tokens
                            
                            return GenerationResult(
                                text=mock_text,
                                finish_reason="stop",
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                generation_time_ms=50.0,  # Mock generation time
                                tokens_per_second=total_tokens / 0.05  # Mock tokens per second
                            )
                        
                        async def generate_stream(self, prompt, config=None):
                            # For simplicity, just return the non-streaming result
                            yield self.generate(prompt, config)
                        
                        def get_model_info(self) -> dict:
                            """Get mock model information."""
                            return {
                                "model_path": "mock_model",
                                "device": "mock",
                                "dtype": "mock",
                                "status": "mock_initialized",
                                "backend": "mock"
                            }
                    
                    class MockTokenizer:
                        def encode(self, text, **kwargs):
                            from inference.tokenizer_service import TokenizationResult
                            words = text.split()
                            return TokenizationResult(
                                tokens=list(range(1, len(words) + 1)),
                                token_count=len(words),
                                text=text
                            )
                        
                        def apply_chat_template(self, messages, **kwargs):
                            formatted = []
                            for msg in messages:
                                formatted.append(f"{msg.role.upper()}: {msg.content}")
                            result = "\n".join(formatted)
                            if kwargs.get('add_generation_prompt', False):
                                result += "\nASSISTANT:"
                            return result
                    
                    runner = MockRunner()
                    tokenizer_service = MockTokenizer()
                else:
                    # Initialize runner
                    runner = TransformersRunner(
                        model_path=model_path,
                        device="auto",
                        torch_dtype="float16"
                    )
                    runner.initialize()

                    # Initialize tokenizer service
                    tokenizer_service = TokenizerService(model_path)
                    tokenizer_service.initialize()

                # Cache the loaded model
                self._loaded_models[model_id] = (runner, tokenizer_service)

                logger.info(f"Successfully loaded model: {model_id}")
                return runner, tokenizer_service

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")

    def get_available_local_models(self) -> list[str]:
        """
        Get list of locally available complete models.
        
        Returns:
            List of model identifiers that can be loaded
        """
        available_models = []
        if not self.models_dir.exists():
            return available_models
            
        for item in self.models_dir.iterdir():
            if item.is_dir():
                model_name = item.name
                # In mock mode, consider all directories as available models
                if self.mock_mode or self.downloader._is_model_complete(item):
                    available_models.append(model_name)
        
        return available_models

    def get_default_model(self) -> str:
        """
        Get the best available local model.
        
        Returns:
            Model identifier to use as default
        """
        available = self.get_available_local_models()
        if not available:
            raise RuntimeError("No complete models available locally")
        
        # Prefer smaller models first for faster loading
        preferences = [
            "Qwen--Qwen2-0.5B-Instruct",  # Smallest and fastest
            "Qwen--Qwen2.5-Coder-7B-Instruct",  # 7B model - good balance
            "deepseek-ai--deepseek-coder-6.7b-instruct"  # Available local model
        ]
        
        for pref in preferences:
            if pref in available:
                return pref
                
        # Fallback to first available
        return available[0]

    def _get_model_path(self, model_id: str) -> str:
        """
        Get the local path for a model.

        Args:
            model_id: Model identifier

        Returns:
            Local path to the model
        """
        # Check if it's a registry model
        model_info = self.registry.get_model(model_id)
        if model_info:
            # Convert HF repo to local path format
            hf_repo = model_info.hf_repo
            model_name = hf_repo.replace("/", "--")
            return str(self.models_dir / model_name)

        # Check if it's already in local format (contains --)
        if "--" in model_id:
            # Assume it's already the local folder name
            return str(self.models_dir / model_id)

        # Assume it's a direct HF repo name
        if "/" in model_id:
            model_name = model_id.replace("/", "--")
            return str(self.models_dir / model_name)

        # Fallback: treat as local path
        return model_id

    def list_loaded_models(self) -> list[str]:
        """List currently loaded model IDs."""
        with self._lock:
            return list(self._loaded_models.keys())

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded, False if not found
        """
        with self._lock:
            if model_id in self._loaded_models:
                runner, tokenizer = self._loaded_models[model_id]
                # Clean up resources if needed
                if hasattr(runner, 'shutdown'):
                    runner.shutdown()
                del self._loaded_models[model_id]
                logger.info(f"Unloaded model: {model_id}")
                return True
            return False

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a loaded model.

        Args:
            model_id: Model identifier

        Returns:
            Model info dict or None if not loaded
        """
        with self._lock:
            if model_id in self._loaded_models:
                runner, _ = self._loaded_models[model_id]
                return runner.get_model_info()
            return None