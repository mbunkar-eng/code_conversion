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

    def get_or_load_model(self, model_id: str) -> tuple:
        """
        Get a loaded model or download and load it if not available.

        Args:
            model_id: Model identifier (registry name or HF repo)

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

            # Download if not exists
            if not Path(model_path).exists():
                logger.info(f"Model {model_id} not found locally, downloading...")
                result = self.downloader.download_model(model_id)
                if not result.success:
                    raise RuntimeError(f"Failed to download model {model_id}: {result.error}")
                model_path = result.model_path
                logger.info(f"Downloaded model to: {model_path}")

            # Load the model
            logger.info(f"Loading model: {model_id} from {model_path}")
            try:
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