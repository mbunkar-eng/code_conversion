"""
Model Registry - Manages model metadata and configurations.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Model information dataclass."""
    name: str
    hf_repo: str
    model_type: str
    parameters: str
    context_length: int
    supported_formats: list[str]
    default_format: str
    gpu_memory_required_gb: float
    recommended_gpu: str
    quantization: Optional[str] = None
    bits: Optional[int] = None
    base_model: Optional[str] = None


class ModelRegistry:
    """Registry for managing available models and their configurations."""

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent / "registry.json"
        self.registry_path = Path(registry_path)
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from JSON file."""
        with open(self.registry_path, "r") as f:
            self._registry = json.load(f)
        self._models = self._registry.get("models", {})
        self._quantized_models = self._registry.get("quantized_models", {})
        self._default_model = self._registry.get("default_model", "qwen2.5-coder-7b")

    def list_models(self) -> list[str]:
        """List all available model identifiers."""
        return list(self._models.keys()) + list(self._quantized_models.keys())

    def list_base_models(self) -> list[str]:
        """List only base (non-quantized) models."""
        return list(self._models.keys())

    def list_quantized_models(self) -> list[str]:
        """List only quantized models."""
        return list(self._quantized_models.keys())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        if model_id in self._models:
            data = self._models[model_id]
            return ModelInfo(
                name=data["name"],
                hf_repo=data["hf_repo"],
                model_type=data["type"],
                parameters=data["parameters"],
                context_length=data["context_length"],
                supported_formats=data["supported_formats"],
                default_format=data["default_format"],
                gpu_memory_required_gb=data["gpu_memory_required_gb"],
                recommended_gpu=data["recommended_gpu"]
            )
        elif model_id in self._quantized_models:
            data = self._quantized_models[model_id]
            base_data = self._models.get(data["base_model"], {})
            return ModelInfo(
                name=f"{base_data.get('name', model_id)} ({data['quantization']})",
                hf_repo=data["hf_repo"],
                model_type=base_data.get("type", "code"),
                parameters=base_data.get("parameters", "unknown"),
                context_length=base_data.get("context_length", 16384),
                supported_formats=[f"int{data['bits']}-{data['quantization'].lower()}"],
                default_format=f"int{data['bits']}-{data['quantization'].lower()}",
                gpu_memory_required_gb=data["gpu_memory_required_gb"],
                recommended_gpu="RTX 3090/4090",
                quantization=data["quantization"],
                bits=data["bits"],
                base_model=data["base_model"]
            )
        return None

    def get_default_model(self) -> str:
        """Get the default model identifier."""
        return self._default_model

    def get_hf_repo(self, model_id: str) -> Optional[str]:
        """Get HuggingFace repository for a model."""
        model = self.get_model(model_id)
        return model.hf_repo if model else None

    def get_memory_requirement(self, model_id: str) -> Optional[float]:
        """Get GPU memory requirement in GB for a model."""
        model = self.get_model(model_id)
        return model.gpu_memory_required_gb if model else None

    def validate_model(self, model_id: str) -> bool:
        """Check if a model ID is valid."""
        return model_id in self._models or model_id in self._quantized_models

    def add_custom_model(
        self,
        model_id: str,
        name: str,
        hf_repo: str,
        parameters: str,
        context_length: int = 16384,
        gpu_memory_gb: float = 14,
        save: bool = True
    ) -> None:
        """Add a custom model to the registry."""
        self._models[model_id] = {
            "name": name,
            "hf_repo": hf_repo,
            "type": "code",
            "parameters": parameters,
            "context_length": context_length,
            "supported_formats": ["fp16"],
            "default_format": "fp16",
            "gpu_memory_required_gb": gpu_memory_gb,
            "recommended_gpu": "A100-40GB"
        }
        if save:
            self._save_registry()

    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        self._registry["models"] = self._models
        self._registry["quantized_models"] = self._quantized_models
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
