"""
Configuration settings for LLM Code Pipeline.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Authentication
    api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    api_keys: Optional[str] = Field(default=None, env="LLM_API_KEYS")
    api_keys_file: Optional[str] = Field(default=None, env="LLM_API_KEYS_FILE")

    # Model Settings
    model_path: Optional[str] = Field(default=None, env="MODEL_PATH")
    models_dir: str = Field(default="./downloaded_models", env="MODELS_DIR")
    default_model: str = Field(default="qwen2.5-coder-7b", env="DEFAULT_MODEL")

    # Inference Settings
    tensor_parallel_size: int = Field(default=1, env="TENSOR_PARALLEL_SIZE")
    gpu_memory_utilization: float = Field(default=0.9, env="GPU_MEMORY_UTILIZATION")
    max_model_len: Optional[int] = Field(default=None, env="MAX_MODEL_LEN")
    dtype: str = Field(default="float16", env="DTYPE")
    quantization: Optional[str] = Field(default=None, env="QUANTIZATION")

    # Generation Defaults
    default_max_tokens: int = Field(default=2048, env="DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_top_p: float = Field(default=0.95, env="DEFAULT_TOP_P")

    # HuggingFace
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    hf_cache_dir: Optional[str] = Field(default=None, env="HF_HOME")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    json_logs: bool = Field(default=False, env="JSON_LOGS")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # CORS
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")

    # Mode
    mock_mode: bool = Field(default=False, env="LLM_PIPELINE_MOCK_MODE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
