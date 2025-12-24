"""
Model Downloader - Downloads models from HuggingFace Hub.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from tqdm import tqdm

from .registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a model download operation."""
    success: bool
    model_path: Optional[str]
    error: Optional[str] = None
    size_gb: Optional[float] = None


class ModelDownloader:
    """Downloads models from HuggingFace Hub."""

    def __init__(
        self,
        models_dir: str = "./downloaded_models",
        hf_token: Optional[str] = None,
        registry: Optional[ModelRegistry] = None
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.registry = registry or ModelRegistry()
        self.api = HfApi(token=self.hf_token)

    def download_model(
        self,
        model_id: str,
        revision: str = "main",
        force_download: bool = False,
        resume_download: bool = True
    ) -> DownloadResult:
        """
        Download a model from HuggingFace Hub.

        Args:
            model_id: Model identifier from registry or HF repo name
            revision: Git revision (branch, tag, or commit)
            force_download: Force re-download even if exists
            resume_download: Resume interrupted downloads

        Returns:
            DownloadResult with success status and model path
        """
        # Get HF repo from registry or use model_id directly
        hf_repo = self.registry.get_hf_repo(model_id)
        if hf_repo is None:
            hf_repo = model_id
            logger.warning(f"Model {model_id} not in registry, using as HF repo directly")

        # Create model directory
        model_name = hf_repo.replace("/", "--")
        model_path = self.models_dir / model_name

        # Check if model is already fully downloaded
        if model_path.exists() and not force_download:
            if self._is_model_complete(model_path):
                logger.info(f"Model already exists and is complete at {model_path}")
                return DownloadResult(
                    success=True,
                    model_path=str(model_path),
                    size_gb=self._get_dir_size_gb(model_path)
                )
            else:
                logger.warning(f"Model directory exists but appears incomplete, re-downloading: {model_path}")

        logger.info(f"Downloading model {hf_repo} to {model_path}")

        try:
            snapshot_download(
                repo_id=hf_repo,
                revision=revision,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=resume_download,
                token=self.hf_token,
                ignore_patterns=["*.md", "*.txt", "*.png", "*.jpg"]
            )

            size_gb = self._get_dir_size_gb(model_path)
            logger.info(f"Successfully downloaded {hf_repo} ({size_gb:.2f} GB)")

            return DownloadResult(
                success=True,
                model_path=str(model_path),
                size_gb=size_gb
            )

        except Exception as e:
            logger.error(f"Failed to download model {hf_repo}: {e}")
            return DownloadResult(
                success=False,
                model_path=None,
                error=str(e)
            )

    def download_tokenizer(
        self,
        model_id: str,
        revision: str = "main"
    ) -> DownloadResult:
        """Download only the tokenizer files for a model."""
        hf_repo = self.registry.get_hf_repo(model_id) or model_id
        model_name = hf_repo.replace("/", "--")
        tokenizer_path = self.models_dir / f"{model_name}-tokenizer"

        try:
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt"
            ]

            tokenizer_path.mkdir(parents=True, exist_ok=True)

            for file in tokenizer_files:
                try:
                    hf_hub_download(
                        repo_id=hf_repo,
                        filename=file,
                        revision=revision,
                        local_dir=str(tokenizer_path),
                        token=self.hf_token
                    )
                except Exception:
                    continue

            return DownloadResult(
                success=True,
                model_path=str(tokenizer_path)
            )

        except Exception as e:
            return DownloadResult(
                success=False,
                model_path=None,
                error=str(e)
            )

    def get_model_info(self, model_id: str) -> dict:
        """Get model information from HuggingFace Hub."""
        hf_repo = self.registry.get_hf_repo(model_id) or model_id
        try:
            info = self.api.model_info(hf_repo, token=self.hf_token)
            return {
                "id": info.id,
                "sha": info.sha,
                "created_at": str(info.created_at) if info.created_at else None,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag
            }
        except Exception as e:
            return {"error": str(e)}

    def list_downloaded_models(self) -> list[dict]:
        """List all downloaded models."""
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_gb": self._get_dir_size_gb(model_dir)
                })
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        hf_repo = self.registry.get_hf_repo(model_id) or model_id
        model_name = hf_repo.replace("/", "--")
        model_path = self.models_dir / model_name

        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            logger.info(f"Deleted model at {model_path}")
            return True
        return False

    def _is_model_complete(self, model_path: Path) -> bool:
        """Check if a model directory contains all necessary files."""
        # Check for essential files
        essential_files = ["config.json", "tokenizer_config.json"]
        for file in essential_files:
            if not (model_path / file).exists():
                return False

        # Check for model weights (either safetensors or pytorch)
        has_weights = False
        for pattern in ["*.safetensors", "*.bin", "pytorch_model.bin"]:
            if list(model_path.glob(pattern)):
                has_weights = True
                break

        if not has_weights:
            return False

        # If there's an index file, check that all referenced files exist
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            try:
                import json
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                weight_map = index.get("weight_map", {})
                referenced_files = set(weight_map.values())
                
                for ref_file in referenced_files:
                    if not (model_path / ref_file).exists():
                        logger.warning(f"Missing referenced model file: {ref_file}")
                        return False
            except Exception as e:
                logger.warning(f"Failed to parse model index file: {e}")
                return False

        return True

    def _get_dir_size_gb(self, path: Path) -> float:
        """Get directory size in GB."""
        total_size = 0
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 ** 3)


def main():
    """CLI for model downloading."""
    import argparse

    parser = argparse.ArgumentParser(description="Download LLM models from HuggingFace")
    parser.add_argument("model_id", help="Model ID from registry or HF repo")
    parser.add_argument("--output-dir", "-o", default="./downloaded_models", help="Output directory")
    parser.add_argument("--token", "-t", help="HuggingFace token")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    parser.add_argument("--list", "-l", action="store_true", help="List downloaded models")
    parser.add_argument("--info", "-i", action="store_true", help="Show model info")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    downloader = ModelDownloader(models_dir=args.output_dir, hf_token=args.token)

    if args.list:
        models = downloader.list_downloaded_models()
        print("\nDownloaded Models:")
        for m in models:
            print(f"  - {m['name']} ({m['size_gb']:.2f} GB)")
        return

    if args.info:
        info = downloader.get_model_info(args.model_id)
        print(f"\nModel Info for {args.model_id}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    result = downloader.download_model(args.model_id, force_download=args.force)
    if result.success:
        print(f"\nModel downloaded to: {result.model_path}")
        print(f"Size: {result.size_gb:.2f} GB")
    else:
        print(f"\nDownload failed: {result.error}")


if __name__ == "__main__":
    main()
