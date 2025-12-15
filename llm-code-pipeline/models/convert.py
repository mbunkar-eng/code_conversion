"""
Model Converter - Converts models between formats (FP16, INT4-AWQ, INT4-GPTQ).
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    INT8 = "int8"
    INT4_AWQ = "int4-awq"
    INT4_GPTQ = "int4-gptq"


@dataclass
class ConversionResult:
    """Result of a model conversion operation."""
    success: bool
    output_path: Optional[str]
    original_size_gb: Optional[float] = None
    converted_size_gb: Optional[float] = None
    error: Optional[str] = None


class ModelConverter:
    """Converts models between different formats and precisions."""

    def __init__(
        self,
        output_dir: str = "./converted_models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def convert_to_fp16(
        self,
        model_path: str,
        output_name: Optional[str] = None
    ) -> ConversionResult:
        """
        Convert model to FP16 format.

        Args:
            model_path: Path to the source model
            output_name: Name for the converted model directory

        Returns:
            ConversionResult with success status and output path
        """
        model_path = Path(model_path)
        if output_name is None:
            output_name = f"{model_path.name}-fp16"
        output_path = self.output_dir / output_name

        logger.info(f"Converting {model_path} to FP16...")

        try:
            # Load model in FP16
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

            # Save converted model
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path), safe_serialization=True)
            tokenizer.save_pretrained(str(output_path))

            original_size = self._get_dir_size_gb(model_path)
            converted_size = self._get_dir_size_gb(output_path)

            logger.info(f"Conversion complete: {original_size:.2f} GB -> {converted_size:.2f} GB")

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                original_size_gb=original_size,
                converted_size_gb=converted_size
            )

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return ConversionResult(
                success=False,
                output_path=None,
                error=str(e)
            )

    def convert_to_awq(
        self,
        model_path: str,
        output_name: Optional[str] = None,
        bits: int = 4,
        group_size: int = 128
    ) -> ConversionResult:
        """
        Convert model to AWQ INT4 quantized format.

        Requires: autoawq library

        Args:
            model_path: Path to the source model
            output_name: Name for the converted model directory
            bits: Quantization bits (default 4)
            group_size: AWQ group size (default 128)

        Returns:
            ConversionResult with success status and output path
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            return ConversionResult(
                success=False,
                output_path=None,
                error="autoawq not installed. Run: pip install autoawq"
            )

        model_path = Path(model_path)
        if output_name is None:
            output_name = f"{model_path.name}-awq-int{bits}"
        output_path = self.output_dir / output_name

        logger.info(f"Converting {model_path} to AWQ INT{bits}...")

        try:
            # Load model for AWQ quantization
            model = AutoAWQForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

            # Quantization config
            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": bits,
                "version": "GEMM"
            }

            # Quantize
            model.quantize(tokenizer, quant_config=quant_config)

            # Save
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_quantized(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            original_size = self._get_dir_size_gb(model_path)
            converted_size = self._get_dir_size_gb(output_path)

            logger.info(f"AWQ conversion complete: {original_size:.2f} GB -> {converted_size:.2f} GB")

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                original_size_gb=original_size,
                converted_size_gb=converted_size
            )

        except Exception as e:
            logger.error(f"AWQ conversion failed: {e}")
            return ConversionResult(
                success=False,
                output_path=None,
                error=str(e)
            )

    def convert_to_gptq(
        self,
        model_path: str,
        output_name: Optional[str] = None,
        bits: int = 4,
        group_size: int = 128,
        dataset: str = "c4"
    ) -> ConversionResult:
        """
        Convert model to GPTQ INT4 quantized format.

        Requires: auto-gptq library

        Args:
            model_path: Path to the source model
            output_name: Name for the converted model directory
            bits: Quantization bits (default 4)
            group_size: GPTQ group size (default 128)
            dataset: Calibration dataset

        Returns:
            ConversionResult with success status and output path
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            return ConversionResult(
                success=False,
                output_path=None,
                error="auto-gptq not installed. Run: pip install auto-gptq"
            )

        model_path = Path(model_path)
        if output_name is None:
            output_name = f"{model_path.name}-gptq-int{bits}"
        output_path = self.output_dir / output_name

        logger.info(f"Converting {model_path} to GPTQ INT{bits}...")

        try:
            # Quantization config
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=False
            )

            # Load model
            model = AutoGPTQForCausalLM.from_pretrained(
                str(model_path),
                quantize_config=quantize_config,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

            # Load calibration data
            from datasets import load_dataset
            calibration_data = load_dataset(dataset, split="train[:1000]")
            calibration_data = [tokenizer(text["text"]) for text in calibration_data]

            # Quantize
            model.quantize(calibration_data)

            # Save
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_quantized(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            original_size = self._get_dir_size_gb(model_path)
            converted_size = self._get_dir_size_gb(output_path)

            logger.info(f"GPTQ conversion complete: {original_size:.2f} GB -> {converted_size:.2f} GB")

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                original_size_gb=original_size,
                converted_size_gb=converted_size
            )

        except Exception as e:
            logger.error(f"GPTQ conversion failed: {e}")
            return ConversionResult(
                success=False,
                output_path=None,
                error=str(e)
            )

    def optimize_for_vllm(
        self,
        model_path: str,
        output_name: Optional[str] = None
    ) -> ConversionResult:
        """
        Optimize model for vLLM inference.
        Converts to safetensors format with proper config.

        Args:
            model_path: Path to the source model
            output_name: Name for the optimized model directory

        Returns:
            ConversionResult with success status and output path
        """
        model_path = Path(model_path)
        if output_name is None:
            output_name = f"{model_path.name}-vllm"
        output_path = self.output_dir / output_name

        logger.info(f"Optimizing {model_path} for vLLM...")

        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

            # Save in safetensors format
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path), safe_serialization=True)
            tokenizer.save_pretrained(str(output_path))
            config.save_pretrained(str(output_path))

            original_size = self._get_dir_size_gb(model_path)
            converted_size = self._get_dir_size_gb(output_path)

            logger.info(f"vLLM optimization complete: {converted_size:.2f} GB")

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                original_size_gb=original_size,
                converted_size_gb=converted_size
            )

        except Exception as e:
            logger.error(f"vLLM optimization failed: {e}")
            return ConversionResult(
                success=False,
                output_path=None,
                error=str(e)
            )

    def _get_dir_size_gb(self, path: Path) -> float:
        """Get directory size in GB."""
        total_size = 0
        for file in path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 ** 3)


def main():
    """CLI for model conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert LLM models between formats")
    parser.add_argument("model_path", help="Path to source model")
    parser.add_argument("--output-dir", "-o", default="./converted_models", help="Output directory")
    parser.add_argument("--format", "-f", choices=["fp16", "awq", "gptq", "vllm"], default="fp16",
                        help="Target format")
    parser.add_argument("--bits", "-b", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group-size", "-g", type=int, default=128, help="Quantization group size")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    converter = ModelConverter(output_dir=args.output_dir)

    if args.format == "fp16":
        result = converter.convert_to_fp16(args.model_path)
    elif args.format == "awq":
        result = converter.convert_to_awq(args.model_path, bits=args.bits, group_size=args.group_size)
    elif args.format == "gptq":
        result = converter.convert_to_gptq(args.model_path, bits=args.bits, group_size=args.group_size)
    elif args.format == "vllm":
        result = converter.optimize_for_vllm(args.model_path)

    if result.success:
        print(f"\nConversion successful!")
        print(f"Output: {result.output_path}")
        print(f"Size: {result.original_size_gb:.2f} GB -> {result.converted_size_gb:.2f} GB")
    else:
        print(f"\nConversion failed: {result.error}")


if __name__ == "__main__":
    main()
