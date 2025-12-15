#!/usr/bin/env python3
"""
Test real inference using Transformers on CPU/MPS.

This script downloads a small model and runs real inference.
Works on Mac with Apple Silicon (MPS) or CPU.

Usage:
    python test_real_inference.py

The first run will download the model (~500MB for phi-2).
"""

import sys
import time
sys.path.insert(0, '.')

import torch

def main():
    print("=" * 60)
    print("Real Inference Test with Transformers")
    print("=" * 60)
    print()

    # Check available device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Using: CPU")
    print()

    # Use a small model for testing
    # Options:
    # - "microsoft/phi-2" (~2.7B params, ~5GB)
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (~1.1B params, ~2GB)
    # - "Qwen/Qwen2-0.5B-Instruct" (~0.5B params, ~1GB) - smallest

    model_name = "Qwen/Qwen2-0.5B-Instruct"  # Smallest option
    print(f"Model: {model_name}")
    print("Note: First run will download the model (~1GB)")
    print()

    try:
        from inference.transformers_runner import TransformersRunner
        from inference.vllm_runner import GenerationConfig

        print("[1/4] Initializing model...")
        start = time.time()

        runner = TransformersRunner(
            model_path=model_name,
            device=device,
            torch_dtype="float32" if device == "cpu" else "float16"
        )
        runner.initialize()

        load_time = time.time() - start
        print(f"      Model loaded in {load_time:.1f}s")
        print()

        # Test 1: Basic generation
        print("[2/4] Testing basic generation...")
        config = GenerationConfig(max_tokens=50, temperature=0.7)

        start = time.time()
        result = runner.generate("def fibonacci(n):", config)
        gen_time = time.time() - start

        print(f"      Prompt: def fibonacci(n):")
        print(f"      Response: {result.text[:100]}...")
        print(f"      Time: {gen_time:.2f}s ({result.tokens_per_second:.1f} tokens/s)")
        print()

        # Test 2: Code conversion
        print("[3/4] Testing code conversion...")
        prompt = """Convert this Python code to JavaScript:

def greet(name):
    return f"Hello, {name}!"

JavaScript:"""

        config = GenerationConfig(max_tokens=100, temperature=0.3)
        start = time.time()
        result = runner.generate(prompt, config)
        gen_time = time.time() - start

        print(f"      Response: {result.text[:150]}...")
        print(f"      Time: {gen_time:.2f}s")
        print()

        # Test 3: Chat-style prompt
        print("[4/4] Testing chat-style prompt...")
        prompt = "<|im_start|>user\nWrite a Python function to check if a number is prime.<|im_end|>\n<|im_start|>assistant\n"

        config = GenerationConfig(max_tokens=150, temperature=0.5)
        start = time.time()
        result = runner.generate(prompt, config)
        gen_time = time.time() - start

        print(f"      Response: {result.text[:200]}...")
        print(f"      Time: {gen_time:.2f}s")
        print()

        # Cleanup
        runner.shutdown()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  - Device: {device}")
        print(f"  - Model: {model_name}")
        print(f"  - Model Load Time: {load_time:.1f}s")
        print()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure transformers is installed: pip install transformers")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
