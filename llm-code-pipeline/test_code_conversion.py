#!/usr/bin/env python3
"""
Test real code conversion use cases with actual LLM model.

Demonstrates:
- Python to Java conversion
- Python to TypeScript conversion
- JavaScript to Python conversion
- Code generation from natural language

Uses Qwen2-0.5B-Instruct model on Apple Silicon (MPS).
"""

import sys
import time
import os

# Enable MPS fallback for unsupported operators
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.insert(0, '.')

import torch

def main():
    print("=" * 70)
    print("Real Code Conversion Tests with Qwen2-0.5B-Instruct")
    print("=" * 70)
    print()

    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("Device: CPU")
    print()

    try:
        from inference.transformers_runner import TransformersRunner
        from inference.vllm_runner import GenerationConfig

        # Initialize model
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        print(f"Loading model: {model_name}")
        print("(First run downloads ~1GB model)")
        print()

        start = time.time()
        runner = TransformersRunner(
            model_path=model_name,
            device=device,
            torch_dtype="float32" if device == "cpu" else "float16"
        )
        runner.initialize()
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.1f}s")
        print()

        # Test cases for code conversion
        test_cases = [
            {
                "name": "Python to Java",
                "prompt": """<|im_start|>system
You are an expert code converter. Convert the given Python code to Java. Output only the Java code.<|im_end|>
<|im_start|>user
Convert this Python code to Java:

def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def main():
    result = calculate_factorial(5)
    print(f"Factorial of 5 is {result}")

if __name__ == "__main__":
    main()
<|im_end|>
<|im_start|>assistant
""",
                "max_tokens": 300
            },
            {
                "name": "Python to TypeScript",
                "prompt": """<|im_start|>system
You are an expert code converter. Convert the given Python code to TypeScript with proper type annotations.<|im_end|>
<|im_start|>user
Convert this Python code to TypeScript:

class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        return f"Hello, I'm {self.name} and I'm {self.age} years old"

def create_users(names: list, ages: list) -> list:
    return [User(n, a) for n, a in zip(names, ages)]
<|im_end|>
<|im_start|>assistant
""",
                "max_tokens": 350
            },
            {
                "name": "JavaScript to Python",
                "prompt": """<|im_start|>system
You are an expert code converter. Convert the given JavaScript code to Python.<|im_end|>
<|im_start|>user
Convert this JavaScript code to Python:

async function fetchUserData(userId) {
    try {
        const response = await fetch(`https://api.example.com/users/${userId}`);
        if (!response.ok) {
            throw new Error('User not found');
        }
        const data = await response.json();
        return {
            id: data.id,
            name: data.name,
            email: data.email
        };
    } catch (error) {
        console.error('Error fetching user:', error.message);
        return null;
    }
}
<|im_end|>
<|im_start|>assistant
""",
                "max_tokens": 300
            },
            {
                "name": "Code Generation - API Endpoint",
                "prompt": """<|im_start|>system
You are an expert Python developer. Generate clean, production-ready code.<|im_end|>
<|im_start|>user
Write a FastAPI endpoint that:
1. Accepts a POST request at /users
2. Takes JSON body with name (str) and email (str)
3. Validates email format
4. Returns the created user with an auto-generated ID
<|im_end|>
<|im_start|>assistant
""",
                "max_tokens": 350
            },
            {
                "name": "Code Generation - Unit Test",
                "prompt": """<|im_start|>system
You are an expert Python developer. Generate comprehensive unit tests.<|im_end|>
<|im_start|>user
Write pytest unit tests for this function:

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
<|im_end|>
<|im_start|>assistant
""",
                "max_tokens": 300
            }
        ]

        print("=" * 70)
        print("Running Code Conversion Tests")
        print("=" * 70)
        print()

        results = []

        for i, test in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test['name']}")
            print("-" * 50)

            config = GenerationConfig(
                max_tokens=test["max_tokens"],
                temperature=0.3,  # Lower temperature for more deterministic code
                top_p=0.9
            )

            start = time.time()
            result = runner.generate(test["prompt"], config)
            gen_time = time.time() - start

            # Clean up the output
            output = result.text.strip()
            if "<|im_end|>" in output:
                output = output.split("<|im_end|>")[0].strip()

            print(f"Generated Code:")
            print("-" * 30)
            # Show first 500 chars or full output if shorter
            display_output = output[:500] + "..." if len(output) > 500 else output
            print(display_output)
            print("-" * 30)
            print(f"Time: {gen_time:.2f}s | Tokens: {result.completion_tokens} | Speed: {result.tokens_per_second:.1f} tok/s")
            print()

            results.append({
                "name": test["name"],
                "time": gen_time,
                "tokens": result.completion_tokens,
                "speed": result.tokens_per_second
            })

        # Summary
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        print()
        print(f"{'Test Name':<35} {'Time (s)':<12} {'Tokens':<10} {'Speed (tok/s)':<15}")
        print("-" * 70)

        total_time = 0
        total_tokens = 0

        for r in results:
            print(f"{r['name']:<35} {r['time']:<12.2f} {r['tokens']:<10} {r['speed']:<15.1f}")
            total_time += r['time']
            total_tokens += r['tokens']

        print("-" * 70)
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        print(f"{'TOTAL':<35} {total_time:<12.2f} {total_tokens:<10} {avg_speed:<15.1f}")
        print()

        print("=" * 70)
        print("All code conversion tests completed successfully!")
        print("=" * 70)

        # Cleanup
        runner.shutdown()

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print("Install required packages: pip install transformers torch accelerate")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
