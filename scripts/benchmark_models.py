#!/usr/bin/env python
"""Benchmark model performance for optimal configuration."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import LLMClient, ModelBenchmark, VLMClient


def benchmark_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    num_requests: int = 100,
):
    """Benchmark LLM with grid search."""
    print(f"\n{'='*70}")
    print(f"Benchmarking LLM: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Total requests: {num_requests}")
    print(f"{'='*70}\n")

    # Create test prompts
    prompts = [
        f"Explain quantum {concept} in simple terms."
        for concept in [
            "superposition",
            "entanglement",
            "measurement",
            "interference",
            "teleportation",
        ]
        * (num_requests // 5)
    ][:num_requests]

    client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        max_tokens=2048,
        max_retries=3,
        retry_delay=1.0,
    )

    benchmark = ModelBenchmark(client)

    # Grid search configurations
    batch_sizes = [5, 10, 20, 32]
    concurrencies = [10, 20, 50, 100]

    print("\n🔍 Running grid search to find optimal configuration...\n")
    results = asyncio.run(
        benchmark.grid_search_llm_async(prompts, batch_sizes, concurrencies, max_tokens=1024)
    )

    # Show top 3 configurations
    print("\n🏆 Top 3 Configurations by Throughput:\n")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result}")

    # Save results
    output_file = Path(f"benchmark_{model_name.replace('/', '_')}_llm.json")
    with open(output_file, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    client.close()
    return results[0]  # Return best config


def benchmark_vlm(
    base_url: str,
    api_key: str,
    model_name: str,
    test_images_dir: Path,
    num_requests: int = 50,
):
    """Benchmark VLM with transcription prompt."""
    print(f"\n{'='*70}")
    print(f"Benchmarking VLM: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Total requests: {num_requests}")
    print(f"{'='*70}\n")

    # Get test images
    test_images = (
        list(test_images_dir.glob("*.png"))
        + list(test_images_dir.glob("*.jpg"))
        + list(test_images_dir.glob("*.avif"))
        + list(test_images_dir.glob("*.svg"))
    )
    test_images = test_images[:num_requests]

    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return None

    print(f"Found {len(test_images)} test images")

    # Create prompts with transcription template
    transcription_prompt = """Provide a detailed, comprehensive description of this image related to quantum computing, mathematics, or physics.
Focus on:
- What quantum concepts, operations, or circuits are shown
- Key visual elements (gates, qubits, states, diagrams, formulas, graphs, etc.)
- Mathematical notation or formulas if present
- Any labels, legends, or annotations
- Charts or graphs description if present

Provide a complete technical description suitable for training data generation."""

    prompts = [(transcription_prompt, img) for img in test_images]

    client = VLMClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        max_tokens=4096,
        max_retries=3,
        retry_delay=2.0,  # Longer delay for VLM
    )

    benchmark = ModelBenchmark(client)

    # Test configurations (lower concurrency for VLM)
    configs = [(2, 2), (4, 2), (4, 4), (8, 4), (8, 8), (16, 8), (16, 16)]

    # Run all tests in single event loop
    async def run_all_tests():
        results = []
        for batch_size, concurrency in configs:
            print(f"\nTesting batch_size={batch_size}, concurrency={concurrency}")
            try:
                result = await benchmark.benchmark_vlm_async(
                    prompts=prompts,
                    batch_size=batch_size,
                    max_concurrent=concurrency,
                    max_tokens=1024,
                )
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Failed: {e}")
        return results

    results = asyncio.run(run_all_tests())

    if results:
        # Sort by throughput
        results.sort(key=lambda r: r.throughput, reverse=True)

        print("\n🏆 Top Configuration:\n")
        print(results[0])

        # Save results
        output_file = Path(f"benchmark_{model_name.replace('/', '_')}_vlm.json")
        with open(output_file, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2, default=str)

        print(f"\n✅ Results saved to {output_file}")
        client.close()
        return results[0]

    client.close()
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--model-type", choices=["llm", "vlm"], required=True)
    parser.add_argument("--base-url", required=True, help="Model API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of test requests")
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        help="Directory with test images (required for VLM)",
    )

    args = parser.parse_args()

    if args.model_type == "llm":
        best = benchmark_llm(args.base_url, args.api_key, args.model_name, args.num_requests)
        print(f"\n✨ Recommended configuration:")
        print(f"   batch_size: {best.batch_size}")
        print(f"   max_concurrent: {best.concurrency}")
        print(f"   Expected throughput: {best.throughput:.2f} req/s")

    elif args.model_type == "vlm":
        if not args.test_images_dir:
            print("❌ --test-images-dir required for VLM benchmarking")
            return

        best = benchmark_vlm(
            args.base_url, args.api_key, args.model_name, args.test_images_dir, args.num_requests
        )
        if best:
            print(f"\n✨ Recommended configuration:")
            print(f"   batch_size: {best.batch_size}")
            print(f"   max_concurrent: {best.concurrency}")
            print(f"   Expected throughput: {best.throughput:.2f} req/s")


if __name__ == "__main__":
    main()
