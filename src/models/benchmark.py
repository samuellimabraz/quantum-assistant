"""Model benchmarking utilities."""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

from models.client import LLMClient, Message, VLMClient


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    test_type: str  # "llm" or "vlm"
    batch_size: int
    concurrency: int
    num_requests: int

    total_time: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    std_latency: float
    throughput: float  # requests/second
    tokens_per_second: float | None = None

    errors: int = 0
    success_rate: float = 1.0

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Benchmark: {self.model_name} ({self.test_type.upper()})\n"
            f"{'='*60}\n"
            f"Configuration:\n"
            f"  Batch size: {self.batch_size}\n"
            f"  Concurrency: {self.concurrency}\n"
            f"  Total requests: {self.num_requests}\n"
            f"\nResults:\n"
            f"  Total time: {self.total_time:.2f}s\n"
            f"  Throughput: {self.throughput:.2f} req/s\n"
            f"  Mean latency: {self.mean_latency:.3f}s\n"
            f"  Median latency: {self.median_latency:.3f}s\n"
            f"  P95 latency: {self.p95_latency:.3f}s\n"
            f"  P99 latency: {self.p99_latency:.3f}s\n"
            f"  Std latency: {self.std_latency:.3f}s\n"
            + (f"  Tokens/s: {self.tokens_per_second:.2f}\n" if self.tokens_per_second else "")
            + (
                f"  Errors: {self.errors} ({(1-self.success_rate)*100:.1f}%)\n"
                if self.errors > 0
                else ""
            )
            + f"{'='*60}\n"
        )


class ModelBenchmark:
    """Benchmark model performance with various configurations."""

    def __init__(self, client: LLMClient | VLMClient):
        """Initialize benchmark with a client."""
        self.client = client

    async def benchmark_llm_async(
        self,
        prompts: list[str],
        batch_size: int = 10,
        max_concurrent: int = 10,
        max_tokens: int = 512,
    ) -> BenchmarkResult:
        """Benchmark LLM performance."""
        if not isinstance(self.client, LLMClient):
            raise ValueError("Client must be LLMClient for LLM benchmarking")

        # Convert prompts to messages
        message_batches = [[Message(role="system", content="Reasoning: low. You are a helpful assistant."), Message(role="user", content=p)] for p in prompts]

        # Track latencies
        latencies = []
        errors = 0
        start_time = time.time()

        # Process in batches
        for i in range(0, len(message_batches), batch_size):
            batch = message_batches[i : i + batch_size]

            try:
                batch_start = time.time()
                await self.client.generate_batch_async(
                    batch, max_tokens=max_tokens, max_concurrent=max_concurrent
                )
                batch_time = time.time() - batch_start

                # Estimate per-request latency
                per_request = batch_time / len(batch)
                latencies.extend([per_request] * len(batch))

                # Small delay between batches to avoid overwhelming server
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                errors += len(batch)
                # Longer delay after error
                await asyncio.sleep(1.0)

        total_time = time.time() - start_time

        # Calculate statistics
        if not latencies:
            raise RuntimeError("All requests failed")

        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return BenchmarkResult(
            model_name=self.client.model_name,
            test_type="llm",
            batch_size=batch_size,
            concurrency=max_concurrent,
            num_requests=len(prompts),
            total_time=total_time,
            mean_latency=mean(latencies),
            median_latency=median(latencies),
            p95_latency=sorted_latencies[p95_idx],
            p99_latency=sorted_latencies[p99_idx],
            std_latency=stdev(latencies) if len(latencies) > 1 else 0.0,
            throughput=len(prompts) / total_time,
            errors=errors,
            success_rate=(len(prompts) - errors) / len(prompts),
        )

    async def benchmark_vlm_async(
        self,
        prompts: list[tuple[str, Path]],
        batch_size: int = 4,
        max_concurrent: int = 4,
        max_tokens: int = 512,
    ) -> BenchmarkResult:
        """Benchmark VLM performance."""
        if not isinstance(self.client, VLMClient):
            raise ValueError("Client must be VLMClient for VLM benchmarking")

        latencies = []
        errors = 0
        start_time = time.time()

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            try:
                batch_start = time.time()
                await self.client.generate_batch_with_images_async(
                    batch, max_tokens=max_tokens, max_concurrent=max_concurrent
                )
                batch_time = time.time() - batch_start

                per_request = batch_time / len(batch)
                latencies.extend([per_request] * len(batch))

                # Delay between batches (longer for VLM)
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                errors += len(batch)
                # Longer delay after error
                await asyncio.sleep(2.0)

        total_time = time.time() - start_time

        if not latencies:
            raise RuntimeError("All requests failed")

        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return BenchmarkResult(
            model_name=self.client.model_name,
            test_type="vlm",
            batch_size=batch_size,
            concurrency=max_concurrent,
            num_requests=len(prompts),
            total_time=total_time,
            mean_latency=mean(latencies),
            median_latency=median(latencies),
            p95_latency=sorted_latencies[p95_idx],
            p99_latency=sorted_latencies[p99_idx],
            std_latency=stdev(latencies) if len(latencies) > 1 else 0.0,
            throughput=len(prompts) / total_time,
            errors=errors,
            success_rate=(len(prompts) - errors) / len(prompts),
        )

    def benchmark_llm(
        self,
        prompts: list[str],
        batch_size: int = 10,
        max_concurrent: int = 10,
        max_tokens: int = 512,
    ) -> BenchmarkResult:
        """Benchmark LLM (sync wrapper)."""
        return asyncio.run(
            self.benchmark_llm_async(prompts, batch_size, max_concurrent, max_tokens)
        )

    def benchmark_vlm(
        self,
        prompts: list[tuple[str, Path]],
        batch_size: int = 4,
        max_concurrent: int = 4,
        max_tokens: int = 512,
    ) -> BenchmarkResult:
        """Benchmark VLM (sync wrapper)."""
        return asyncio.run(
            self.benchmark_vlm_async(prompts, batch_size, max_concurrent, max_tokens)
        )

    async def grid_search_llm_async(
        self,
        prompts: list[str],
        batch_sizes: list[int] = [5, 10, 20, 50],
        concurrencies: list[int] = [5, 10, 20, 50],
        max_tokens: int = 512,
    ) -> list[BenchmarkResult]:
        """Test multiple configurations to find optimal settings."""
        results = []

        for batch_size in batch_sizes:
            for concurrency in concurrencies:
                print(f"\nTesting batch_size={batch_size}, concurrency={concurrency}")
                try:
                    result = await self.benchmark_llm_async(
                        prompts, batch_size, concurrency, max_tokens
                    )
                    results.append(result)
                    print(result)
                except Exception as e:
                    print(f"Failed: {e}")

        results.sort(key=lambda r: r.throughput, reverse=True)
        return results

    def grid_search_llm(
        self,
        prompts: list[str],
        batch_sizes: list[int] = [5, 10, 20, 50],
        concurrencies: list[int] = [5, 10, 20, 50],
        max_tokens: int = 512,
    ) -> list[BenchmarkResult]:
        """Test multiple configurations (sync wrapper)."""
        return asyncio.run(
            self.grid_search_llm_async(prompts, batch_sizes, concurrencies, max_tokens)
        )
