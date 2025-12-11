# Models Module

Model clients with async batch support and benchmarking.

## Features

- **Async batch processing** for high throughput
- **LLM and VLM clients** with OpenAI-compatible API
- **Built-in benchmarking** to find optimal configurations
- **Registry pattern** for managing multiple models
- **Used across all modules** (not just synthetic data)

## Quick Start

```python
from models import LLMClient, VLMClient, ModelBenchmark

# Create client
client = LLMClient(
    base_url="http://localhost:8000/v1",
    api_key="your-key",
    model_name="gpt-oss-20b"
)

# Single request
from models import Message
response = client.generate([Message(role="user", content="Hello")])

# Batch requests (async)
prompts = [[Message(role="user", content=f"Question {i}")] for i in range(100)]
responses = client.generate_batch(prompts, max_concurrent=20)
```

## Benchmarking

### CLI Benchmark

```bash
# Benchmark LLM
python scripts/benchmark_models.py \
    --model-type llm \
    --base-url http://localhost:8000/v1 \
    --model-name gpt-oss-20b \
    --num-requests 100

# Benchmark VLM  
python scripts/benchmark_models.py \
    --model-type vlm \
    --base-url http://localhost:8000/v1 \
    --model-name olmocr \
    --test-images-dir assets/tests \
    --num-requests 50
```

### Programmatic Benchmark

```python
from models import LLMClient, ModelBenchmark

client = LLMClient(base_url="...", model_name="...")
benchmark = ModelBenchmark(client)

# Test single configuration
prompts = ["Explain quantum superposition"] * 100
result = benchmark.benchmark_llm(
    prompts, 
    batch_size=10, 
    max_concurrent=20
)
print(result)  # Shows latency, throughput, etc.

# Grid search for optimal config
results = benchmark.grid_search_llm(
    prompts,
    batch_sizes=[5, 10, 20, 50],
    concurrencies=[10, 20, 50, 100]
)
best = results[0]  # Sorted by throughput
```

## Your vLLM Configuration

Based on your setup:

### gpt-oss-20b (A100 80GB)
- `max_num_seqs=128` 
- `max_batched_tokens=65536`
- **Recommended:** `batch_size=20-32`, `concurrency=50-100`

### Qwen2.5-Coder-14B-Qiskit (A100 80GB)
- Same config as above
- **Recommended:** `batch_size=20-32`, `concurrency=50-100`

### olmOCR-2-7B (RTX 5090 34GB)
- `max_num_seqs=4`
- **Recommended:** `batch_size=2-4`, `concurrency=2-4`
- VLM has lower concurrency due to image processing
- **Important:** VLM servers may have rate limiting - built-in retry logic handles this

## Handling Server Issues

The clients include automatic retry logic with exponential backoff:
- 3 retries by default
- Exponential backoff: 1s, 2s, 4s
- Delays between batches to avoid overwhelming servers
- Graceful error handling

If you see intermittent 404 errors, this is typically **server rate limiting** or **temporary unavailability**, not a code issue. The retry logic will handle these automatically.

## Architecture

```
models/
├── client.py        # LLMClient, VLMClient with async
├── registry.py      # ModelRegistry for managing clients
└── benchmark.py     # Benchmarking utilities

synthetic_data/models/
└── __init__.py      # Adapter for backward compatibility
```

## Async Batch API

### LLM Batch

```python
# Sync
responses = client.generate_batch(
    batch=message_lists,
    max_concurrent=20
)

# Async
import asyncio
responses = await client.generate_batch_async(
    batch=message_lists,
    max_concurrent=20
)
```

### VLM Batch

```python
# With images
prompts = [
    ("Describe this image", Path("img1.png")),
    ("Describe this image", Path("img2.png")),
]

responses = vlm_client.generate_batch_with_images(
    prompts,
    max_concurrent=4  # Lower for VLM
)
```

## Benchmark Metrics

- **Throughput:** Requests per second
- **Mean/Median Latency:** Average and median response time
- **P95/P99 Latency:** 95th and 99th percentile
- **Success Rate:** Percentage of successful requests
- **Tokens/s:** Token generation speed (if available)

## Integration with Synthetic Data

The synthetic data pipeline automatically uses these optimized clients:

```yaml
# config.yaml
generation:
  batch_size: 20          # From benchmark results
  max_concurrent: 50      # From benchmark results
```

Pipeline will use async batching automatically for better throughput.

