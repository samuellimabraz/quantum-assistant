# Synthetic Data Generation Pipeline

Clean, modular pipeline for generating synthetic datasets for fine-tuning LLMs and VLMs.

## Features

- Multi-format parsing (Jupyter, MDX, PDF)
- Multimodal support (text + images with VLM transcription)
- **Async batch processing** for high throughput
- LLM-based classification and quality control
- Intelligent caching for incremental processing
- Detailed progress tracking with time estimates
- Deduplication
- Difficulty stratification
- OpenAI-compatible API (vLLM, TGI, etc.)
- HuggingFace export

## Performance

- **Async batch generation** using `models/` module
- **Per-source caching** - interrupt-safe processing
- **Automatic retry logic** with exponential backoff
- **Progress bars** showing files, time, and throughput

Use `scripts/benchmark_models.py` to find optimal batch sizes for your hardware.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Test all pipeline steps
python tests/test_pipeline_steps.py src/synthetic_data/yaml/config_test.yaml

# 2. Parse documents (with caching)
synthetic-data parse --config src/synthetic_data/yaml/config.yaml

# 3. Generate dataset
synthetic-data generate --config src/synthetic_data/yaml/config.yaml

# Cache commands
synthetic-data cache-info --config src/synthetic_data/yaml/config.yaml
synthetic-data cache-clear --config src/synthetic_data/yaml/config.yaml
synthetic-data parse --config config.yaml --no-cache  # Skip cache
```

## Pipeline

```
Parse → Chunk → Classify → Generate → Deduplicate → Split → Export
  ↓       ↓        ↓          ↓           ↓          ↓       ↓
parsed/ chunks  categories samples  unique    train/  final/
                          (LLM)    (w/diff)  val/test
```

## Configuration

Main config: `src/synthetic_data/yaml/config.yaml`

Key settings:
- 14 quantum computing categories
- 4 question types (QA, Code, Caption, Summary)
- 3 difficulty levels (Easy, Medium, Hard)
- Balanced distribution across all dimensions
- VLM endpoint configured

See `docs/synthetic.md` for complete schema and diagrams.

## Optimization

### Benchmark Your Models

```bash
python src/models/benchmark.py \
  --base-url http://localhost:8000/v1 \
  --model-name your-model \
  --test-type llm
```

### Configure Based on Benchmarks

Update `config.yaml` with optimal settings:

```yaml
generation:
  llm_batch_size: 10        # From benchmark results
  llm_concurrency: 20       # From benchmark results
  vlm_batch_size: 16
  vlm_concurrency: 16
```

### Test with Limited Data

```yaml
sources:
  - path: /path/to/data
    max_files: 5  # Limit for testing
```

Or use test config:

```bash
python tests/test_pipeline_steps.py src/synthetic_data/yaml/config_test.yaml
```

See `docs/synthetic.md` for complete optimization guide.

## Testing

### Quick Test (Recommended)

Test all pipeline steps with limited data (5 files, 20 samples):

```bash
python tests/test_pipeline_steps.py src/synthetic_data/yaml/config_test.yaml
```

**Tests all pipeline steps:**
1. Configuration loading and validation
2. Model connections (LLM + VLM with test requests)
3. Document parsing (respects `max_files`)
4. Image resolution + transcription (async batching)
5. Content chunking
6. Quality filtering (content + images)
7. Category classification (LLM-based)
8. Sample generation (async batching)
9. Cache functionality (save/load/clear)
