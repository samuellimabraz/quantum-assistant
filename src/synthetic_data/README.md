# Synthetic Data Module

Generate high-quality multimodal datasets for quantum computing VLM fine-tuning from open-source documentation.

## Features

- **Multimodal Support**: Extract and process images (circuits, charts, Bloch spheres) alongside code
- **Executable Verification**: Validate generated code samples pass unit tests
- **Quality Control**: Multi-stage filtering with VLM and LLM-based quality checks
- **Diversity Allocation**: Balanced coverage of source documents and images
- **Checkpoint System**: Resume interrupted generation with automatic state saving
- **Three Task Types**: Function completion, code generation, and conceptual Q&A

## Installation

```bash
# From project root
pip install -e ".[synthetic]"

# Or with uv
uv sync
```

## Quick Start

### 1. Set Environment Variables

Create `.env` file in project root:

```bash
# Vision model (image transcription and quality filtering)
VISION_MODEL_BASE_URL=https://api.openai.com/v1
VISION_MODEL_API_KEY=sk-...
VISION_MODEL_NAME=gpt-4o

# Question generation model
QUESTION_MODEL_BASE_URL=http://localhost:8000/v1
QUESTION_MODEL_API_KEY=your-key
QUESTION_MODEL_NAME=gpt-oss-120b

# Answer generation model
ANSWER_MODEL_BASE_URL=http://localhost:8000/v1
ANSWER_MODEL_API_KEY=your-key
ANSWER_MODEL_NAME=gpt-oss-120b

# Curation/classification model
CURATE_MODEL_BASE_URL=http://localhost:8000/v1
CURATE_MODEL_API_KEY=your-key
CURATE_MODEL_NAME=gpt-oss-120b
```

### 2. Run Complete Pipeline

```bash
synthetic-data pipeline --config src/synthetic_data/yaml/config.yaml
```

This executes all stages sequentially:
1. Parse source documents
2. Transcribe images with VLM
3. Filter low-quality images
4. Chunk into semantic units
5. Filter low-quality chunks
6. Generate question-answer pairs
7. Build train/val/test splits
8. Export to HuggingFace format
9. Generate statistics and plots

### 3. Run Individual Stages

For debugging or customization, run stages separately:

```bash
# Document processing
synthetic-data parse --config config.yaml
synthetic-data transcribe --config config.yaml
synthetic-data filter-images --config config.yaml
synthetic-data chunk --config config.yaml
synthetic-data filter-chunks --config config.yaml

# Sample generation (can be run as substages)
synthetic-data plan --config config.yaml              # Plan inputs
synthetic-data filter-candidates --config config.yaml # Filter inputs
synthetic-data answer --config config.yaml            # Generate answers
synthetic-data curate --config config.yaml            # Quality curation
synthetic-data classify --config config.yaml          # Categorize

# Or run all generation stages together
synthetic-data generate --config config.yaml

# Dataset assembly
synthetic-data build --config config.yaml
synthetic-data export --config config.yaml --hub-id username/dataset

# Analysis
synthetic-data analyze --config config.yaml
```

## Output Dataset

### Schema

Each generated sample contains:

```python
{
    "question": str,       # Task prompt or question
    "answer": str,         # Expected code or explanation
    "category": str,       # One of 7 quantum categories
    "type": str,           # function_completion, code_generation, qa
    "test_code": str,      # Unit test (code types only)
    "entry_point": str,    # Function name (code types only)
    "image": PIL.Image,    # Associated image (multimodal only)
    "source": str,         # Source document path
}
```

### Question Types

| Type | Description | Answer Format | Test Coverage |
|------|-------------|---------------|---------------|
| `function_completion` | Function stub with signature + docstring | Body code only | 100% |
| `code_generation` | Natural language task description | Full code with imports | 100% |
| `qa` | Conceptual/theory question | Textual explanation | N/A |

### Categories

7 quantum computing categories covering:

1. `circuits_and_gates` - Circuit construction, gates
2. `quantum_info_and_operators` - States, operators, fidelity
3. `algorithms_and_applications` - VQE, QAOA, Grover, QML
4. `hardware_and_providers` - Backend properties, ISA circuits
5. `transpilation_and_compilation` - Pass managers, optimization
6. `primitives_and_execution` - Sampler, Estimator, sessions
7. `noise_and_error_mitigation` - ZNE, TREX, error modeling

### Output Structure

```
outputs/
├── parsed/            # Parsed documents with image references
├── transcribed/       # Documents with image descriptions
├── filtered_images/   # Quality-filtered images
├── chunks/            # Semantic content chunks
├── filtered/          # Quality-filtered chunks
├── planned/           # Generated question/test candidates
├── filtered_candidates/ # Quality-filtered candidates
├── answered/          # Samples with validated answers
├── curated/           # Final curated samples
├── generated/         # Classified samples (JSONL)
├── splits/            # Train/val/test splits
├── analysis/          # Statistics and plots
└── final/             # HuggingFace DatasetDict
    ├── train/
    ├── validation/
    └── test/
```

## Configuration

Edit `src/synthetic_data/yaml/config.yaml`:

### Key Parameters

```yaml
# Target distribution
generation:
  target_samples: 8000
  over_allocation_factor: 1.8     # Generate extra to compensate for filtering
  diversity_weight: 0.4           # Balance quality vs coverage
  
  # Task type ratios
  type_allocations:
    qa:
      ratio: 0.30                 # 30% Q&A
      multimodal_ratio: 0.70      # 70% of Q&A have images
    code_generation:
      ratio: 0.35
      multimodal_ratio: 0.30
    function_completion:
      ratio: 0.35
      multimodal_ratio: 0.30

  # Concurrency for parallel processing
  llm_concurrency: 16             # Parallel LLM requests
  vlm_concurrency: 16             # Parallel VLM requests

  # Code verification
  enable_code_verification: true
  code_verification_max_iterations: 7  # Correction attempts
  code_verification_timeout: 60        # Execution timeout (seconds)

# Dataset splits
dataset:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

See full configuration template in `src/synthetic_data/yaml/config.yaml`.

## Concurrency Tuning

Adjust based on your LLM/VLM endpoints:

| Endpoint Type | `llm_concurrency` | `vlm_concurrency` |
|---------------|-------------------|-------------------|
| OpenAI API | 10-20 | 10-20 |
| Local vLLM (high-end GPU) | 50-100 | 20-40 |
| Cloud with high limits | 50+ | 50+ |

Higher concurrency speeds up generation but may hit rate limits. The system includes automatic retry with exponential backoff.

## Inspection and Debugging

### View Generated Artifacts

```bash
# Inspect chunks
synthetic-data inspect --config config.yaml --stage chunks --count 5

# Inspect samples
synthetic-data inspect --config config.yaml --stage samples --random

# View generation traces (conversation history)
synthetic-data inspect-traces --config config.yaml --stage answer_generation

# View failed generations
synthetic-data inspect-traces --config config.yaml --failed --count 10
```

### Debug Files

Generated in `outputs/generated/`:

- `rejected_samples.jsonl` - Quality curation rejections with reasons
- `code_verification_failures.jsonl` - Test failures with error traces
- `traces/conversations.jsonl` - Full LLM conversation logs

### Configuration Validation

```bash
# Check configuration
synthetic-data info --config config.yaml

# Validate schema
synthetic-data validate-config --config config.yaml

# Cache management
synthetic-data cache-info --config config.yaml
synthetic-data cache-clear --config config.yaml --stage chunks
```

## Checkpoint System

All stages save progress automatically. If interrupted, rerun the same command to resume:

```bash
# This will resume from where it left off
synthetic-data generate --config config.yaml
```

Progress is tracked by processed item IDs (not just counts), ensuring exact resumption without duplicates.

To force restart:

```bash
synthetic-data generate --config config.yaml --no-cache
```

## Example: Custom Dataset

### 1. Prepare Source Documents

Place Jupyter notebooks (`.ipynb`), MDX files (`.mdx`), or PDFs in `data/`:

```
data/
├── qiskit_docs/         # Official documentation
├── tutorials/           # Tutorial notebooks
└── papers/              # Scientific papers
```

### 2. Configure Sources

Edit `config.yaml`:

```yaml
sources:
  - path: data/qiskit_docs
    type: directory
    include_patterns: ["*.ipynb", "*.mdx"]
  - path: data/tutorials
    type: directory
    include_patterns: ["*.ipynb"]
  - path: data/papers
    type: directory
    include_patterns: ["*.pdf"]
```

### 3. Adjust Generation Parameters

```yaml
generation:
  target_samples: 5000    # Generate 5k samples
  
  type_allocations:
    qa:
      ratio: 0.40         # More Q&A
      multimodal_ratio: 0.80  # More images in Q&A
    code_generation:
      ratio: 0.30
      multimodal_ratio: 0.25
    function_completion:
      ratio: 0.30
      multimodal_ratio: 0.25
```

### 4. Run Pipeline

```bash
synthetic-data pipeline --config config.yaml
```

### 5. Upload to HuggingFace

```bash
# Export with automatic upload
synthetic-data export --config config.yaml --hub-id username/my-quantum-dataset

# Or push manually
from datasets import load_from_disk
dataset = load_from_disk("outputs/final")
dataset.push_to_hub("username/my-quantum-dataset")
```


## Common Issues

### Out of Memory

Reduce batch sizes or target samples:

```yaml
generation:
  target_samples: 4000    # Reduce from 8000
  llm_concurrency: 8      # Lower concurrency
```

### Rate Limits

Automatic retry handles temporary limits. For persistent issues:

```yaml
models:
  endpoints:
    - name: question-model
      max_retries: 10       # Increase from 5
      retry_delay: 2.0      # Longer backoff (seconds)
```

### Poor Quality Samples

Strengthen filtering:

```yaml
generation:
  enable_candidate_filtering: true
  enable_curate_filtering: true
  similarity_threshold: 0.90    # Stricter deduplication
```

## Documentation

- **Architecture and Algorithms**: See [docs/synthetic.md](../../docs/synthetic.md) for detailed pipeline design, allocation strategy, and implementation details
- **Model Serving**: See [docs/models.md](../../docs/models.md) for LLM/VLM client setup and benchmarking

## License

Apache 2.0
