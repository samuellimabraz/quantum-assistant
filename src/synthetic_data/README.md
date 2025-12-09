# Synthetic Data Module

Generate high-quality multimodal training datasets for quantum computing VLM/LLM fine-tuning.

## Quick Start

```bash
# Install dependencies
pip install -e ".[synthetic]"

# Set environment variables for model endpoints
export VISION_MODEL_BASE_URL="https://api.openai.com/v1"
export VISION_MODEL_API_KEY="sk-..."
export VISION_MODEL_NAME="gpt-4o"
# ... (see config.yaml for all required variables)

# Run complete pipeline
synthetic-data pipeline --config yaml/config.yaml

# Or run individual stages
synthetic-data parse --config yaml/config.yaml
synthetic-data transcribe --config yaml/config.yaml
synthetic-data generate --config yaml/config.yaml
```

## Module Structure

```
synthetic_data/
├── cli/                    # Command-line interface
│   ├── main.py            # CLI entry point
│   ├── commands.py        # All CLI commands
│   └── generation_commands.py  # Generation stage commands
├── parsers/               # Document parsing
│   ├── base.py           # Base classes (Document, ImageReference)
│   ├── jupyter.py        # Jupyter notebook parser
│   ├── mdx.py            # MDX/Markdown parser
│   └── pdf.py            # PDF parser
├── extractors/            # Content extraction
│   ├── ingestion.py      # Document ingestion
│   ├── transcriber.py    # VLM image transcription
│   └── chunker.py        # Semantic chunking with context prioritization
├── generators/            # Sample generation
│   ├── allocation.py     # Diversity-aware allocation
│   ├── prompts.py        # Prompt templates and utilities
│   ├── types.py          # Type definitions (InputCandidate, Sample)
│   ├── sessions.py       # Answer generation with validation
│   └── stages/           # Generation stages
│       ├── plan.py       # Question/test generation with refinement
│       ├── filter_candidates.py  # Candidate quality filtering
│       ├── answer.py     # Answer generation with test loop
│       ├── curate.py     # Quality curation
│       └── classify.py   # Category classification
├── dataset/              # Dataset management
│   ├── builder.py        # Train/val/test splitting
│   ├── exporter.py       # HuggingFace export
│   ├── analyzer.py       # Statistics computation
│   └── plotter.py        # Visualization generation
├── utils/                # Utilities
│   ├── quality.py        # Content quality filtering
│   ├── image_filter.py   # Image quality filtering
│   ├── checkpoint.py     # Checkpoint management
│   ├── tracer.py         # Generation tracing
│   └── deduplication.py  # Sample deduplication
├── tools/                # Analysis tools
│   ├── pipeline_analyzer.py
│   └── pipeline_plotter.py
├── models/               # Model client wrappers
└── config/               # Configuration schema
```

## CLI Commands

### Pipeline Commands

| Command | Description |
|---------|-------------|
| `pipeline` | Run complete pipeline |
| `parse` | Parse source documents |
| `transcribe` | Transcribe images with VLM |
| `filter-images` | Filter low-quality images |
| `chunk` | Split into semantic chunks |
| `filter-chunks` | Filter low-quality content |
| `generate` | Generate samples |
| `build` | Create train/val/test splits |
| `export` | Export to HuggingFace format |
| `analyze` | Generate statistics and plots |

### Inspection Commands

| Command | Description |
|---------|-------------|
| `inspect` | Inspect pipeline artifacts |
| `inspect-traces` | View generation traces |
| `info` | Show configuration info |
| `validate-config` | Validate configuration |
| `cache-info` | Show cache status |
| `cache-clear` | Clear cache |

### Usage Examples

```bash
# Complete pipeline with analysis
synthetic-data pipeline --config yaml/config.yaml --hub-id user/dataset

# Generate with tracing disabled
synthetic-data generate --config yaml/config.yaml --no-trace

# Inspect chunks
synthetic-data inspect --config yaml/config.yaml --stage chunks --count 5

# View failed generation traces
synthetic-data inspect-traces --config yaml/config.yaml --failed

# Export with analysis to HuggingFace
synthetic-data export --config yaml/config.yaml --hub-id user/dataset --analyze

# Standalone analysis
synthetic-data analyze --config yaml/config.yaml --source splits
```

## Configuration

All settings are centralized in `yaml/config.yaml`:

```yaml
# Source documents
sources:
  - path: /path/to/data
    type: directory
    include_patterns: ["*.ipynb", "*.mdx"]

# Model endpoints
models:
  endpoints:
    - name: vision-model
      base_url: ${VISION_MODEL_BASE_URL}
      api_key: ${VISION_MODEL_API_KEY}
      model_name: ${VISION_MODEL_NAME}
    # ... question-model, answer-model, curate-model

# Generation settings
generation:
  target_samples: 8000
  over_allocation_factor: 1.8
  diversity_weight: 0.4
  
  type_allocations:
    qa:
      ratio: 0.30
      multimodal_ratio: 0.70
    code_generation:
      ratio: 0.35
      multimodal_ratio: 0.30
    function_completion:
      ratio: 0.35
      multimodal_ratio: 0.30

# Dataset output
dataset:
  name: quantum-multimodal
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

### Environment Variables

Required environment variables for model endpoints:

```bash
# Vision model (image transcription)
VISION_MODEL_BASE_URL
VISION_MODEL_API_KEY
VISION_MODEL_NAME

# Question generation model
QUESTION_MODEL_BASE_URL
QUESTION_MODEL_API_KEY
QUESTION_MODEL_NAME

# Answer generation model
ANSWER_MODEL_BASE_URL
ANSWER_MODEL_API_KEY
ANSWER_MODEL_NAME

# Curation/classification model
CURATE_MODEL_BASE_URL
CURATE_MODEL_API_KEY
CURATE_MODEL_NAME
```

## Output Dataset Schema

```python
{
    "question": str,       # Input prompt/question
    "answer": str,         # Reference solution/answer
    "category": str,       # Knowledge category
    "type": str,           # function_completion, code_generation, qa
    "test_code": str,      # Unit test (code types only)
    "entry_point": str,    # Function name (code types only)
    "image": PIL.Image,    # Associated image (multimodal only)
    "source": str,         # Source document path
}
```

### Question Types

| Type | Question Format | Answer Format | Test |
|------|-----------------|---------------|------|
| `function_completion` | Function stub with `pass` | Body code only | ✓ |
| `code_generation` | Natural language task | Full code with imports | ✓ |
| `qa` | Theory question | Explanation | ✗ |

### Plan Stage (6a) Features

The planning stage uses a unified session-based approach per task:

1. **Question Generation** - Type-specific prompt with prioritized context
2. **Question Refinement** - Ensures context grounding, avoids over-description
3. **Test Generation** - Based on context patterns, not invented logic
4. **Test Correction Loop** - Up to 3 correction attempts with error feedback

**Context Prioritization for Multimodal:**
- `[PRIORITY - Code That Generated Target Image]` appears first
- Image descriptions are secondary, truncated when code context exists
- Prevents layer-by-layer image enumeration in questions/tests

### Categories

7 quantum computing categories:
- `circuits_and_gates`
- `algorithms_and_applications`
- `transpilation_and_compilation`
- `primitives_and_execution`
- `noise_and_error_mitigation`
- `quantum_info_and_operators`
- `hardware_and_providers`

## Generated Visualizations

Analysis produces professional plots using Qiskit brand colors:

| Plot | Description |
|------|-------------|
| `overview_dashboard.png` | Multi-panel dataset summary |
| `split_distribution.png` | Samples per split |
| `type_distribution.png` | Question type distribution |
| `category_distribution.png` | Category distribution |
| `modality_distribution.png` | Multimodal vs text-only |
| `multimodal_breakdown.png` | Multimodal by dimensions |
| `source_analysis.png` | Source file statistics |
| `chunk_distribution.png` | Chunk size distribution |

## Checkpoint System

All stages support checkpoint-based resumption:

```bash
# Resume from checkpoint automatically
synthetic-data generate --config yaml/config.yaml

# Force restart (no cache)
synthetic-data generate --config yaml/config.yaml --no-cache
```

Checkpoints track:
- Processed item IDs (not just counts)
- Partial results for atomic resumption
- Substage progress in generation

## Programmatic Usage

```python
from synthetic_data.config import load_config
from synthetic_data.extractors import DocumentIngestion, ContentChunker
from synthetic_data.generators import GenerationPipeline
from synthetic_data.models import ModelRegistry

# Load configuration
config = load_config("yaml/config.yaml")

# Initialize components
registry = ModelRegistry(config.models)
ingestion = DocumentIngestion(images_output_dir=Path("outputs/images"))

# Parse documents
documents = []
for source in config.sources:
    documents.extend(ingestion.ingest_source(source))

# Chunk documents
chunker = ContentChunker(
    max_length=config.generation.max_context_length,
    min_length=config.generation.min_chunk_length,
)
chunks = []
for doc in documents:
    chunks.extend(chunker.chunk_document(doc))

# Generate samples
pipeline = GenerationPipeline(config, registry)
samples = pipeline.generate_samples(chunks)

# Export
from synthetic_data.dataset import DatasetBuilder, HuggingFaceExporter

builder = DatasetBuilder(config.dataset)
train, val, test = builder.stratified_build(samples)

exporter = HuggingFaceExporter(config.dataset)
dataset_dict = exporter.export(train, val, test)
dataset_dict.push_to_hub("username/dataset")
```

## Performance Tuning

### Concurrency Settings

```yaml
generation:
  llm_concurrency: 16   # Parallel LLM requests
  vlm_concurrency: 16   # Parallel VLM requests
```

**Recommendations:**
- OpenAI API: 10-20 concurrent requests
- Local inference (vLLM): 20-50 concurrent requests
- High-limit cloud: 50+ concurrent requests

### Allocation Tuning

```yaml
generation:
  over_allocation_factor: 1.8  # Generate 80% more candidates
  diversity_weight: 0.4        # Balance quality vs diversity
  max_generation_attempts: 3   # Retry rounds
```

Higher `over_allocation_factor` reduces retry attempts but increases cost.
Higher `diversity_weight` improves chunk/image coverage but may reduce quality.

## Quality Control

### Prompt Improvements for Grounded Generation

The prompts are designed to ensure generated content is grounded in context:

**Test Generation:**
- Tests must be based on context patterns, not copied from examples
- Simple structural checks preferred over complex operator comparisons
- Explicit instruction to avoid inventing `param_map` or other patterns not in context

**Question Refinement:**
- Catches layer-by-layer image descriptions
- Ensures multimodal questions reference images naturally
- Validates context grounding and Qiskit 2.0 compliance

**Answer Generation:**
- Forbidden patterns: MethodType, monkey-patching, overriding `__init__`
- Must use context code patterns directly
- If context uses loops, answer should use loops (not enumerate each element)

## Debugging

### Trace Inspection

```bash
# View generation traces
synthetic-data inspect-traces --config yaml/config.yaml

# Filter by stage
synthetic-data inspect-traces --config yaml/config.yaml --stage answer_generation

# View only failures
synthetic-data inspect-traces --config yaml/config.yaml --failed --count 10

# Full content (no truncation)
synthetic-data inspect-traces --config yaml/config.yaml --index 5 --full
```

### Debug Files

Generated in `outputs/generated/`:
- `rejected_samples.jsonl` - Quality curation rejections
- `code_verification_failures.jsonl` - Test failures with error history
- `traces/conversations.jsonl` - Full generation traces

## Documentation

For detailed architecture and implementation:

- **Architecture:** See `docs/synthetic.md`
- **Allocation Algorithm:** See `generators/ALLOCATION.md`

## License

Apache 2.0
