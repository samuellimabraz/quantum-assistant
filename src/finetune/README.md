# Fine-tuning Data Preparation

Prepare quantum computing datasets for VLM fine-tuning with ms-swift framework.

## Features

- **Image Processing**: Resize images maintaining aspect ratio
- **ms-swift Format**: Convert to JSONL with proper message structure
- **Multimodal Support**: Handle text-only and image+text samples
- **System Prompts**: Configurable quantum computing expert prompts
- **Split Support**: Process train/validation/test splits

## Quick Start

```bash
# Prepare dataset with default settings
finetune prepare --dataset-path outputs/final --output-dir outputs/finetune

# Use configuration file
finetune prepare --config src/finetune/yaml/finetune_config.yaml

# Generate default config
finetune init-config --output my_config.yaml
```

## Output Format

The prepared data follows ms-swift's expected format:

```json
{
    "messages": [
        {"role": "system", "content": "You are a quantum computing expert..."},
        {"role": "user", "content": "<image>\nComplete the function..."},
        {"role": "assistant", "content": "return QuantumCircuit(n_qubits)"}
    ],
    "images": ["images/circuit_abc123.jpg"]
}
```

## Configuration

Key settings in `finetune_config.yaml`:

```yaml
# Image processing
image:
  max_size: 640  # Maximum dimension
  quality: 95    # JPEG quality
  format: JPEG   # Output format
  preserve_aspect_ratio: true

# ms-swift format
swift:
  image_placeholder: "<image>"
  include_system_prompt: true
  system_prompt: "You are a quantum computing expert..."

# Processing
max_samples: null  # null for all, int for testing
question_types: null  # Filter by type
```

## Output Structure

```
outputs/finetune/
├── train.jsonl       # Training data
├── validation.jsonl  # Validation data
├── test.jsonl        # Test data
├── images/           # Processed images
│   ├── circuit_abc123.jpg
│   └── ...
└── summary.json      # Statistics
```

## CLI Options

```bash
finetune prepare [OPTIONS]

Options:
  -c, --config PATH          YAML configuration file
  -d, --dataset-path PATH    HuggingFace dataset directory [default: outputs/final]
  -o, --output-dir PATH      Output directory [default: outputs/finetune]
  --max-size INTEGER         Maximum image dimension [default: 640]
  --image-format [JPEG|PNG]  Output image format [default: JPEG]
  --max-samples INTEGER      Maximum samples per split (for testing)
  --question-types TEXT      Comma-separated question types
  --no-system-prompt         Exclude system prompt from messages
```

## Python API

```python
from finetune import DatasetPreparer, FinetuneConfig

# Load configuration
config = FinetuneConfig.from_yaml("finetune_config.yaml")

# Or create programmatically
config = FinetuneConfig(
    dataset_path="outputs/final",
    output_dir="outputs/finetune",
)
config.image.max_size = 512
config.swift.include_system_prompt = True

# Prepare dataset
preparer = DatasetPreparer(config)
result = preparer.prepare()

print(f"Processed {result.statistics['train']['total']} training samples")
```

## Integration with ms-swift

After preparing the dataset, use it with ms-swift:

```bash
# Train with prepared data
swift sft \
    --model_type qwen2_vl-7b-instruct \
    --dataset outputs/finetune/train.jsonl \
    --val_dataset outputs/finetune/validation.jsonl \
    --output_dir outputs/models/qwen2-vl-quantum
```

Or use YAML configuration:

```yaml
# swift_train.yaml
model_type: qwen2_vl-7b-instruct
dataset: outputs/finetune/train.jsonl
val_dataset: outputs/finetune/validation.jsonl
output_dir: outputs/models/qwen2-vl-quantum
```

## Sample Types

The preparer handles all three question types:

| Type | Description | Has Image |
|------|-------------|-----------|
| `function_completion` | Complete function body | Optional |
| `code_generation` | Generate full code | Optional |
| `qa` | Theory/concepts | Optional |

All types support multimodal (with image) and text-only variants.

