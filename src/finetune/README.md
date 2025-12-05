# Fine-tuning Data Preparation

Prepare quantum computing datasets for VLM fine-tuning with ms-swift framework.

## Features

- **Multiple Sources**: Load from HuggingFace Hub, local parquet, or Arrow format
- **Image Processing**: Resize images maintaining aspect ratio
- **ms-swift Format**: Convert to JSONL with proper message structure
- **Multimodal Support**: Handle text-only and image+text samples
- **System Prompts**: Configurable quantum computing expert prompts

## Quick Start

```bash
# From HuggingFace Hub (recommended for Colab/remote)
finetune prepare --hub-id samuellimabraz/quantum-test --output-dir ./swift_data

# From local parquet (downloaded from Hub)
finetune prepare --dataset-path /content/quantum-test --output-dir ./swift_data

# From local Arrow format (from save_to_disk)
finetune prepare --dataset-path outputs/final --output-dir outputs/finetune
```

## Dataset Sources

The preparer supports three dataset sources:

| Source | Option | Example |
|--------|--------|---------|
| **HuggingFace Hub** | `--hub-id` | `samuellimabraz/quantum-test` |
| **Local Parquet** | `--dataset-path` | `/content/quantum-test` (with `data/*.parquet`) |
| **Local Arrow** | `--dataset-path` | `outputs/final` (from `save_to_disk`) |

### HuggingFace Hub (Recommended)

Best for Google Colab or remote training environments:

```bash
finetune prepare --hub-id samuellimabraz/quantum-test --output-dir ./swift_data
```

### Local Parquet

For datasets downloaded from Hub with structure:

```
/content/quantum-test/
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
└── README.md
```

```bash
finetune prepare --dataset-path /content/quantum-test --output-dir ./swift_data
```

### Local Arrow

For datasets created with `save_to_disk`:

```bash
finetune prepare --dataset-path outputs/final --output-dir outputs/finetune
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
  -h, --hub-id TEXT          HuggingFace Hub dataset ID
  -d, --dataset-path PATH    Local path to dataset directory
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

# From HuggingFace Hub
config = FinetuneConfig(
    hub_id="samuellimabraz/quantum-test",
    output_dir="./swift_data",
)

# Or from local path
config = FinetuneConfig(
    dataset_path="/content/quantum-test",
    output_dir="./swift_data",
)

# Configure options
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
    --dataset ./swift_data/train.jsonl \
    --val_dataset ./swift_data/validation.jsonl \
    --output_dir ./models/qwen2-vl-quantum
```

## Google Colab Example

```python
# Install dependencies
!pip install -q datasets pillow tqdm pyyaml pydantic typer rich

# Clone repo (or install package)
!git clone https://github.com/user/quantum-assistant.git
%cd quantum-assistant
!pip install -e .

# Prepare dataset from Hub
!finetune prepare \
    --hub-id samuellimabraz/quantum-test \
    --output-dir ./swift_data \
    --max-size 640

# Train with ms-swift
!swift sft \
    --model_type qwen2_vl-2b-instruct \
    --dataset ./swift_data/train.jsonl \
    --val_dataset ./swift_data/validation.jsonl \
    --output_dir ./outputs/qwen2-quantum
```
