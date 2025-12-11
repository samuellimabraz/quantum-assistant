# Fine-tuning Data Preparation

Prepare quantum computing datasets for VLM fine-tuning with ms-swift framework.

## Overview

This module converts the Quantum Assistant dataset into ms-swift compatible format for training Vision-Language Models. It handles:

- Image processing (resize, format conversion)
- Message formatting (system/user/assistant structure)
- Dataset loading from multiple sources (HuggingFace Hub, local parquet, Arrow format)
- Support for text-only and multimodal samples

## Installation

```bash
# Install finetune module
uv sync --package finetune

# Or with pip
cd src/finetune
pip install -e .
```

**Optional: Install with GPU extras**

```bash
uv sync --package finetune --extra gpu
```

**Flash Attention (recommended for training efficiency):**

Flash Attention requires manual installation due to compilation:

```bash
uv pip install flash-attn --no-build-isolation

# prebuilt wheel (recommended):
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3%2Bcu129torch2.8-cp312-cp312-linux_x86_64.whl
```

## Quick Start

### From HuggingFace Hub 

```bash
finetune prepare --hub-id samuellimabraz/quantum-assistant --output-dir ./swift_data
```

### From Local Dataset

```bash
# From downloaded parquet files
finetune prepare --dataset-path /path/to/quantum-assistant --output-dir ./swift_data

# From Arrow format (if saved with save_to_disk)
finetune prepare --dataset-path outputs/final --output-dir outputs/finetune
```

## Dataset Sources

The preparer supports three dataset formats:

| Source | Option | Example | Use Case |
|--------|--------|---------|----------|
| **HuggingFace Hub** | `--hub-id` | `samuellimabraz/quantum-assistant` | Colab, remote training |
| **Local Parquet** | `--dataset-path` | `/content/quantum-assistant` | Downloaded from Hub |
| **Local Arrow** | `--dataset-path` | `outputs/final` | From pipeline output |

### Local Parquet Structure

```
/path/to/quantum-assistant/
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
└── README.md
```

## Output Format

### ms-swift Message Structure

Each sample is converted to:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a quantum computing expert specializing in Qiskit..."
        },
        {
            "role": "user",
            "content": "<image>\nComplete the function to create a Bell state..."
        },
        {
            "role": "assistant",
            "content": "qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nreturn qc"
        }
    ],
    "images": ["images/circuit_abc123.jpg"]
}
```

For text-only samples, `images` field is omitted.

### Output Directory Structure

```
swift_data/
├── train.jsonl          # Training samples
├── validation.jsonl     # Validation samples
├── test.jsonl           # Test samples
├── images/              # Processed images
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── summary.json         # Dataset statistics
```

## CLI Options

```bash
finetune prepare [OPTIONS]

Options:
  -c, --config PATH          YAML configuration file
  -h, --hub-id TEXT          HuggingFace Hub dataset ID
  -d, --dataset-path PATH    Local path to dataset directory
  -o, --output-dir PATH      Output directory [default: outputs/finetune]
  
  --max-size INTEGER         Max image dimension in pixels [default: 640]
  --image-format [JPEG|PNG]  Output format [default: JPEG]
  --max-samples INTEGER      Limit samples per split (for testing)
  --question-types TEXT      Filter by type: "code_generation,qa"
  --no-system-prompt         Exclude system prompt from messages
```

### Examples

**Basic usage:**

```bash
finetune prepare --hub-id samuellimabraz/quantum-assistant
```

**Custom image size:**

```bash
finetune prepare \
    --hub-id samuellimabraz/quantum-assistant \
    --max-size 512 \
    --image-format PNG
```

**Filter by question type:**

```bash
finetune prepare \
    --hub-id samuellimabraz/quantum-assistant \
    --question-types "code_generation,function_completion"
```

**Testing with limited samples:**

```bash
finetune prepare \
    --hub-id samuellimabraz/quantum-assistant \
    --max-samples 100
```

## Python API

```python
from finetune import DatasetPreparer, FinetuneConfig

# Configure from HuggingFace Hub
config = FinetuneConfig(
    hub_id="samuellimabraz/quantum-assistant",
    output_dir="./swift_data",
)

# Or from local path
config = FinetuneConfig(
    dataset_path="/path/to/dataset",
    output_dir="./swift_data",
)

# Customize options
config.image.max_size = 512
config.image.format = "PNG"
config.swift.include_system_prompt = True
config.max_samples_per_split = 1000

# Prepare dataset
preparer = DatasetPreparer(config)
result = preparer.prepare()

print(f"Processed {result.statistics['train']['total']} training samples")
print(f"Output directory: {result.output_dir}")
```

## Integration with ms-swift

After preparing data, train with ms-swift:

```bash
swift sft \
    --model_type qwen3_vl-8b-instruct \
    --dataset ./swift_data/train.jsonl \
    --val_dataset ./swift_data/validation.jsonl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_rslora true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --lora_dropout 0.10 \
    --output_dir ./outputs/qwen3-vl-quantum
```

**Recommended configuration based on experiments:**

- **PEFT Method**: rsLoRA (rank-stabilized LoRA)
- **Rank**: 32
- **Epochs**: 1 (avoids overfitting on specialized dataset)
- **Batch Size**: 32 effective (16 × 2 accumulation steps)
- **Learning Rate**: 2e-4 with cosine decay
- **Regularization**: Weight decay 0.05, dropout 0.10

See [docs/finetune.md](../../docs/finetune.md) for detailed experiment results.

## Google Colab Example

```python
# Install dependencies
!pip install -q datasets pillow tqdm pyyaml pydantic typer rich

# Clone repo
!git clone https://github.com/samuellimabraz/quantum-assistant.git
%cd quantum-assistant
!pip install -e ./src/finetune

# Prepare dataset from Hub
!finetune prepare \
    --hub-id samuellimabraz/quantum-assistant \
    --output-dir ./swift_data \
    --max-size 640

# Install ms-swift
!pip install ms-swift[llm] -U

# Train (requires GPU runtime)
!swift sft \
    --model_type qwen3_vl-8b-instruct \
    --dataset ./swift_data/train.jsonl \
    --val_dataset ./swift_data/validation.jsonl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_rslora true \
    --num_train_epochs 1 \
    --output_dir ./outputs/qwen3-vl-quantum
```

## System Prompts

The default system prompt specializes the model for Qiskit:

```
You are a quantum computing expert assistant specializing in Qiskit.
Provide accurate, clear, and well-structured responses about quantum 
computing concepts, algorithms, and code implementation. Use Qiskit 2.0 
best practices and avoid deprecated APIs.
```

To use a minimal prompt:

```bash
finetune prepare --hub-id samuellimabraz/quantum-assistant --no-system-prompt
```

Or customize via Python API:

```python
config.swift.system_prompt = "Custom quantum assistant prompt..."
```

## Image Processing

Images are automatically:

1. **Resized**: Maintaining aspect ratio, max dimension = `--max-size` (default 640px)
2. **Converted**: To specified format (JPEG default for smaller size)
3. **Saved**: In `{output_dir}/images/` with unique filenames

**Trade-offs:**

- **JPEG**: Smaller files (~50-70% of PNG), slight quality loss, faster training
- **PNG**: Lossless quality, larger files, slower I/O

For high-detail circuit diagrams, consider PNG with `--max-size 1024`.

## Troubleshooting

### Import Error: No module named 'finetune'

Ensure you're using the correct environment:

```bash
# With uv
uv sync --package finetune
uv run finetune prepare --hub-id samuellimabraz/quantum-assistant

# Or activate venv
source .venv/bin/activate
finetune prepare --hub-id samuellimabraz/quantum-assistant
```

### Dataset Not Found

For local datasets, verify structure:

```bash
ls /path/to/quantum-assistant/data/
# Should show: train-*.parquet, validation-*.parquet, test-*.parquet
```

### Out of Memory During Training

Reduce batch size or use gradient checkpointing:

```bash
swift sft \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    ...
```

## Documentation

- **Fine-tuning Experiments**: See [docs/finetune.md](../../docs/finetune.md) for detailed PEFT comparison, hyperparameter tuning results, and configuration recommendations

## License

Apache 2.0
