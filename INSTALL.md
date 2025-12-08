# Installation Guide

This guide explains how to install and use the quantum-assistant modules with uv workspaces.

## Project Structure

The project uses **uv workspaces** with:
- **Root package** (`synthetic-data`): Contains `synthetic_data`, `evaluate`, and shared `models` modules
- **Workspace member** (`finetune`): Separate ML dependencies for finetuning

```
quantum-assistant/
├── pyproject.toml              # Root: synthetic_data + evaluate
├── .venv/                      # Single shared virtual environment
└── src/
    ├── synthetic_data/         # Dataset generation
    ├── evaluate/               # Model evaluation
    ├── models/                 # Shared model clients
    └── finetune/
        └── pyproject.toml      # Finetuning dependencies
```

## Installation

### Option 1: Install Everything

```bash
cd /root/quantum-assistant

# Install all dependencies (root + finetune + GPU libs)
uv sync --package finetune --extra gpu
```

### Option 2: Install Selectively

```bash
# Just root dependencies (synthetic_data + evaluate)
uv sync

# Add finetune later
uv sync --package finetune

# Add GPU libraries
uv sync --package finetune --extra gpu
```

### Flash Attention (Optional)

Flash-attn must be installed separately due to compilation time (~10 minutes):

```bash
# After installing finetune
uv pip install flash-attn --no-build-isolation
```

## Usage

### Running Commands

```bash
# Finetune
uv run finetune --help
uv run finetune prepare --hub-id samuellimabraz/quantum-test

# Evaluate
uv run evaluate --help
uv run evaluate run --config config/eval_config.yaml

# Synthetic-data
uv run synthetic-data --help
uv run synthetic-data pipeline --config yaml/config.yaml
```

### Alternative: Direct Execution

```bash
cd /root/quantum-assistant
source .venv/bin/activate

finetune prepare --hub-id samuellimabraz/quantum-test
evaluate run --config config/eval_config.yaml
synthetic-data pipeline --config yaml/config.yaml
```

## Installed Packages

After full installation, you'll have:

| Package | Location | Dependencies |
|---------|----------|--------------|
| `synthetic-data` | Root | Qiskit, httpx, pydantic, datasets, etc. |
| `finetune` | `src/finetune` | ms-swift, transformers, torch, qwen-vl-utils |
| GPU extras | Optional | deepspeed, liger-kernel |
| flash-attn | Manual | Flash attention (requires CUDA) |

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'models'`:

1. Ensure you're using `uv run` or activated `.venv`
2. Reinstall: `uv sync --reinstall-package synthetic-data`

### Flash-attn Build Fails

If flash-attn compilation fails:

1. Ensure torch is installed first: `uv sync --package finetune`
2. Install manually: `uv pip install flash-attn --no-build-isolation`
3. Or skip it - it's optional for training

### Slow Installation

The first installation downloads ~3GB of packages. Subsequent syncs are fast due to uv's cache.

## Environment Variables

Create a `.env` file in the project root:

```bash
# For synthetic data generation
VISION_MODEL_BASE_URL=https://api.openai.com/v1
VISION_MODEL_API_KEY=sk-...
VISION_MODEL_NAME=gpt-4o

QUESTION_MODEL_BASE_URL=...
QUESTION_MODEL_API_KEY=...
QUESTION_MODEL_NAME=...

# For evaluation
MODEL_BASE_URL=http://localhost:8000/v1
API_KEY=your-key
MODEL_NAME=qwen2.5-coder-14b

# For HuggingFace uploads
HF_TOKEN=hf_...
```

## Development

### Adding Dependencies

**Root package** (synthetic_data/evaluate):
```bash
# Edit pyproject.toml in root
uv add <package>
```

**Finetune package**:
```bash
# Edit src/finetune/pyproject.toml
uv add --package finetune <package>
```

### Running Tests

```bash
uv run pytest tests/
```

## Next Steps

- **Generate synthetic data**: See `src/synthetic_data/README.md`
- **Evaluate models**: See `src/evaluate/README.md`
- **Prepare finetuning data**: See `src/finetune/README.md`

