#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# Swift Fine-tuning Script for Quantum Assistant
#
# Usage: ./scripts/tune.sh [OPTIONS] [config_file]
#
# Options:
#   --export     Export and merge LoRA weights locally (vLLM-ready)
#   --push       Export, merge, and push to HuggingFace Hub
#   --skip-train Skip training, only export/push existing checkpoint
#
# Examples:
#   ./scripts/tune.sh                           # Train only
#   ./scripts/tune.sh --export                  # Train + merge LoRA locally
#   ./scripts/tune.sh --push                    # Train + merge + push to Hub
#   ./scripts/tune.sh --skip-train --push       # Push existing checkpoint
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
EXPORT_MODEL=false
PUSH_TO_HUB=false
SKIP_TRAIN=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --export)
            EXPORT_MODEL=true
            shift
            ;;
        --push)
            EXPORT_MODEL=true
            PUSH_TO_HUB=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# Default config file path
DEFAULT_CONFIG="${PROJECT_ROOT}/src/finetune/yaml/swift.yaml"
CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG}"

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[error] Config file not found: $CONFIG_FILE"
    echo "[info] Usage: $0 [--export] [--push] [--skip-train] [path/to/config.yaml]"
    exit 1
fi

#=============================================================================
# Parse configuration from YAML
#=============================================================================

# Helper function to extract value from YAML (simple grep-based parser)
yaml_get() {
    local key="$1"
    local file="$2"
    grep -E "^${key}:" "$file" | head -1 | sed "s/^${key}:[[:space:]]*//" | tr -d "'" | tr -d '"'
}

# Read config values from YAML
CONFIG_MODEL=$(yaml_get "model" "$CONFIG_FILE")
CONFIG_OUTPUT_DIR=$(yaml_get "output_dir" "$CONFIG_FILE")
CONFIG_HUB_MODEL_ID=$(yaml_get "hub_model_id" "$CONFIG_FILE")
CONFIG_MODEL_NAME=$(yaml_get "model_name" "$CONFIG_FILE")

# Resolve relative paths
if [[ "$CONFIG_OUTPUT_DIR" == ./* ]]; then
    CONFIG_OUTPUT_DIR="${PROJECT_ROOT}/${CONFIG_OUTPUT_DIR#./}"
fi

echo "=============================================="
echo "[swift] Quantum Assistant Fine-tuning"
echo "=============================================="
echo "[swift] Project root: $PROJECT_ROOT"
echo "[swift] Config file: $CONFIG_FILE"
echo "[swift] Model: $CONFIG_MODEL"
echo "[swift] Model name: $CONFIG_MODEL_NAME"
echo "[swift] Output dir: $CONFIG_OUTPUT_DIR"
echo "[swift] Hub model ID: $CONFIG_HUB_MODEL_ID"
echo "[swift] Skip training: $SKIP_TRAIN"
echo "[swift] Export (merge LoRA): $EXPORT_MODEL"
echo "[swift] Push to Hub: $PUSH_TO_HUB"

#=============================================================================
# Environment Variables
#=============================================================================

# HuggingFace token (required for model downloads and hub uploads)
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "[warning] HF_TOKEN not set. Hub features may not work."
fi

# WandB settings (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-quantum-assistant}"

# Image processing settings for Qwen3-VL
export MAX_PIXELS="${MAX_PIXELS:-$((1280 * 28 * 28))}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Use HuggingFace hub
export USE_HF="${USE_HF:-1}"

#=============================================================================
# GPU Configuration
#=============================================================================

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${GPU_IDS:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    else
        echo "[warning] Neither CUDA_VISIBLE_DEVICES nor GPU_IDS set. Defaulting to GPU 0."
        export CUDA_VISIBLE_DEVICES="0"
    fi
fi

# Calculate number of GPUs for distributed training
gpu_ids_clean=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ')
IFS=',' read -ra gpu_array <<< "$gpu_ids_clean"
NPROC_PER_NODE=${#gpu_array[@]}
export NPROC_PER_NODE

echo "----------------------------------------------"
echo "[swift] Environment Configuration:"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  - MAX_PIXELS: $MAX_PIXELS"
echo "  - IMAGE_MAX_TOKEN_NUM: $IMAGE_MAX_TOKEN_NUM"
echo "  - HF_TOKEN: ${HF_TOKEN:+[SET]}"
echo "  - WANDB_PROJECT: $WANDB_PROJECT"
echo "  - USE_HF: $USE_HF"
echo "----------------------------------------------"

cd "$PROJECT_ROOT"

#=============================================================================
# Run Training
#=============================================================================

if [[ "$SKIP_TRAIN" == "false" ]]; then
    echo "[swift] Starting training job..."
    echo "[swift] Command: uv run --no-sync swift sft --config $CONFIG_FILE"
    echo "=============================================="

    if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
        # Multi-GPU training with torchrun
        echo "[swift] Running distributed training on $NPROC_PER_NODE GPUs..."
        NPROC_PER_NODE=$NPROC_PER_NODE uv run --no-sync swift sft --config "$CONFIG_FILE"
    else
        # Single GPU training
        uv run --no-sync swift sft --config "$CONFIG_FILE"
    fi

    echo "=============================================="
    echo "[swift] Training completed successfully!"
    echo "=============================================="
fi

#=============================================================================
# Export / Merge LoRA / Push to Hub
#=============================================================================

if [[ "$EXPORT_MODEL" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "[swift] Exporting and merging LoRA weights..."
    echo "=============================================="
    
    # Find the latest training run directory (read from config)
    OUTPUT_BASE="$CONFIG_OUTPUT_DIR"
    LATEST_RUN=$(ls -td "${OUTPUT_BASE}"/v*/ 2>/dev/null | head -1)
    
    if [[ -z "$LATEST_RUN" ]]; then
        echo "[error] No training output found in $OUTPUT_BASE"
        exit 1
    fi
    
    # Find the best or latest checkpoint
    BEST_CHECKPOINT=$(ls -td "${LATEST_RUN}"checkpoint-*/ 2>/dev/null | head -1)
    
    if [[ -z "$BEST_CHECKPOINT" ]]; then
        echo "[error] No checkpoint found in $LATEST_RUN"
        exit 1
    fi
    
    echo "[swift] Found checkpoint: $BEST_CHECKPOINT"
    
    # Configuration from YAML (with env var overrides)
    HUB_MODEL_ID="${HUB_MODEL_ID:-$CONFIG_HUB_MODEL_ID}"
    MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/merged/${CONFIG_MODEL_NAME}}"
    
    echo "[swift] Merged output dir: $MERGED_OUTPUT_DIR"
    
    # Build export command
    # --merge_lora true: Merges LoRA weights into base model (vLLM-ready)
    # --model: Specifies base model for correct metadata
    # --output_dir: Where to save merged model locally
    # --exist_ok true: Allow pushing to existing repos
    EXPORT_CMD=(
        uv run --no-sync swift export
        --adapters "$BEST_CHECKPOINT"
        --model "$CONFIG_MODEL"
        --merge_lora true
        --output_dir "$MERGED_OUTPUT_DIR"
        --use_hf true
        --exist_ok true
    )
    
    # Add push options if requested
    if [[ "$PUSH_TO_HUB" == "true" ]]; then
        EXPORT_CMD+=(
            --push_to_hub true
            --hub_model_id "$HUB_MODEL_ID"
            --hub_private_repo true
        )
        echo "[swift] Will push merged model to: https://huggingface.co/$HUB_MODEL_ID"
    fi
    
    echo "[swift] Running export command..."
    "${EXPORT_CMD[@]}"
    
    echo "=============================================="
    echo "[swift] Export completed!"
    echo "[swift] Merged model saved to: $MERGED_OUTPUT_DIR"
    
    if [[ "$PUSH_TO_HUB" == "true" ]]; then
        echo "[swift] Model pushed to: https://huggingface.co/$HUB_MODEL_ID"
    fi
    
    echo ""
    echo "[swift] To use with vLLM:"
    echo "  vllm serve $MERGED_OUTPUT_DIR"
    echo "=============================================="
fi
