#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# PEFT Experiment Runner
# Runs experiments defined in experiments.yaml using base.yaml as foundation
#
# Usage: ./scripts/experiment.sh <experiment_name> [--export] [--push]
#
# Examples:
#   ./scripts/experiment.sh rslora              # Train rsLoRA
#   ./scripts/experiment.sh pissa --export      # Train PiSSA + merge
#   ./scripts/experiment.sh dora --push         # Train DoRA + merge + push
#   ./scripts/experiment.sh all                 # Run all experiments
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_CONFIG="${PROJECT_ROOT}/src/finetune/yaml/base.yaml"
EXPERIMENTS_FILE="${PROJECT_ROOT}/src/finetune/experiments.yaml"

# Parse arguments
EXPERIMENT_NAME=""
EXPORT_MODEL=false
PUSH_TO_HUB=false

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
        -h|--help)
            head -15 "$0" | tail -10
            echo ""
            echo "Available experiments:"
            grep -E "^[a-z_]+:" "$EXPERIMENTS_FILE" | sed 's/://' | sed 's/^/  - /'
            exit 0
            ;;
        *)
            EXPERIMENT_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "[error] No experiment name provided"
    echo "Usage: $0 <experiment_name> [--export] [--push]"
    echo ""
    echo "Available experiments:"
    grep -E "^[a-z_]+:" "$EXPERIMENTS_FILE" | sed 's/://' | sed 's/^/  - /'
    exit 1
fi

#=============================================================================
# Environment Variables
#=============================================================================

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-quantum-assistant}"
export MAX_PIXELS="${MAX_PIXELS:-$((1280 * 28 * 28))}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_HF="${USE_HF:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
fi

gpu_ids_clean=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ')
IFS=',' read -ra gpu_array <<< "$gpu_ids_clean"
export NPROC_PER_NODE=${#gpu_array[@]}

#=============================================================================
# Parse experiment config
#=============================================================================

parse_experiment() {
    local exp_name="$1"
    python3 -c "
import yaml
import sys

with open('$EXPERIMENTS_FILE') as f:
    experiments = yaml.safe_load(f)

if '$exp_name' not in experiments:
    print(f'Error: Experiment \"$exp_name\" not found', file=sys.stderr)
    sys.exit(1)

exp = experiments['$exp_name']
for key, value in exp.items():
    # Convert Python types to shell-friendly format
    if isinstance(value, bool):
        value = str(value).lower()
    print(f'{key}={value}')
"
}

run_experiment() {
    local exp_name="$1"
    
    echo "=============================================="
    echo "[experiment] Running: $exp_name"
    echo "=============================================="
    
    # Parse experiment overrides
    local overrides
    overrides=$(parse_experiment "$exp_name")
    
    if [[ $? -ne 0 ]]; then
        echo "[error] Failed to parse experiment: $exp_name"
        return 1
    fi
    
    # Build CLI override arguments
    local cli_args=""
    while IFS='=' read -r key value; do
        cli_args="$cli_args --$key $value"
    done <<< "$overrides"
    
    echo "[experiment] Base config: $BASE_CONFIG"
    echo "[experiment] Overrides: $cli_args"
    echo "----------------------------------------------"
    
    cd "$PROJECT_ROOT"
    
    # Run training
    if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
        NPROC_PER_NODE=$NPROC_PER_NODE uv run --no-sync swift sft \
            --config "$BASE_CONFIG" $cli_args
    else
        uv run --no-sync swift sft --config "$BASE_CONFIG" $cli_args
    fi
    
    echo "[experiment] Training completed: $exp_name"
    
    # Export if requested
    if [[ "$EXPORT_MODEL" == "true" ]]; then
        export_experiment "$exp_name"
    fi
}

export_experiment() {
    local exp_name="$1"
    
    echo ""
    echo "=============================================="
    echo "[experiment] Exporting: $exp_name"
    echo "=============================================="
    
    # Get experiment config
    local output_dir model_name hub_model_id model
    output_dir=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENTS_FILE')); print(d['$exp_name'].get('output_dir', ''))")
    model_name=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENTS_FILE')); print(d['$exp_name'].get('model_name', ''))")
    hub_model_id=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENTS_FILE')); print(d['$exp_name'].get('hub_model_id', ''))")
    model=$(python3 -c "import yaml; d=yaml.safe_load(open('$BASE_CONFIG')); print(d.get('model', ''))")
    
    # Find latest checkpoint
    local latest_run best_checkpoint
    latest_run=$(ls -td "${PROJECT_ROOT}/${output_dir#./}"/v*/ 2>/dev/null | head -1)
    
    if [[ -z "$latest_run" ]]; then
        echo "[error] No training output found for $exp_name"
        return 1
    fi
    
    best_checkpoint=$(ls -td "${latest_run}"checkpoint-*/ 2>/dev/null | head -1)
    
    if [[ -z "$best_checkpoint" ]]; then
        echo "[error] No checkpoint found in $latest_run"
        return 1
    fi
    
    echo "[experiment] Checkpoint: $best_checkpoint"
    
    local merged_dir="${PROJECT_ROOT}/outputs/merged/${model_name}"
    
    # Build export command
    local export_cmd=(
        uv run --no-sync swift export
        --adapters "$best_checkpoint"
        --model "$model"
        --merge_lora true
        --output_dir "$merged_dir"
        --use_hf true
        --exist_ok true
    )
    
    if [[ "$PUSH_TO_HUB" == "true" && -n "$hub_model_id" ]]; then
        export_cmd+=(
            --push_to_hub true
            --hub_model_id "$hub_model_id"
            --hub_private_repo true
        )
        echo "[experiment] Will push to: https://huggingface.co/$hub_model_id"
    fi
    
    "${export_cmd[@]}"
    
    echo "[experiment] Merged model: $merged_dir"
}

#=============================================================================
# Main
#=============================================================================

cd "$PROJECT_ROOT"

echo "=============================================="
echo "[experiment] PEFT Experiment Runner"
echo "=============================================="
echo "[experiment] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[experiment] NPROC_PER_NODE: $NPROC_PER_NODE"
echo "[experiment] Export: $EXPORT_MODEL"
echo "[experiment] Push to Hub: $PUSH_TO_HUB"
echo "----------------------------------------------"

if [[ "$EXPERIMENT_NAME" == "all" ]]; then
    # Run all experiments
    experiments=$(grep -E "^[a-z_]+:" "$EXPERIMENTS_FILE" | sed 's/://')
    for exp in $experiments; do
        run_experiment "$exp"
    done
else
    run_experiment "$EXPERIMENT_NAME"
fi

echo "=============================================="
echo "[experiment] All done!"
echo "=============================================="

