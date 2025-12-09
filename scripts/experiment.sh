#!/usr/bin/env bash
set -euo pipefail

#=============================================================================
# PEFT Experiment Runner
# Runs experiments defined in experiments.yaml using base.yaml as foundation
#
# Usage: ./scripts/experiment.sh <experiment_name> [options]
#
# Examples:
#   ./scripts/experiment.sh rslora              # Train rsLoRA
#   ./scripts/experiment.sh pissa --export      # Train PiSSA + merge
#   ./scripts/experiment.sh dora --push         # Train DoRA + merge + push
#   ./scripts/experiment.sh all                 # Run all experiments
#   ./scripts/experiment.sh rslora_r32 --export-only  # Just merge (skip train)
#   ./scripts/experiment.sh rslora_r32 --push-only    # Just merge + push (skip train)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_CONFIG="${PROJECT_ROOT}/src/finetune/yaml/base.yaml"
EXPERIMENTS_FILE="${PROJECT_ROOT}/src/finetune/experiments.yaml"

# Parse arguments
EXPERIMENT_NAME=""
EXPORT_MODEL=false
PUSH_TO_HUB=false
PUSH_ARTIFACTS_ONLY=false
EXPORT_ONLY=false
SKIP_TRAIN=false

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
        --push-artifacts)
            PUSH_ARTIFACTS_ONLY=true
            shift
            ;;
        --export-only)
            # Skip training, only export (useful when training completed but export failed)
            EXPORT_ONLY=true
            EXPORT_MODEL=true
            SKIP_TRAIN=true
            shift
            ;;
        --push-only)
            # Skip training, export and push (useful when training completed but export failed)
            EXPORT_ONLY=true
            EXPORT_MODEL=true
            PUSH_TO_HUB=true
            SKIP_TRAIN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <experiment_name> [--export] [--push] [--push-artifacts] [--export-only] [--push-only]"
            echo ""
            echo "Options:"
            echo "  --export          Merge LoRA weights after training"
            echo "  --push            Merge + push model to HuggingFace Hub"
            echo "  --push-artifacts  Only upload training artifacts (no training)"
            echo "  --export-only     Skip training, only merge LoRA weights (for failed exports)"
            echo "  --push-only       Skip training, merge and push to Hub (for failed exports)"
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
    echo "[experiment] Training run: $latest_run"
    
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
            # --hub_private_repo true
        )
        echo "[experiment] Will push merged model to: https://huggingface.co/$hub_model_id"
    fi
    
    "${export_cmd[@]}"
    
    echo "[experiment] Merged model: $merged_dir"
    
    # Upload training artifacts if pushing to hub
    if [[ "$PUSH_TO_HUB" == "true" && -n "$hub_model_id" ]]; then
        upload_training_artifacts "$hub_model_id" "$latest_run" "$exp_name"
    fi
}

push_artifacts_only() {
    local exp_name="$1"
    
    echo "=============================================="
    echo "[experiment] Pushing artifacts only: $exp_name"
    echo "=============================================="
    
    # Get experiment config
    local output_dir hub_model_id
    output_dir=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENTS_FILE')); print(d['$exp_name'].get('output_dir', ''))")
    hub_model_id=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENTS_FILE')); print(d['$exp_name'].get('hub_model_id', ''))")
    
    if [[ -z "$hub_model_id" ]]; then
        echo "[error] No hub_model_id found for $exp_name"
        return 1
    fi
    
    # Find latest run
    local latest_run
    latest_run=$(ls -td "${PROJECT_ROOT}/${output_dir#./}"/v*/ 2>/dev/null | head -1)
    
    if [[ -z "$latest_run" ]]; then
        echo "[error] No training output found for $exp_name in ${output_dir}"
        return 1
    fi
    
    echo "[experiment] Found run: $latest_run"
    echo "[experiment] Hub repo: $hub_model_id"
    
    upload_training_artifacts "$hub_model_id" "$latest_run" "$exp_name"
}

upload_training_artifacts() {
    local hub_model_id="$1"
    local training_run_dir="$2"
    local exp_name="$3"
    
    echo ""
    echo "=============================================="
    echo "[experiment] Uploading training artifacts..."
    echo "=============================================="
    echo "[experiment] Repo: $hub_model_id"
    echo "[experiment] Source: $training_run_dir"
    
    # Get the run version name (e.g., v0-20251208-191800)
    local run_version
    run_version=$(basename "$training_run_dir")
    local artifacts_path="training_artifacts/${run_version}"
    
    # Create a temporary directory with all artifacts to upload in one commit
    local temp_dir
    temp_dir=$(mktemp -d)
    local artifacts_dir="${temp_dir}/${artifacts_path}"
    mkdir -p "$artifacts_dir"
    
    # Copy training artifacts
    echo "[experiment] Collecting artifacts..."
    
    # Training logs
    [[ -f "${training_run_dir}/logging.jsonl" ]] && \
        cp "${training_run_dir}/logging.jsonl" "$artifacts_dir/"
    
    # Arguments/config
    [[ -f "${training_run_dir}/args.json" ]] && \
        cp "${training_run_dir}/args.json" "$artifacts_dir/"
    
    # Trainer state
    [[ -f "${training_run_dir}/trainer_state.json" ]] && \
        cp "${training_run_dir}/trainer_state.json" "$artifacts_dir/"
    
    # Training images (loss curves, etc.)
    [[ -d "${training_run_dir}/images" ]] && \
        cp -r "${training_run_dir}/images" "$artifacts_dir/"
    
    # TensorBoard runs
    [[ -d "${training_run_dir}/runs" ]] && \
        cp -r "${training_run_dir}/runs" "$artifacts_dir/"
    
    # Copy experiment configs
    cp "$EXPERIMENTS_FILE" "$artifacts_dir/experiments.yaml"
    cp "$BASE_CONFIG" "$artifacts_dir/base.yaml"
    
    echo "[experiment] Artifacts collected:"
    ls -la "$artifacts_dir"
    
    # Upload all artifacts in one commit using huggingface-cli
    echo "[experiment] Uploading to HuggingFace Hub..."
    uv run --no-sync huggingface-cli upload \
        "$hub_model_id" \
        "$artifacts_dir" \
        "$artifacts_path" \
        --repo-type model \
        --commit-message "Add training artifacts for ${run_version}"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    echo "=============================================="
    echo "[experiment] Artifacts uploaded successfully!"
    echo "[experiment] View at: https://huggingface.co/$hub_model_id/tree/main/$artifacts_path"
    echo "=============================================="
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
echo "[experiment] Push artifacts only: $PUSH_ARTIFACTS_ONLY"
echo "[experiment] Export only (skip train): $EXPORT_ONLY"
echo "----------------------------------------------"

# Handle push-artifacts-only mode
if [[ "$PUSH_ARTIFACTS_ONLY" == "true" ]]; then
    push_artifacts_only "$EXPERIMENT_NAME"
    exit 0
fi

# Handle export-only mode (skip training)
if [[ "$EXPORT_ONLY" == "true" ]]; then
    echo "[experiment] Skipping training, running export only..."
    export_experiment "$EXPERIMENT_NAME"
    exit 0
fi

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
