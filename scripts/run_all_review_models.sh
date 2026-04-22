#!/usr/bin/env bash
# =============================================================================
# Top-level orchestrator for the review compute pass.
# Iterates over all six models and runs serve_and_eval.sh for each. Designed
# to be launched once with nohup and then walked away from.
#
# Checkpoint file at outputs/logs/driver/.completed records which models have
# finished so that reruns resume cleanly.
# =============================================================================

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${PROJECT_ROOT}/outputs/logs/driver/.completed"
mkdir -p "$(dirname "${CHECKPOINT}")"
touch "${CHECKPOINT}"

# Defaults that apply to every model; individual entries may override.
: "${QHE_PATH:?QHE_PATH required}"
: "${QHE_HARD_PATH:?QHE_HARD_PATH required}"
: "${SYNTH_DATASET_PATH:?SYNTH_DATASET_PATH required}"
export QHE_PATH QHE_HARD_PATH SYNTH_DATASET_PATH
export MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-EMPTY}"

# Model table — edit HF ids if needed (r32-1ep / r64-1ep TBC).
# Format: MODEL_NAME|HF_MODEL_ID|IS_VLM|RUN_QHE|RUN_PASSK|RUN_QHE_HARD|RUN_SYNTH|RUN_IMG_MASK
MODELS=(
    "qwen3-vl-base|Qwen/Qwen3-VL-8B-Instruct|true|true|true|true|true|true"
    "qwen3-vl-ft-r32-1ep|samuellimabraz/Qwen3-VL-8B-rslora-r32|true|true|true|true|true|true"
    "qwen3-vl-ft-r32-2ep|samuellimabraz/Qwen3-VL-8B-rslora-r32-2|true|true|true|true|true|true"
    "qwen3-vl-ft-r64-1ep|samuellimabraz/Qwen3-VL-8B-rslora-r64|true|true|true|true|true|true"
    "granite-3.3-8b-qiskit|Qiskit/granite-3.3-8b-qiskit|false|true|true|true|true|false"
    "qwen2.5-coder-14b-qiskit|Qiskit/Qwen2.5-Coder-14B-Qiskit|false|false|true|false|false|false"
)

is_completed() {
    grep -Fxq "$1" "${CHECKPOINT}" 2>/dev/null
}
mark_completed() {
    grep -Fxq "$1" "${CHECKPOINT}" 2>/dev/null || echo "$1" >> "${CHECKPOINT}"
}

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name hf_id is_vlm r_qhe r_passk r_hard r_synth r_mask <<< "${entry}"

    if is_completed "${name}"; then
        echo "[run_all] skip (already completed): ${name}"
        continue
    fi

    echo ""
    echo "=========================================================="
    echo "[run_all] starting: ${name}  (hf=${hf_id}, is_vlm=${is_vlm})"
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    echo "=========================================================="

    MODEL_NAME="${name}" \
    HF_MODEL_ID="${hf_id}" \
    IS_VLM="${is_vlm}" \
    RUN_QHE="${r_qhe}" \
    RUN_PASSK="${r_passk}" \
    RUN_QHE_HARD="${r_hard}" \
    RUN_SYNTH="${r_synth}" \
    RUN_IMG_MASK="${r_mask}" \
    bash "${SCRIPT_DIR}/serve_and_eval.sh"
    status=$?

    if [[ ${status} -eq 0 ]]; then
        mark_completed "${name}"
        echo "[run_all] ${name} completed (checkpointed)."
    else
        echo "[run_all] ${name} FAILED (exit=${status}); stopping."
        exit "${status}"
    fi
done

echo ""
echo "[run_all] All models completed. Running post-aggregation steps..."
bash "${SCRIPT_DIR}/post_aggregate.sh"
