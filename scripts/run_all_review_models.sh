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

# Throughput envelope for this box (RTX PRO 6000, 96 GB). Can be overridden
# from the caller. Config C in the A/B study: 3-draw Pass@1 envelope on r32-2
# = 42.38-43.71 % (1.33 pp), ~105 s per 151-task run. Matches the paper's
# 43.71 % at its upper draw; 34 % tighter than max_num_seqs=64 at the same
# wall-clock.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.93}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs 32 --max-num-batched-tokens 98304 --enable-chunked-prefill --enable-prefix-caching}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-32}"
export EVAL_EXEC_MAX_CONCURRENT="${EVAL_EXEC_MAX_CONCURRENT:-32}"

# Fine-tune system prompt text, used by eval_review_experiments.sh when
# PROMPT_MODE=finetune. Defaults to the file in src/evaluate/config/.
export FINETUNE_SYSTEM_PROMPT_FILE="${FINETUNE_SYSTEM_PROMPT_FILE:-${PROJECT_ROOT}/src/evaluate/config/finetune_system_prompt.txt}"

# Model matrix for the full re-eval pass on RTX PRO 6000 (policy-aligned).
# See paper/review_response_plan.md §1.13.1 for the prompt policy.
#
# Format: MODEL_NAME|HF_MODEL_ID|IS_VLM|RUN_QHE|RUN_PASSK|RUN_QHE_HARD|RUN_PASSK_HARD|RUN_SYNTH|PROMPT_MODE
#
# PROMPT_MODE:
#   finetune : inject the fine-tune-time system prompt (our FT models + all
#              baseline VLMs, for clean base-vs-FT comparison under one prompt)
#   null     : use model's own chat-template default, no system message
#              (IBM Qiskit specialized models, matching their published
#              benchmark convention)
MODELS=(
    # Base VLMs (fine-tune prompt; baseline selection phase)
    "qwen3-vl-base|Qwen/Qwen3-VL-8B-Instruct|true|true|false|true|false|true|finetune"
    "internvl3_5-8b|OpenGVLab/InternVL3_5-8B|true|true|false|true|false|true|finetune"
    "ministral-3-8b|mistralai/Ministral-3-8B-Instruct-2512|true|true|false|true|false|true|finetune"
    "minicpm-v-2_6|openbmb/MiniCPM-V-2_6|true|true|false|true|false|true|finetune"
    # Our fine-tuned VLMs (fine-tune prompt = native invocation)
    "qwen3-vl-ft-r32-1ep|samuellimabraz/Qwen3-VL-8B-rslora-r32|true|true|false|true|false|true|finetune"
    "qwen3-vl-ft-r32-2ep|samuellimabraz/Qwen3-VL-8B-rslora-r32-2|true|true|false|true|false|true|finetune"
    "qwen3-vl-ft-r64-1ep|samuellimabraz/Qwen3-VL-8B-rslora-r64|true|true|false|true|false|true|finetune"
    "qwen3-vl-ft-r32-2ep-textonly|samuellimabraz/Qwen3-VL-8B-rslora-r32-2-textonly|true|true|false|true|false|true|finetune"
    # IBM specialized LLMs (null = model's native chat template, no system msg)
    "granite-3.3-8b-qiskit|Qiskit/granite-3.3-8b-qiskit|false|true|false|true|false|true|null"
    "qwen2.5-coder-14b-qiskit|Qiskit/Qwen2.5-Coder-14B-Qiskit|false|true|false|true|false|true|null"
)

# Pass@k subset (second phase, run after Pass@1 table is complete).
# Four models: best FT, text-only FT control, the paper's strongest specialized
# IBM baseline (14B), and the size-matched 8B IBM baseline for a clean
# parameter-scale comparison at Pass@k.
PASSK_MODELS=(
    "qwen3-vl-ft-r32-2ep-passk|samuellimabraz/Qwen3-VL-8B-rslora-r32-2|true|false|true|false|true|false|finetune"
    "qwen3-vl-ft-r32-2ep-textonly-passk|samuellimabraz/Qwen3-VL-8B-rslora-r32-2-textonly|true|false|true|false|true|false|finetune"
    "qwen2.5-coder-14b-qiskit-passk|Qiskit/Qwen2.5-Coder-14B-Qiskit|false|false|true|false|true|false|null"
    "granite-3.3-8b-qiskit-passk|Qiskit/granite-3.3-8b-qiskit|false|false|true|false|true|false|null"
)

# Caller selects which phase to run:
#   PHASE=pass1 (default) -> MODELS
#   PHASE=passk           -> PASSK_MODELS
#   PHASE=all             -> MODELS then PASSK_MODELS
PHASE="${PHASE:-pass1}"
active_models=()
case "${PHASE}" in
    pass1) active_models=("${MODELS[@]}") ;;
    passk) active_models=("${PASSK_MODELS[@]}") ;;
    all)   active_models=("${MODELS[@]}" "${PASSK_MODELS[@]}") ;;
    *) echo "[run_all] unknown PHASE=${PHASE}; use pass1|passk|all" >&2; exit 2 ;;
esac

is_completed() {
    grep -Fxq "$1" "${CHECKPOINT}" 2>/dev/null
}
mark_completed() {
    grep -Fxq "$1" "${CHECKPOINT}" 2>/dev/null || echo "$1" >> "${CHECKPOINT}"
}

for entry in "${active_models[@]}"; do
    IFS='|' read -r name hf_id is_vlm r_qhe r_passk r_hard r_passk_hard r_synth prompt_mode <<< "${entry}"

    if is_completed "${name}"; then
        echo "[run_all] skip (already completed): ${name}"
        continue
    fi

    echo ""
    echo "=========================================================="
    echo "[run_all] starting: ${name}  (hf=${hf_id}, is_vlm=${is_vlm}, prompt=${prompt_mode})"
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    echo "=========================================================="

    MODEL_NAME="${name}" \
    HF_MODEL_ID="${hf_id}" \
    IS_VLM="${is_vlm}" \
    RUN_QHE="${r_qhe}" \
    RUN_PASSK="${r_passk}" \
    RUN_QHE_HARD="${r_hard}" \
    RUN_PASSK_HARD="${r_passk_hard}" \
    RUN_SYNTH="${r_synth}" \
    PROMPT_MODE="${prompt_mode}" \
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
