#!/usr/bin/env bash
# =============================================================================
# Serve a model with vLLM, wait until /v1/models answers, run the review
# compute driver, then tear down. Single entry point so the whole per-model
# run fits one command.
#
# Required env:
#   HF_MODEL_ID      HuggingFace id for ``vllm serve`` (e.g. Qwen/Qwen3-VL-8B-Instruct)
#   MODEL_NAME       --served-model-name AND filesystem tag (e.g. qwen3-vl-base)
#   IS_VLM           "true" for VLMs, "false" for text-only LLMs
#
# Optional env (all default sensibly):
#   VLLM_PORT            default 8000
#   VLLM_LOG_DIR         default outputs/logs/vllm
#   DRIVER_LOG           default outputs/logs/driver/${MODEL_NAME}.log
#   DTYPE                default bfloat16
#   MAX_MODEL_LEN        default 8192
#   GPU_MEM_UTIL         default 0.92
#   VLLM_EXTRA_ARGS      free-form extra args forwarded to vllm serve
#   READY_TIMEOUT_S      default 600 (10 min)
#   RUN_PASSK, RUN_QHE, RUN_QHE_HARD, RUN_SYNTH, RUN_IMG_MASK — see eval_review_experiments.sh
#
# Writes vLLM logs to ${VLLM_LOG_DIR}/${MODEL_NAME}.log so a background run can
# be inspected later with tail -f.
# =============================================================================

set -u -o pipefail

# Optional shared env file (e.g. for HF_HOME on boxes where the default cache
# location is not large enough). See /workspace/hf_env.sh on the review box.
if [[ -f "${HF_ENV_FILE:-/workspace/hf_env.sh}" ]]; then
    # shellcheck disable=SC1090
    source "${HF_ENV_FILE:-/workspace/hf_env.sh}"
fi

: "${HF_MODEL_ID:?HF_MODEL_ID is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${IS_VLM:?IS_VLM is required (true|false)}"

VLLM_PORT="${VLLM_PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${PROJECT_ROOT}/outputs/logs/vllm}"
DRIVER_LOG="${DRIVER_LOG:-${PROJECT_ROOT}/outputs/logs/driver/${MODEL_NAME}.log}"
mkdir -p "${VLLM_LOG_DIR}" "$(dirname "${DRIVER_LOG}")"

VLLM_BIN="${VLLM_BIN:-${PROJECT_ROOT}/.venv/bin/vllm}"
if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "[serve_and_eval] vLLM not found at ${VLLM_BIN}" >&2
    echo "[serve_and_eval] Set VLLM_BIN or run 'uv pip install vllm' in the project venv." >&2
    exit 10
fi

# Purge cached weights for this model before serving (set PURGE_HF_CACHE=1).
# We observed that a partial or interleaved HF download can leave the snapshot
# tree tagged complete while serving garbled weights, producing ~25 pp Pass@1
# drops that recover after a clean re-fetch. Purging is quick (directory move,
# background delete) and robust, so is on by default for scripted runs.
if [[ "${PURGE_HF_CACHE:-1}" == "1" ]]; then
    hf_hub_cache="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
    model_dir_name="models--${HF_MODEL_ID//\//--}"
    cache_target="${hf_hub_cache}/${model_dir_name}"
    if [[ -d "${cache_target}" ]]; then
        trash_root="${PURGE_HF_TRASH:-${hf_hub_cache}/.trash}"
        mkdir -p "${trash_root}"
        trash_path="${trash_root}/${model_dir_name}.$$.$(date -u +%s)"
        echo "[serve_and_eval] purging HF cache for ${HF_MODEL_ID} -> ${trash_path}"
        mv "${cache_target}" "${trash_path}" || true
        (rm -rf "${trash_path}" >/dev/null 2>&1 &)
    fi
fi

export MODEL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME

vllm_log="${VLLM_LOG_DIR}/${MODEL_NAME}.log"

declare -a vllm_cmd=(
    "${VLLM_BIN}" serve "${HF_MODEL_ID}"
    --port "${VLLM_PORT}"
    --served-model-name "${MODEL_NAME}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --trust-remote-code
)

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    vllm_cmd+=(${VLLM_EXTRA_ARGS})
fi

printf '[serve_and_eval] starting vllm: %s\n' "${vllm_cmd[*]}"
"${vllm_cmd[@]}" >"${vllm_log}" 2>&1 &
vllm_pid=$!

cleanup() {
    if kill -0 "${vllm_pid}" 2>/dev/null; then
        echo "[serve_and_eval] stopping vllm pid=${vllm_pid}"
        kill "${vllm_pid}" 2>/dev/null || true
        # Give vllm a few seconds to exit cleanly; escalate if it refuses.
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "${vllm_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${vllm_pid}" 2>/dev/null || true
    fi
    # Best-effort wait for the port to be freed before the next model loads.
    sleep 5
}
trap cleanup EXIT INT TERM

# Wait for the server to answer /v1/models.
echo "[serve_and_eval] waiting up to ${READY_TIMEOUT_S}s for ${MODEL_BASE_URL} ..."
ready=0
for i in $(seq 1 "${READY_TIMEOUT_S}"); do
    if ! kill -0 "${vllm_pid}" 2>/dev/null; then
        echo "[serve_and_eval] vllm exited during startup; tail of log:" >&2
        tail -n 80 "${vllm_log}" >&2
        exit 20
    fi
    if curl -fsS -m 5 "${MODEL_BASE_URL}/models" >/dev/null 2>&1; then
        ready=1
        echo "[serve_and_eval] vllm ready after ${i}s"
        break
    fi
    sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
    echo "[serve_and_eval] timeout waiting for vllm; tail of log:" >&2
    tail -n 80 "${vllm_log}" >&2
    exit 21
fi

echo "[serve_and_eval] running driver; log -> ${DRIVER_LOG}"
bash "${SCRIPT_DIR}/eval_review_experiments.sh" 2>&1 | tee "${DRIVER_LOG}"
driver_status=${PIPESTATUS[0]}

echo "[serve_and_eval] driver exit=${driver_status}"
exit "${driver_status}"
