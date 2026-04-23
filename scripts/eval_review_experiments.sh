#!/usr/bin/env bash
# =============================================================================
# Review compute experiments driver (one model at a time).
#
# Assumes a vLLM server is already running at $MODEL_BASE_URL with served
# model name $MODEL_NAME. Runs the subset of steps appropriate for the model
# class, toggled by environment variables:
#
#   MODEL_NAME       served-model-name (required; also used as filesystem tag)
#   MODEL_BASE_URL   vLLM endpoint (default http://localhost:8000/v1)
#   API_KEY          dummy api key for vLLM (default EMPTY)
#   IS_VLM           "true" for Qwen3-VL candidates, "false" for text LLMs
#   RUN_QHE          "true" (default) to run QHE Pass@1
#   RUN_QHE_HARD     "true" (default) to run QHE-Hard Pass@1
#   RUN_PASSK        "true" to run QHE Pass@k sweep (R1.4), "false" to skip
#   RUN_PASSK_HARD   "true" to run QHE-Hard Pass@k sweep (R1.4), "false" to skip
#   RUN_SYNTH        "true" (default) to run full synthetic test split
#                     (VLMs use multimodal path, text LLMs use --text-only)
#   QHE_PATH                /workspace/qiskit-human-eval/.../dataset_qiskit_test_human_eval.json
#   QHE_HARD_PATH           /workspace/qiskit-human-eval/.../dataset_qiskit_test_human_eval_hard.json
#   SYNTH_DATASET_PATH      HF dataset directory (save_to_disk or parquet layout)
#
# Outputs are written under outputs/evaluate/<benchmark>/<MODEL_NAME>/ so
# prior results are never overwritten.
# =============================================================================

set -u
set -o pipefail

MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
IS_VLM="${IS_VLM:-false}"
RUN_QHE="${RUN_QHE:-true}"
RUN_QHE_HARD="${RUN_QHE_HARD:-true}"
RUN_PASSK="${RUN_PASSK:-false}"
RUN_PASSK_HARD="${RUN_PASSK_HARD:-false}"
RUN_SYNTH="${RUN_SYNTH:-true}"

if [[ -z "${MODEL_NAME:-}" ]]; then
    echo "ERROR: MODEL_NAME is required." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
cd "${PROJECT_ROOT}/src"

export MODEL_BASE_URL API_KEY MODEL_NAME
export QHE_PATH QHE_HARD_PATH SYNTH_DATASET_PATH

# Tag each run's output directory with the model name so multiple models
# produce clearly separated files.
results_root="${PROJECT_ROOT}/outputs/evaluate"
export QHE_RESULTS_DIR="${results_root}/qiskit-humaneval/${MODEL_NAME}"
export QHE_HARD_RESULTS_DIR="${results_root}/qiskit-humaneval-hard/${MODEL_NAME}"
export QHE_PASSK_RESULTS_DIR="${results_root}/qiskit-humaneval-passk/${MODEL_NAME}"
export QHE_HARD_PASSK_RESULTS_DIR="${results_root}/qiskit-humaneval-hard-passk/${MODEL_NAME}"
export SYNTH_RESULTS_DIR="${results_root}/synthetic/${MODEL_NAME}"
mkdir -p "${QHE_RESULTS_DIR}" "${QHE_HARD_RESULTS_DIR}" "${QHE_PASSK_RESULTS_DIR}" \
    "${QHE_HARD_PASSK_RESULTS_DIR}" "${SYNTH_RESULTS_DIR}"

CONFIG_DIR="evaluate/config"

log_header() {
    printf "\n========== %s ==========\n" "$*"
}

# Pin the evaluate CLI's output path so each benchmark/condition lands at a
# distinct, resumable file, and apply per-model overrides (prompt policy,
# concurrency) driven by the following environment variables:
#
#   PROMPT_MODE             "finetune" | "null" | "default" (default: keep base config)
#   FINETUNE_SYSTEM_PROMPT  inlined prompt text (used when PROMPT_MODE=finetune;
#                           defaults to src/evaluate/config/finetune_system_prompt.txt)
#   EVAL_MAX_CONCURRENT         override metrics.max_concurrent
#   EVAL_EXEC_MAX_CONCURRENT    override metrics.execution_max_concurrent
#
# Usage: run_config <base_config_path> <output_dir> <slug>
#   - base_config_path: shipped YAML, unchanged
#   - output_dir: where the results file should be written
#   - slug: short tag added to the filename (e.g. "qhe", "qhe_passk", "synth")
run_config() {
    local base_config="$1" target_dir="$2" slug="$3"
    mkdir -p "${target_dir}"
    local stamp
    stamp="$(date -u +'%Y%m%dT%H%M%SZ')"
    local result_file="${target_dir}/${MODEL_NAME}_${slug}_${stamp}.json"
    local tmp_config
    tmp_config="$(mktemp --suffix=.yaml)"

    # Build the effective config via a small Python merger. YAML duplicate-key
    # tricks lose nested fields when PyYAML overwrites whole blocks; deep-merge
    # avoids that.
    BASE_CONFIG="${base_config}" TMP_CONFIG="${tmp_config}" \
    RESULT_FILE="${result_file}" TARGET_DIR="${target_dir}" \
    "${PYTHON}" - <<'PY'
import os, yaml

src = os.environ["BASE_CONFIG"]
dst = os.environ["TMP_CONFIG"]
result_file = os.environ["RESULT_FILE"]
target_dir = os.environ["TARGET_DIR"]

with open(src) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("output", {})
cfg["output"]["results_file"] = result_file
cfg["output"]["results_dir"] = target_dir
cfg["output"]["auto_filename"] = False

m = cfg.setdefault("metrics", {})
mode = os.environ.get("PROMPT_MODE", "default")
if mode == "finetune":
    prompt = os.environ.get("FINETUNE_SYSTEM_PROMPT", "").strip()
    if not prompt:
        prompt_file = os.environ.get(
            "FINETUNE_SYSTEM_PROMPT_FILE",
            os.path.join(os.path.dirname(src), "finetune_system_prompt.txt"),
        )
        with open(prompt_file) as pf:
            prompt = pf.read().strip()
    m["system_prompt_type"] = "custom"
    m["custom_system_prompt"] = prompt
elif mode == "null":
    m["system_prompt_type"] = None
    m["custom_system_prompt"] = None
# else: "default" — leave whatever the base config specifies

if os.environ.get("EVAL_MAX_CONCURRENT"):
    m["max_concurrent"] = int(os.environ["EVAL_MAX_CONCURRENT"])
if os.environ.get("EVAL_EXEC_MAX_CONCURRENT"):
    m["execution_max_concurrent"] = int(os.environ["EVAL_EXEC_MAX_CONCURRENT"])

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    local merge_status=$?
    if [[ "${merge_status}" -ne 0 ]]; then
        echo "[eval_review] config merge failed for ${slug}" >&2
        rm -f "${tmp_config}"
        return "${merge_status}"
    fi

    "${PYTHON}" -m evaluate.cli run --config "${tmp_config}"
    local status=$?
    rm -f "${tmp_config}"
    return "${status}"
}

overall_status=0

# [1] Qiskit HumanEval Pass@1
if [[ "${RUN_QHE}" == "true" ]]; then
    log_header "[1] QHE Pass@1 — ${MODEL_NAME}"
    run_config "${CONFIG_DIR}/qiskit_humaneval_normal.yaml" "${QHE_RESULTS_DIR}" "qhe" || overall_status=$?
fi

# [2] Qiskit HumanEval Pass@k (R1.4)
if [[ "${RUN_PASSK}" == "true" ]]; then
    log_header "[2] QHE Pass@k (1,5,10) — ${MODEL_NAME}"
    run_config "${CONFIG_DIR}/qiskit_humaneval_passk.yaml" "${QHE_PASSK_RESULTS_DIR}" "qhe_passk" || overall_status=$?
fi

# [3] Qiskit HumanEval Hard Pass@1
if [[ "${RUN_QHE_HARD}" == "true" ]]; then
    log_header "[3] QHE-Hard Pass@1 — ${MODEL_NAME}"
    run_config "${CONFIG_DIR}/qiskit_humaneval_hard.yaml" "${QHE_HARD_RESULTS_DIR}" "qhe_hard" || overall_status=$?
fi

# [4] Qiskit HumanEval Hard Pass@k (R1.4)
if [[ "${RUN_PASSK_HARD}" == "true" ]]; then
    log_header "[4] QHE-Hard Pass@k (1,5,10) — ${MODEL_NAME}"
    run_config "${CONFIG_DIR}/qiskit_humaneval_hard_passk.yaml" "${QHE_HARD_PASSK_RESULTS_DIR}" "qhe_hard_passk" || overall_status=$?
fi

# [5] Synthetic test split
if [[ "${RUN_SYNTH}" == "true" ]]; then
    if [[ "${IS_VLM}" == "true" ]]; then
        log_header "[5] Synthetic test — full set (VLM) — ${MODEL_NAME}"
        run_config "${CONFIG_DIR}/synthetic.yaml" "${SYNTH_RESULTS_DIR}" "synth" || overall_status=$?
    else
        log_header "[5] Synthetic test — text-only subset (LLM) — ${MODEL_NAME}"
        run_config "${CONFIG_DIR}/synthetic_text_only.yaml" "${SYNTH_RESULTS_DIR}" "synth_text" || overall_status=$?
    fi
fi

log_header "Driver finished for ${MODEL_NAME} (exit=${overall_status})"
exit "${overall_status}"
