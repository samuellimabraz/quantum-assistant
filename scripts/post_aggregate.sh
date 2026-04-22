#!/usr/bin/env bash
# =============================================================================
# Post-experiment aggregation: stratify QHE-Hard (R2.7), image-masking deltas
# (R1.1), and build the audit sample (R1.2).
# =============================================================================

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${QHE_HARD_PATH:?QHE_HARD_PATH required}"
: "${SYNTH_DATASET_PATH:?SYNTH_DATASET_PATH required}"

STRATIFIED_DIR="${PROJECT_ROOT}/outputs/evaluate/stratified"
MASK_DIR="${PROJECT_ROOT}/outputs/evaluate/image_masking"
mkdir -p "${STRATIFIED_DIR}" "${MASK_DIR}"

ALL_MODELS=(
    "qwen3-vl-base"
    "qwen3-vl-ft-r32-1ep"
    "qwen3-vl-ft-r32-2ep"
    "qwen3-vl-ft-r64-1ep"
    "granite-3.3-8b-qiskit"
    "qwen2.5-coder-14b-qiskit"
)
VLM_MODELS=(
    "qwen3-vl-base"
    "qwen3-vl-ft-r32-1ep"
    "qwen3-vl-ft-r32-2ep"
    "qwen3-vl-ft-r64-1ep"
)

# 1. QHE-Hard stratification per model (R2.7)
# Driver writes files as ${MODEL_NAME}_qhe_hard_*.json under
# outputs/evaluate/qiskit-humaneval-hard/${MODEL_NAME}/
echo "[post] stratifying QHE-Hard per model"
for m in "${ALL_MODELS[@]}"; do
    pattern="${PROJECT_ROOT}/outputs/evaluate/qiskit-humaneval-hard/${m}/${m}_qhe_hard_*.json"
    if ! compgen -G "${pattern}" > /dev/null; then
        echo "  skip ${m} (no QHE-Hard results matching ${pattern})"
        continue
    fi
    ./.venv/bin/python scripts/stratify_qhe_hard.py \
        --results "${pattern}" \
        --dataset "${QHE_HARD_PATH}" \
        --out "${STRATIFIED_DIR}/${m}.json" \
    || echo "  stratify failed for ${m}"
done

# 2. Image-masking deltas (R1.1) — VLM only.
# Driver writes:
#   ${MODEL_NAME}_synth_mm_kept_*.json
#   ${MODEL_NAME}_synth_mm_masked_*.json
echo "[post] computing image-masking deltas"
for m in "${VLM_MODELS[@]}"; do
    synth_dir="${PROJECT_ROOT}/outputs/evaluate/synthetic/${m}"
    with_image="$(ls -1t "${synth_dir}/${m}_synth_mm_kept_"*.json 2>/dev/null | head -1)"
    masked="$(ls -1t "${synth_dir}/${m}_synth_mm_masked_"*.json 2>/dev/null | head -1)"
    if [[ -z "${with_image}" || -z "${masked}" ]]; then
        echo "  skip ${m} (need both mm_kept and mm_masked result files in ${synth_dir})"
        continue
    fi
    ./.venv/bin/python scripts/image_masking_delta.py \
        --with-image "${with_image}" \
        --masked     "${masked}" \
        --model      "${m}" \
        --out        "${MASK_DIR}/${m}.json" \
    || echo "  delta failed for ${m}"
done

# Aggregate image-masking summary across models.
./.venv/bin/python - <<'PY'
import glob, json, pathlib
root = pathlib.Path("outputs/evaluate/image_masking")
summary = {}
for p in sorted(root.glob("*.json")):
    if p.name == "summary.json":
        continue
    with open(p) as f:
        data = json.load(f)
    summary[data.get("model", p.stem)] = {
        "matched_samples": data.get("matched_samples"),
        "overall": data.get("overall"),
        "by_question_type": data.get("by_question_type"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", root / "summary.json")
PY

# 3. Audit set extraction (R1.2) — no GPU
echo "[post] extracting audit set"
./.venv/bin/python scripts/sample_audit_set.py \
    --dataset "${SYNTH_DATASET_PATH}" \
    --split test \
    --per-cell 25 \
    --seed 42 \
    --out /workspace/paper/review_data/audit_set.jsonl \
    --images-dir /workspace/paper/review_data/audit_images

echo "[post] done"
