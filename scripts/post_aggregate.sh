#!/usr/bin/env bash
# =============================================================================
# Post-experiment aggregation for the ESWA revision:
#   1. R2.7 — QHE-Hard stratification per model.
#   2. R1.1 (redesigned) — per-image-type breakdown on synthetic MM subset
#      and base-vs-FT delta on the same subset.
#   3. R1.2 — audit sample extraction (stratified; manual review follows).
# =============================================================================

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${QHE_HARD_PATH:?QHE_HARD_PATH required}"
: "${SYNTH_DATASET_PATH:?SYNTH_DATASET_PATH required}"

STRATIFIED_DIR="${PROJECT_ROOT}/outputs/evaluate/stratified"
IMG_TYPE_DIR="${PROJECT_ROOT}/outputs/evaluate/image_type"
mkdir -p "${STRATIFIED_DIR}" "${IMG_TYPE_DIR}"

# All models whose QHE-Hard results may exist (paper-stack r32-2/base +
# revision-stack additions). Missing ones are skipped silently.
ALL_MODELS=(
    "qwen3-vl-base"
    "qwen3-vl-ft-r32-1ep"
    "qwen3-vl-ft-r32-2"
    "qwen3-vl-ft-r32-2ep"
    "qwen3-vl-ft-r64-1ep"
    "granite-3.3-8b-qiskit"
    "qwen2.5-coder-14b-qiskit"
)
VLM_MODELS=(
    "qwen3-vl-base"
    "qwen3-vl-ft-r32-2"
    "qwen3-vl-ft-r32-1ep"
    "qwen3-vl-ft-r32-2ep"
    "qwen3-vl-ft-r64-1ep"
)

# 1. QHE-Hard stratification per model (R2.7)
echo "[post] stratifying QHE-Hard per model"
for m in "${ALL_MODELS[@]}"; do
    pattern="${PROJECT_ROOT}/outputs/evaluate/qiskit-humaneval-hard/${m}/${m}_qhe_hard_*.json"
    alt_pattern="${PROJECT_ROOT}/outputs/evaluate/qiskit-humaneval-hard/${m}_n1_k1_*.json"
    src=""
    if compgen -G "${pattern}" > /dev/null; then
        src="${pattern}"
    elif compgen -G "${alt_pattern}" > /dev/null; then
        src="${alt_pattern}"
    else
        echo "  skip ${m} (no QHE-Hard results)"
        continue
    fi
    ./.venv/bin/python scripts/stratify_qhe_hard.py \
        --results "${src}" \
        --dataset "${QHE_HARD_PATH}" \
        --out "${STRATIFIED_DIR}/${m}.json" \
    || echo "  stratify failed for ${m}"
done

# 2. Per-image-type breakdown on synthetic MM subset (R1.1 redesigned)
echo "[post] per-image-type breakdown on synthetic MM subset"
for m in "${VLM_MODELS[@]}"; do
    pattern_a="${PROJECT_ROOT}/outputs/evaluate/synthetic/${m}/${m}_synth_*.json"
    pattern_b="${PROJECT_ROOT}/outputs/evaluate/synthetic/${m}_n1_k1_*.json"
    src=""
    if compgen -G "${pattern_a}" > /dev/null; then
        src="$(ls -1t ${pattern_a} | head -1)"
    elif compgen -G "${pattern_b}" > /dev/null; then
        src="$(ls -1t ${pattern_b} | head -1)"
    else
        echo "  skip ${m} (no synthetic results)"
        continue
    fi
    ./.venv/bin/python scripts/per_image_type_breakdown.py \
        --results "${src}" \
        --dataset "${SYNTH_DATASET_PATH}" \
        --model   "${m}" \
        --out     "${IMG_TYPE_DIR}/${m}.json" \
    || echo "  image_type breakdown failed for ${m}"
done

# Aggregate image-type summary across models (incl. within-MM base-vs-FT delta).
./.venv/bin/python - <<'PY'
import json, pathlib
root = pathlib.Path("outputs/evaluate/image_type")
summary = {}
for p in sorted(root.glob("*.json")):
    if p.name == "summary.json":
        continue
    with open(p) as f:
        data = json.load(f)
    summary[data.get("model", p.stem)] = {
        "multimodal_samples": data.get("multimodal_samples"),
        "overall_mm_pass@1": data.get("overall_mm_pass@1"),
        "by_image_type": data.get("by_image_type"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", root / "summary.json")
PY

# 3. Audit set extraction (R1.2) — no GPU, author reviews manually.
echo "[post] extracting audit set"
./.venv/bin/python scripts/sample_audit_set.py \
    --dataset "${SYNTH_DATASET_PATH}" \
    --split test \
    --per-cell 25 \
    --seed 42 \
    --out /workspace/paper/review_data/audit_set.jsonl \
    --images-dir /workspace/paper/review_data/audit_images \
|| echo "  audit extraction failed"

echo "[post] done"
