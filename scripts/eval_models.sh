#!/bin/bash
# =============================================================================
# Model Evaluation Script for Quantum Computing Benchmarks
# =============================================================================
#
# This script evaluates VLMs on three benchmarks:
#   1. Qiskit HumanEval Normal (code completion)
#   2. Qiskit HumanEval Hard (full code generation)
#   3. Synthetic Multimodal Dataset (code + QA with images)
#
# Usage:
#   ./scripts/eval_models.sh
#
# Requirements:
#   - Set MODEL_BASE_URL and API_KEY in .env or environment
#   - Model server running at MODEL_BASE_URL
#
# =============================================================================

set -e  # Exit on error

# Change to project root
cd "$(dirname "$0")/.."

# Load environment variables from .env if exists
if [ -f .env ]; then
    echo "Loading environment from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Configuration paths (relative to src/)
CONFIG_DIR="evaluate/config"
NORMAL_CONFIG="${CONFIG_DIR}/qiskit_humaneval_normal.yaml"
HARD_CONFIG="${CONFIG_DIR}/qiskit_humaneval_hard.yaml"
SYNTHETIC_CONFIG="${CONFIG_DIR}/synthetic.yaml"

# Models to evaluate
# Uncomment/modify the models you want to evaluate
MODELS=(
    "google/gemini-2.5-flash-lite"
    # "OpenGVLab/InternVL2-8B"
    # "microsoft/Phi-3.5-vision-instruct"
    # "Qwen/Qwen2-VL-2B-Instruct"
)

echo "=============================================="
echo "  Quantum Computing Model Evaluation"
echo "=============================================="
echo ""
echo "Base URL: ${MODEL_BASE_URL:-'Not set'}"
echo "Models to evaluate: ${#MODELS[@]}"
echo ""

# Check if MODEL_BASE_URL is set
if [ -z "$MODEL_BASE_URL" ]; then
    echo "Error: MODEL_BASE_URL is not set"
    echo "Please set it in .env or export it:"
    echo "  export MODEL_BASE_URL=http://localhost:8000/v1"
    exit 1
fi

# Enter src directory for module imports
cd src

for MODEL in "${MODELS[@]}"; do
    export MODEL_NAME="$MODEL"
    
    echo ""
    echo "=============================================="
    echo "Evaluating: $MODEL"
    echo "=============================================="
    
    # Qiskit HumanEval Normal (Code Completion)
    echo ""
    echo "[1/3] Qiskit HumanEval Normal (Code Completion)"
    echo "----------------------------------------------"
    uv run python -m evaluate.cli run --config "$NORMAL_CONFIG" || {
        echo "Warning: Normal evaluation failed for $MODEL"
    }
    
    # Qiskit HumanEval Hard (Full Code Generation)
    echo ""
    echo "[2/3] Qiskit HumanEval Hard (Full Generation)"
    echo "----------------------------------------------"
    uv run python -m evaluate.cli run --config "$HARD_CONFIG" || {
        echo "Warning: Hard evaluation failed for $MODEL"
    }
    
    # Synthetic Dataset (Multimodal)
    echo ""
    echo "[3/3] Synthetic Dataset (Multimodal)"
    echo "----------------------------------------------"
    uv run python -m evaluate.cli run --config "$SYNTHETIC_CONFIG" || {
        echo "Warning: Synthetic evaluation failed for $MODEL"
    }
    
    echo ""
    echo "Completed evaluation for: $MODEL"
done

echo ""
echo "=============================================="
echo "  Generating Comparison Report"
echo "=============================================="

# Compare all results
uv run python -m evaluate.cli compare --results-dir ../outputs/evaluate

echo ""
echo "=============================================="
echo "  Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results saved in: outputs/evaluate/"
echo "  - qiskit-humaneval/      (normal benchmark)"
echo "  - qiskit-humaneval-hard/ (hard benchmark)"
echo "  - synthetic/             (multimodal dataset)"
