# Evaluation Module

Evaluate quantum computing code generation models on Qiskit HumanEval benchmarks and synthetic multimodal datasets.

## Overview

This module provides a framework for evaluating LLMs and VLMs on quantum computing tasks:

- **Qiskit HumanEval**: IBM's benchmark with 151 function completion problems
- **Qiskit HumanEval Hard**: 151 full code generation problems from natural language
- **Synthetic Dataset**: Multimodal evaluation with circuits, charts, and Q&A

Supports Pass@k metrics for code and ROUGE-L/BLEU for conceptual questions.

## Installation

```bash
# From project root
pip install -e ".[evaluate]"

# Or with uv
uv sync
```

## Quick Start

### Qiskit HumanEval Evaluation

```bash
evaluate qiskit-humaneval \
    --dataset path/to/qiskit_humaneval.json \
    --model-url http://localhost:8000/v1 \
    --model-name qwen2.5-coder-14b \
    --num-samples 10 \
    --k-values "1,5,10"
```

### Qiskit HumanEval Hard

```bash
evaluate qiskit-humaneval \
    --dataset path/to/qiskit_humaneval_hard.json \
    --dataset-type hard \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --num-samples 10
```

### Synthetic Multimodal Dataset

```bash
evaluate synthetic \
    --dataset outputs/final \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --split test
```

## Supported Benchmarks

### 1. Qiskit HumanEval

**Format**: Function completion (stub with signature + docstring)

**Example:**

```python
# Prompt
from qiskit import QuantumCircuit

def create_ghz_state(n_qubits: int) -> QuantumCircuit:
    """Create a GHZ state on n qubits."""
    pass

# Model completes the function body
# Tests verify correctness
```

**Metrics**: Pass@k (k=1,5,10)

### 2. Qiskit HumanEval Hard

**Format**: Full code generation from natural language

**Example:**

```
Create a 3-qubit GHZ state and return the circuit.
You must implement this using a function named `create_ghz` with no arguments.
```

**Metrics**: Pass@k (k=1,5,10)

### 3. Synthetic Dataset

**Formats**: Function completion, code generation, and question answering

**Question Types:**

| Type | Evaluation | Metric |
|------|------------|--------|
| `function_completion` | Unit test execution | Pass@k |
| `code_generation` | Unit test execution | Pass@k |
| `qa` | Text similarity | ROUGE-L, BLEU |

**Multimodal Support**: Handles samples with circuit diagrams, charts, Bloch spheres

## Configuration

### Using YAML (Recommended)

Create `eval_config.yaml`:

```yaml
model:
  base_url: "${MODEL_BASE_URL}"
  api_key: "${API_KEY}"
  model_name: "qwen2.5-coder-14b"
  is_vlm: false

dataset:
  type: "qiskit_humaneval"
  path: "path/to/dataset.json"
  dataset_variant: "normal"  # or "hard"

metrics:
  num_samples_per_task: 10
  k_values: [1, 5, 10]
  system_prompt_type: "qiskit_humaneval"  # See system prompts below

output:
  results_dir: "outputs/evaluate"
  auto_filename: true
```

Run with:

```bash
evaluate run --config eval_config.yaml
```

### Environment Variables

Set in `.env`:

```bash
MODEL_BASE_URL=http://localhost:8000/v1
API_KEY=your-key
MODEL_NAME=qwen2.5-coder-14b
```

## System Prompts

Different prompts for fair comparison:

| Type | Description | Use Case |
|------|-------------|----------|
| `qiskit_humaneval` | Full IBM prompt with best practices | Fair comparison with IBM results |
| `qiskit_humaneval_minimal` | Concise Qiskit prompt | Baseline testing |
| `generic` | Generic code assistant | Non-specialized models |
| `null` | No system prompt | Zero-shot evaluation |

**Example prompts:**

```python
# qiskit_humaneval (full IBM prompt)
"""You are an expert in quantum computing and Qiskit framework.
Generate precise, well-documented Qiskit code following these guidelines:
1. Use Qiskit 2.0 APIs exclusively
2. Prefer primitives (SamplerV2, EstimatorV2) over legacy execute()
3. Include proper imports
..."""

# qiskit_humaneval_minimal
"""You are a quantum computing expert specializing in Qiskit.
Provide accurate code using Qiskit 2.0 best practices."""
```

See [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md) for full prompts.

## Metrics

### Code Metrics

**Pass@k**: Probability that at least one of k solutions passes all tests.

$$
\text{Pass@k} = \underset{\text{problems}}{\mathbb{E}} \left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

where:
- $n$ = total solutions generated per problem
- $c$ = number of correct solutions

**Execution Accuracy**: Percentage of code that executes without errors (ignoring test results).

### Text Metrics (for QA)

**ROUGE-L**: Longest common subsequence F1 score between generated and reference answers.

**BLEU**: N-gram overlap score (1 to 4-grams).

**Exact Match**: String equality (case-sensitive).

## Python API

### Qiskit HumanEval

```python
from evaluate.runners import QiskitHumanEvalRunner
from evaluate.config.system_prompts import get_system_prompt
from models import LLMClient

# Setup client
client = LLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen2.5-coder-14b"
)

# Create runner
runner = QiskitHumanEvalRunner(
    dataset_path="data/qiskit_humaneval.json",
    model_client=client,
    k_values=[1, 5, 10],
    num_samples_per_task=10
)

# Load and evaluate
samples = runner.load_dataset()
results = runner.evaluate(
    samples,
    system_prompt=get_system_prompt("qiskit_humaneval"),
    save_results="results/qwen_humaneval.json"
)

print(f"Pass@1: {results.metrics['pass@1']:.2%}")
print(f"Pass@5: {results.metrics['pass@5']:.2%}")
print(f"Pass@10: {results.metrics['pass@10']:.2%}")
```

### Synthetic Dataset

```python
from evaluate.runners import SyntheticDatasetRunner
from models import VLMClient

client = VLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen3-vl-quantum"
)

runner = SyntheticDatasetRunner(
    dataset_path="outputs/final",
    model_client=client,
    images_dir="outputs/images",
    k_values=[1],
    num_samples_per_task=1
)

samples = runner.load_dataset(split="test")
results = runner.evaluate(
    samples,
    system_prompt=get_system_prompt("qiskit_humaneval_minimal"),
    save_results="results/qwen_synthetic.json"
)

# By type
print(f"Function Completion Pass@1: {results.metrics['by_type.function_completion']['pass@1']:.2%}")
print(f"Code Generation Pass@1: {results.metrics['by_type.code_generation']['pass@1']:.2%}")
print(f"QA ROUGE-L: {results.metrics['by_type.qa']['rouge_l']:.2%}")

# By modality
print(f"Text-only Pass@1: {results.metrics['by_modality.text']['pass@1']:.2%}")
print(f"Multimodal Pass@1: {results.metrics['by_modality.multimodal']['pass@1']:.2%}")
```

## Results Format

Results are saved as JSON with metadata:

```json
{
  "metadata": {
    "run_info": {
      "timestamp": "2025-12-11T14:30:22",
      "duration_seconds": 1847
    },
    "model": {
      "name": "qwen3-vl-quantum",
      "type": "vlm",
      "base_url": "http://localhost:8000/v1"
    },
    "dataset": {
      "type": "synthetic",
      "path": "outputs/final",
      "split": "test",
      "num_samples": 1290
    },
    "evaluation": {
      "solutions_per_task": 1,
      "k_values": [1],
      "timeout": 60,
      "system_prompt_type": "qiskit_humaneval_minimal"
    }
  },
  "metrics": {
    "overall": {
      "pass@1": 0.5050,
      "rouge_l": 0.3802
    },
    "by_type.function_completion": {
      "count": 388,
      "pass@1": 0.5696
    },
    "by_type.code_generation": {
      "count": 408,
      "pass@1": 0.4436
    },
    "by_type.qa": {
      "count": 494,
      "rouge_l": 0.3802,
      "bleu": 0.2004
    },
    "by_category.circuits_and_gates": {
      "count": 436,
      "pass@1": 0.6147
    }
  },
  "results": [...]
}
```

## CLI Commands

### Run Evaluation

```bash
# From config file
evaluate run --config config.yaml

# Direct CLI arguments
evaluate qiskit-humaneval \
    --dataset data/qiskit_humaneval.json \
    --model-url http://localhost:8000/v1 \
    --model-name qwen2.5-coder-14b \
    --num-samples 10 \
    --k-values "1,5,10" \
    --system-prompt qiskit \
    --output results/qwen_humaneval.json
```

### Compare Results

```bash
# Compare all results in directory
evaluate compare --results-dir outputs/evaluate

# Generates comparison table and plots
```

### Verify Canonical Solutions

Ensure benchmark correctness by testing canonical solutions:

```bash
evaluate verify-canonical \
    --dataset data/qiskit_humaneval.json \
    --dataset-type normal

# Should show 100% pass rate
```

## Code Execution Sandbox

Generated code is executed in an isolated subprocess:

- **Timeout**: Configurable (default 60s)
- **Safe execution**: Limited syscalls, no network access
- **Test validation**: Automatic check against unit tests
- **Error reporting**: Captures stack traces for debugging

Configuration:

```yaml
metrics:
  execution_timeout: 60        # Seconds
  max_concurrent_executions: 4 # Parallel test runs
```

## Model Serving with vLLM

For efficient inference, use vLLM:

```bash
# Start vLLM server
vllm serve qwen2.5-coder-14b \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# Evaluate
evaluate qiskit-humaneval \
    --model-url http://localhost:8000/v1 \
    --model-name qwen2.5-coder-14b \
    --dataset data/qiskit_humaneval.json
```

For VLMs (multimodal):

```bash
vllm serve qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 4  # Lower for VLMs
```

## Example: Full Evaluation Pipeline

```bash
#!/bin/bash

# 1. Start vLLM server (in separate terminal)
vllm serve qwen/Qwen3-VL-8B-Instruct --port 8000 &

# Wait for server to start
sleep 30

# 2. Evaluate on Qiskit HumanEval
evaluate qiskit-humaneval \
    --dataset benchmarks/qiskit_humaneval.json \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --num-samples 1 \
    --k-values "1" \
    --output results/humaneval.json

# 3. Evaluate on Qiskit HumanEval Hard
evaluate qiskit-humaneval \
    --dataset benchmarks/qiskit_humaneval_hard.json \
    --dataset-type hard \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --num-samples 1 \
    --output results/humaneval_hard.json

# 4. Evaluate on synthetic dataset
evaluate synthetic \
    --dataset outputs/final \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --split test \
    --output results/synthetic.json

# 5. Compare results
evaluate compare --results-dir results/

# Stop vLLM server
pkill -f vllm
```

## Performance Tips

### Batch Processing

Enable batch processing for faster evaluation:

```python
runner = QiskitHumanEvalRunner(
    ...,
    batch_size=10,  # Process 10 samples at once
)
```

### Caching

Results are cached by default. To force re-evaluation:

```bash
evaluate qiskit-humaneval ... --no-cache
```

### Parallel Execution

For code execution, adjust concurrency:

```yaml
metrics:
  max_concurrent_executions: 8  # Increase for faster test runs
```

## Troubleshooting

### Timeout Errors

Increase execution timeout for complex code:

```yaml
metrics:
  execution_timeout: 120  # 2 minutes
```

### Connection Errors

Check vLLM server is running:

```bash
curl http://localhost:8000/v1/models
```

### Memory Issues

For large evaluations, process in batches:

```bash
# Evaluate first 50 samples
evaluate qiskit-humaneval ... --max-samples 50
```

## Documentation

- **Evaluation Methodology**: See [docs/evaluate.md](../../docs/evaluate.md) for detailed analysis of results, metrics interpretation, and comparative studies
- **System Prompts**: See [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md) for complete prompt templates

## References

- [Qiskit HumanEval Paper](https://arxiv.org/abs/2406.14712) - Vishwakarma et al., 2024
- [HumanEval Paper](https://arxiv.org/abs/2107.03374) - Chen et al., 2021
- [vLLM](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023

## License

Apache 2.0
