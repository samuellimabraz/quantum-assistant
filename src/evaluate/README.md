"""
Evaluation Module for Quantum Computing Models
===============================================

A evaluation framework for assessing quantum computing code generation
and multimodal understanding models.

## Architecture

The evaluation module follows clean OOP principles with the following structure:

```
evaluate/
├── evaluators/       # Evaluation logic for different task types
├── metrics/          # Metric implementations (Pass@k, BLEU, ROUGE, etc.)
├── runners/          # High-level runners for specific benchmarks
├── execution/        # Safe code execution sandbox
└── cli.py           # Command-line interface
```

## Key Components

### Evaluators

- **CodeEvaluator**: Evaluates code generation with Pass@k metrics
- **MultimodalEvaluator**: Evaluates multimodal samples with various question types

### Metrics

**Code Metrics:**
- Pass@k: Unbiased estimator following HumanEval methodology
- Execution Accuracy: Percentage of code that executes successfully

**Text Metrics:**
- Exact Match: String equality
- BLEU: N-gram overlap score
- ROUGE-L: Longest common subsequence F1

### Runners

- **QiskitHumanEvalRunner**: Evaluates models on the Qiskit HumanEval benchmark
- **SyntheticDatasetRunner**: Evaluates on synthetic multimodal dataset

## Usage

### YAML Configuration (Recommended)

Create a YAML configuration file (see `config/eval_config.yaml` for examples):

```yaml
model:
  base_url: "${MODEL_BASE_URL}"
  api_key: "${API_KEY}"
  model_name: "qwen2.5-coder-14b"
  is_vlm: false

dataset:
  type: "qiskit_humaneval"
  path: "qiskit-human-eval/dataset/dataset_qiskit_test_human_eval.json"
  dataset_type: "normal"  # "normal" for completion, "hard" for full generation
  max_samples: null

metrics:
  num_samples_per_task: 10
  k_values: [1, 5, 10]
  execution_timeout: 30
  system_prompt_type: "qiskit_humaneval"  # Use IBM Qiskit prompt for fair comparison
  custom_system_prompt: null
  verify_canonical: false  # Verify canonical solutions first

output:
  results_file: "results/qiskit_eval.json"
```

**System Prompt Configuration:**

For fair comparison with IBM's Qiskit HumanEval results, use the full Qiskit system prompt:

```yaml
metrics:
  system_prompt_type: "qiskit_humaneval"  # Full prompt (recommended)
  # Other options: "qiskit_humaneval_minimal", "generic", "custom", or null
```

See [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md) for detailed information on available prompts.

Then run:

```bash
# Set environment variables in .env file or export them
export MODEL_BASE_URL=http://localhost:8000/v1
export API_KEY=your-key-here

# Run evaluation
python -m evaluate.cli run --config config/eval_config.yaml
```

### System Prompts for Fair Comparison

The evaluation framework supports configurable system prompts to ensure fair comparison across models. IBM's Qwen2.5-Coder-14B-Qiskit model uses a detailed system prompt with Qiskit 2.0 best practices.

**Available Prompts:**

1. **`qiskit_humaneval`** (recommended): Full Qiskit assistant prompt matching IBM's approach
   - Comprehensive Qiskit 2.0 guidelines
   - Deprecated method warnings
   - Error mitigation/suppression techniques
   - Best practices for PassManagers and Primitives

2. **`qiskit_humaneval_minimal`**: Minimal prompt for base models

3. **`generic`**: Generic code assistant (no Qiskit-specific)

4. **`custom`**: Your own custom prompt

5. **`null`**: No system prompt (zero-shot)

**Configuration:**

```yaml
metrics:
  system_prompt_type: "qiskit_humaneval"  # For fair comparison with IBM results
  custom_system_prompt: null
```

**CLI Usage:**

```bash
# Full Qiskit prompt
python -m evaluate.cli qiskit-humaneval \
    --dataset dataset.json \
    --model-url http://localhost:8000/v1 \
    --system-prompt qiskit

# Minimal prompt
python -m evaluate.cli qiskit-humaneval \
    --dataset dataset.json \
    --model-url http://localhost:8000/v1 \
    --system-prompt minimal

# Custom prompt
python -m evaluate.cli qiskit-humaneval \
    --dataset dataset.json \
    --model-url http://localhost:8000/v1 \
    --system-prompt "Your custom prompt..."
```

For detailed information, see [SYSTEM_PROMPTS.md](SYSTEM_PROMPTS.md).

### Command Line (Legacy)

**Qiskit HumanEval Evaluation:**

```bash
python -m evaluate.cli qiskit-humaneval \\
    --dataset /path/to/qiskit_humaneval.json \\
    --model-url http://localhost:8000/v1 \\
    --model-name qwen \\
    --num-samples 10 \\
    --k-values 1,5,10 \\
    --output results/qiskit_eval.json
```

**Synthetic Dataset Evaluation:**

```bash
python -m evaluate.cli synthetic \\
    --test-split outputs/splits/test.pkl \\
    --model-url http://localhost:8000/v1 \\
    --model-name qwen \\
    --images-dir outputs/images \\
    --vlm \\
    --output results/synthetic_eval.json
```

**Compare Results:**

```bash
python -m evaluate.cli compare --results-dir results/
```

**Verify Canonical Solutions:**

Before running evaluations, verify that the evaluation setup works correctly:

```bash
# Verify normal dataset (code completion)
python -m evaluate.cli verify-canonical \\
    --dataset /path/to/qiskit_humaneval.json \\
    --dataset-type normal

# Verify hard dataset (full code generation)
python -m evaluate.cli verify-canonical \\
    --dataset /path/to/qiskit_humaneval_hard.json \\
    --dataset-type hard
```

### Python API

**Using YAML Configuration:**

```python
from evaluate.config.schema import EvaluationConfig

# Load configuration
config = EvaluationConfig.from_yaml("config/eval_config.yaml")

# Modify if needed
config.dataset.max_samples = 50

# Run evaluation directly from config
from evaluate.cli import _run_evaluation_from_config
results = _run_evaluation_from_config(config)
```

**Qiskit HumanEval:**

```python
from evaluate.runners import QiskitHumanEvalRunner
from evaluate.config.system_prompts import get_system_prompt
from models.client import LLMClient

# Create client
client = LLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen"
)

# Get system prompt for fair comparison with IBM results
system_prompt = get_system_prompt("qiskit_humaneval")

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
    system_prompt=system_prompt,
    save_results="results.json"
)

print(f"Pass@1: {results.metrics['pass@1']:.4f}")
print(f"Pass@5: {results.metrics['pass@5']:.4f}")
print(f"Pass@10: {results.metrics['pass@10']:.4f}")
```

**Synthetic Dataset:**

```python
from evaluate.runners import SyntheticDatasetRunner
from models.client import VLMClient

# Create VLM client
client = VLMClient(
    base_url="http://localhost:8000/v1",
    model_name="qwen2-vl"
)

# Create runner
runner = SyntheticDatasetRunner(
    test_split_path="outputs/splits/test.pkl",
    model_client=client,
    images_dir="outputs/images"
)

# Evaluate
results = runner.evaluate(num_predictions=1, save_results="results.json")

print(f"Success Rate: {results.success_rate:.1%}")
for metric, value in results.metrics["overall"].items():
    print(f"{metric}: {value:.4f}")
```

## Configuration

### YAML Configuration Files

Configuration files are stored in `config/`:

- `eval_config.yaml` - Qiskit HumanEval evaluation
- `eval_synthetic.yaml` - Synthetic multimodal dataset evaluation
- `eval.env.example` - Environment variables template

### Environment Variables

The evaluation module uses the project's main `.env` file located at the project root.

Add these variables to your `.env` file:

```bash
# Evaluation Model Configuration
MODEL_BASE_URL=https://api.deepinfra.com/v1
API_KEY=your-api-key-here
MODEL_NAME=openai/gpt-oss-120b
```

These are automatically loaded and resolved in YAML files using `${VAR_NAME}` syntax.

### Configuration Schema

The configuration uses Pydantic models with validation:

**ModelConfig:**
- `base_url`: Model API endpoint
- `api_key`: API key (supports env vars)
- `model_name`: Model identifier
- `is_vlm`: Whether it's a vision-language model
- `max_tokens`, `temperature`, `timeout`: Generation parameters

**DatasetConfig:**
- `type`: "qiskit_humaneval" or "synthetic"
- `path`: Path to dataset file
- `dataset_type`: "normal" (completion) or "hard" (full generation)
- `images_dir`: Directory for images (synthetic only)
- `max_samples`: Limit for testing

**MetricsConfig:**
- `num_samples_per_task`: Solutions per task (Pass@k)
- `k_values`: K values for Pass@k computation
- `execution_timeout`: Code execution timeout
- `num_predictions`: Predictions per sample
- `system_prompt_type`: System prompt type ("qiskit_humaneval", "qiskit_humaneval_minimal", "generic", "custom", or null)
- `custom_system_prompt`: Custom prompt text (when system_prompt_type is "custom")
- `verify_canonical`: Whether to verify canonical solutions first

**OutputConfig:**
- `results_file`: Where to save results JSON
- `results_dir`: Results directory

## Dataset Types

The Qiskit HumanEval benchmark has two dataset types:

### Normal (Code Completion)
The prompt contains initial code with imports and function signatures.
The model completes the function body. The final executable code is:
`prompt + generated_completion`

### Hard (Full Code Generation)
The prompt is a natural language question describing the task.
The model generates the complete code from scratch. The final executable code
is just the generated code.

Use `--dataset-type normal` or `--dataset-type hard` to specify the type,
or set `dataset_type` in the YAML configuration.

## Metrics Explanation

### Pass@k

Pass@k measures the probability that at least one of k generated code samples
passes all test cases. We use the unbiased estimator from the HumanEval paper:

```
pass@k = 1 - (n-c choose k) / (n choose k)
```

where:
- n = total samples generated
- c = number of correct samples
- k = number of samples to consider

### BLEU

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
predicted and reference texts. Useful for caption and summary evaluation.

### ROUGE-L

ROUGE-L measures the longest common subsequence between predicted and
reference texts. Better for semantic similarity than BLEU.

## Code Execution Sandbox

The code executor runs generated code in an isolated subprocess with:
- Configurable timeout (default: 30s)
- Safe execution environment
- Automatic test case validation
- Detailed error reporting

## Results Organization

Results are automatically organized into a structured directory:

```
outputs/evaluate/
├── qiskit-humaneval/
│   ├── qwen2.5-coder-14b_n10_k1-5-10_20241125_143022.json
│   └── gpt-oss-120b_n10_k1-5-10_20241125_150000.json
├── qiskit-humaneval-hard/
│   └── ...
└── synthetic/
    └── ...
```

Filenames include: `{model}_{n-samples}_{k-values}_{timestamp}.json`

## Evaluation Results Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "run_info": {
      "timestamp": "2024-11-25T14:30:22",
      "timestamp_unix": 1732545022
    },
    "model": {
      "name": "qwen2.5-coder-14b"
    },
    "dataset": {
      "path": "path/to/dataset.json",
      "type": "qiskit_humaneval",
      "variant": "normal",
      "num_samples": 151
    },
    "evaluation": {
      "solutions_per_task": 10,
      "k_values": [1, 5, 10],
      "timeout": 30,
      "system_prompt": "You are the Qiskit code assistant...",
      "verify_canonical": true
    }
  },
  "metrics": {
    "pass@1": 0.4567,
    "pass@5": 0.7234,
    "pass@10": 0.8456,
    "execution_accuracy": 0.5123
  },
  "canonical_verification": {
    "verified": true,
    "passed": 151,
    "failed": 0
  },
  "results": [
    {
      "task_id": "qiskitHumanEval/0",
      "prompt": "from qiskit import...",
      "canonical_solution": "def solution():...",
      "test": "check(solution)",
      "generated_solutions": [
        {
          "code": "def solution():...",
          "passed": true,
          "error": null
        }
      ],
      "num_passed": 8,
      "pass_rate": 0.8
    }
  ]
}
```

The results include:
- **prompt**: The original input prompt
- **canonical_solution**: The expected correct solution
- **test**: The test code for verification
- **generated_solutions**: Each generated solution with execution results
- **pass_rate**: Percentage of solutions that passed

## Extending the Framework

### Adding a New Metric

```python
from evaluate.metrics.base import Metric

class MyMetric(Metric):
    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, predictions, references, **kwargs) -> float:
        # Implement your metric logic
        return score
```

### Adding a New Evaluator

```python
from evaluate.evaluators.base import Evaluator, EvaluationResult

class MyEvaluator(Evaluator):
    def evaluate_sample(self, sample, predictions):
        # Implement evaluation logic
        return EvaluationResult(...)

    def aggregate_results(self, results):
        # Aggregate individual results
        return AggregatedResults(...)
```

## Testing

Run tests with:

```bash
pytest tests/test_evaluation.py -v
```

## References

- Qiskit HumanEval: https://arxiv.org/abs/2406.14712
- HumanEval: https://arxiv.org/abs/2107.03374
- BLEU: https://aclanthology.org/P02-1040/
- ROUGE: https://aclanthology.org/W04-1013/
"""

