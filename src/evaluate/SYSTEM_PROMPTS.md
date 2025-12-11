# System Prompts for Evaluation

This document describes the system prompt configuration for fair evaluation across different models on the Qiskit HumanEval benchmark and synthetic datasets.

## Background

The IBM Qiskit team's [Qwen2.5-Coder-14B-Qiskit](https://huggingface.co/Qiskit/Qwen2.5-Coder-14B-Qiskit) model uses a detailed system prompt that includes:

1. **Role Definition**: Defines the model as a Qiskit code assistant
2. **Code Generation Guidelines**: Qiskit 2.0 best practices, deprecated methods to avoid
3. **Best Practices**: Error mitigation techniques, proper imports, transpilation methods
4. **Context Awareness**: Professional standards and OOP design patterns

For fair comparison across models in benchmarks, we provide configurable system prompts that match or adapt from IBM's approach.

## Available System Prompts

### 1. `qiskit_humaneval` (Default)
**Purpose**: Full Qiskit code assistant prompt based on IBM's Qwen2.5-Coder-14B-Qiskit

**Use when**: 
- Evaluating on Qiskit HumanEval benchmark
- Comparing against IBM's published results
- Fine-tuned models need detailed guidelines

**Content**:
- Complete role definition as Qiskit code assistant
- Comprehensive Qiskit 2.0 guidelines
- Deprecated method warnings (transpile, execute, assemble)
- Error mitigation techniques (TREX, ZNE, PEA, PEC)
- Error suppression methods (Dynamical decoupling, Pauli Twirling)
- Best practices for PassManagers and Primitives
- Clean code and OOP design patterns

### 2. `qiskit_humaneval_minimal`
**Purpose**: Minimal Qiskit prompt for base models

**Use when**:
- Evaluating base/pre-trained models without fine-tuning
- Quick benchmarks
- Models that don't need extensive context

**Content**:
- Brief role definition
- Core Qiskit 2.0 requirements
- Avoid deprecated methods reminder
- Clean code instruction

### 3. `generic`
**Purpose**: Generic code assistant (no Qiskit-specific)

**Use when**:
- Comparing with non-quantum code models
- General code generation benchmarks

**Content**:
- Basic coding assistant role
- Best practices reminder
- Code completion focus

### 4. `custom`
**Purpose**: User-provided custom prompt

**Use when**:
- Testing specific prompt engineering strategies
- Ablation studies on prompt components
- Domain-specific requirements

### 5. `None` (No system prompt)
**Purpose**: Evaluate without any system message

**Use when**:
- Testing zero-shot capabilities
- Comparing with/without system guidance
- Models fine-tuned with instructions baked in

## Configuration

### YAML Configuration (`eval_config.yaml`)

```yaml
metrics:
  # System prompt configuration
  system_prompt_type: "qiskit_humaneval"  # or "qiskit_humaneval_minimal", "generic", "custom", null
  custom_system_prompt: null  # Set this if system_prompt_type is "custom"
```

**Examples**:

```yaml
# Full Qiskit prompt (recommended for fair comparison with IBM results)
system_prompt_type: "qiskit_humaneval"
custom_system_prompt: null

# Minimal prompt
system_prompt_type: "qiskit_humaneval_minimal"
custom_system_prompt: null

# No system prompt
system_prompt_type: null
custom_system_prompt: null

# Custom prompt
system_prompt_type: "custom"
custom_system_prompt: |
  You are a quantum computing expert specializing in Qiskit.
  Generate clean, well-documented code following Qiskit 2.0 standards.
```

### CLI Usage

#### Using config file:
```bash
python -m evaluate.cli run --config eval_config.yaml
```

#### Legacy CLI with shortcuts:
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
  --system-prompt "You are a Qiskit expert..."

# No system prompt
python -m evaluate.cli qiskit-humaneval \
  --dataset dataset.json \
  --model-url http://localhost:8000/v1
```

### Python API

```python
from evaluate.config.system_prompts import get_system_prompt
from evaluate.runners.qiskit_humaneval import QiskitHumanEvalRunner

# Get predefined prompt
system_prompt = get_system_prompt("qiskit_humaneval")

# Or use custom
system_prompt = "Your custom prompt here"

# Or None for no system prompt
system_prompt = None

runner = QiskitHumanEvalRunner(...)
results = runner.evaluate(
    samples=samples,
    system_prompt=system_prompt,
    ...
)
```

## Fair Comparison Guidelines

For reproducible and fair comparisons:

1. **Always specify the system prompt used** in your evaluation metadata
2. **Use the same prompt across models** when benchmarking
3. **Report the prompt type** in results (automatically saved in metadata)
4. **For comparison with IBM's results**, use `qiskit_humaneval` prompt
5. **Document any modifications** to prompts for transparency

## Prompt Content Reference

The full `qiskit_humaneval` prompt includes guidance on:

### Qiskit 2.0 Best Practices
- Use `generate_preset_pass_manager` instead of `transpile`
- Use `SamplerV2` and `EstimatorV2` instead of `execute`
- Use `qiskit-ibm-runtime` instead of deprecated `qiskit-ibmq-provider`

### Error Mitigation
- TREX (Twirled Readout Error eXtinction)
- ZNE (Zero-Noise Extrapolation)
- PEA (Probabilistic Error Amplification)
- PEC (Probabilistic Error Cancellation)

### Error Suppression
- Dynamical Decoupling
- Pauli Twirling

### Execution Patterns
1. Map problem to quantum circuits
2. Optimize for target hardware
3. Execute on target hardware
4. Post-process results

### Coding Standards
- Write clean, objective code
- Follow OOP design patterns when appropriate
- Avoid unnecessary explanations unless requested
- Use proper imports and Qiskit 2.0 APIs

## Metadata Tracking

All evaluation results automatically include system prompt information:

```json
{
  "metadata": {
    "evaluation": {
      "system_prompt": "You are the Qiskit code assistant...",
      "system_prompt_type": "qiskit_humaneval",
      ...
    }
  }
}
```

This ensures reproducibility and transparency in benchmarking.

## References

- [Qiskit HumanEval Paper](https://arxiv.org/abs/2406.14712)
- [Qwen2.5-Coder-14B-Qiskit on HuggingFace](https://huggingface.co/Qiskit/Qwen2.5-Coder-14B-Qiskit)
- [IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs/en)


