# Quantum Assistant

Specializing multimodal vision-language models for quantum computing with Qiskit through synthetic data generation, efficient fine-tuning, and evaluation.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Quantum%20Assistant-yellow)](https://huggingface.co/datasets/samuellimabraz/quantum-assistant)
[![Models](https://img.shields.io/badge/🤗%20Models-Collection-orange)](https://huggingface.co/collections/samuellimabraz/quantum-assistant)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0-6929C4)](https://qiskit.org/)

## Overview

This project addresses a fundamental limitation in quantum computing code assistants: they process only text, ignoring the visual representations that are central to the field—quantum circuit diagrams, Bloch spheres, and measurement histograms. We present:

1. **Synthetic Dataset Pipeline**: Automated extraction and generation of multimodal training data from open-source Qiskit documentation, with executable code verification
2. **Quantum Assistant Dataset**: 8,366 samples (45% multimodal) across 7 quantum computing categories, publicly available under Apache 2.0
3. **VLM Specialization**: Fine-tuning with Parameter-Efficient techniques (LoRA and variants) using ms-swift framework
4. **Evaluation**: Assessment on Qiskit HumanEval benchmarks and multimodal tasks

<p align="center">
  <img src="assets/images/synthetic-pipeline.png" alt="Synthetic Data Pipeline" width="95%">
  <br>
  <em>End-to-end pipeline for generating multimodal quantum computing training data</em>
</p>

### Key Results

| Model | QHE | QHE Hard | Syn. FC | Syn. CG | Syn. QA | Text | MM |
|-------|:---:|:--------:|:-------:|:-------:|:-------:|:----:|:--:|
| | Pass@1 | Pass@1 | Pass@1 | Pass@1 | ROUGE-L | Pass@1 | Pass@1 |
| **Fine-tuned** ||||||||
| Qwen3-VL-FT (r32, 2ep) | 43.71 | 28.48 | **56.96** | **44.36** | 38.02 | **45.45** | **63.39** |
| Qwen3-VL-FT (r32, 1ep) | 40.40 | **29.14** | 51.55 | 41.91 | 37.31 | 42.49 | 57.14 |
| Qwen3-VL-FT (r64, 1ep) | 38.41 | 22.52 | 52.84 | 42.89 | **38.24** | 42.66 | 60.71 |
| **Specialized** ||||||||
| Qwen2.5-Coder-14B-Qiskit† | **49.01** | 25.17 | 47.48 | 25.51 | 19.46 | 36.19 | — |
| **Baseline** ||||||||
| Qwen3-VL-8B-Instruct | 32.45 | 11.92 | 38.92 | 25.98 | 20.66 | 30.24 | 37.50 |
| InternVL3.5-8B-MPO | 20.53 | 9.27 | 32.47 | 19.61 | 25.81 | 21.85 | 36.16 |
| Ministral-3-8B-Instruct-2512 | 17.88 | 11.26 | 29.12 | 21.81 | 20.50 | 20.98 | 36.61 |

<sub>**QHE**: Qiskit HumanEval (function completion) · **QHE Hard**: code generation · **Syn. FC/CG/QA**: Synthetic Function Completion/Code Generation/Question Answering · **Text**: text-only samples · **MM**: multimodal samples · †evaluated on text-only samples (55% of synthetic dataset)</sub>

Best configuration: **rsLoRA with rank=32** on Qwen3-VL-8B-Instruct.

<p align="center">
  <img src="assets/images/fig_combined_results.png" alt="Evaluation Results" width="95%">
  <br>
  <em>Evaluation across benchmarks showing fine-tuning gains and multimodal advantage</em>
</p>

## Project Structure

```
quantum-assistant/
├── src/
│   ├── synthetic_data/    # Dataset generation pipeline
│   ├── finetune/          # Fine-tuning data preparation
│   ├── evaluate/          # Evaluation framework
│   └── models/            # LLM/VLM client utilities
├── data/                  # Source documents (documentation, papers)
├── outputs/               # Generated datasets, models, results
├── docs/                  # Detailed documentation
│   ├── synthetic.md       # Pipeline architecture
│   ├── models.md          # Model serving and benchmarking
│   ├── finetune.md        # Fine-tuning experiments
│   └── evaluate.md        # Evaluation methodology
└── tests/                 # Unit tests
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/samuellimabraz/quantum-assistant.git
cd quantum-assistant

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

**Optional GPU dependencies for fine-tuning:**

```bash
uv sync --package finetune --extra gpu
```

### Environment Setup

Create `.env` file in project root:

```bash
# For synthetic data generation (VLM/LLM endpoints)
VISION_MODEL_BASE_URL=https://api.openai.com/v1
VISION_MODEL_API_KEY=sk-...
VISION_MODEL_NAME=gpt-4o

QUESTION_MODEL_BASE_URL=http://localhost:8000/v1
QUESTION_MODEL_API_KEY=your-key
QUESTION_MODEL_NAME=gpt-oss-120b

# For evaluation
MODEL_BASE_URL=http://localhost:8000/v1
API_KEY=your-key  
MODEL_NAME=qwen2.5-coder-14b

# For HuggingFace uploads
HF_TOKEN=hf_...
```

## Usage

### 1. Generate Synthetic Dataset

```bash
# Run complete pipeline
synthetic-data pipeline --config src/synthetic_data/yaml/config.yaml

# Or individual stages
synthetic-data parse --config src/synthetic_data/yaml/config.yaml
synthetic-data transcribe --config src/synthetic_data/yaml/config.yaml
synthetic-data chunk --config src/synthetic_data/yaml/config.yaml
synthetic-data generate --config src/synthetic_data/yaml/config.yaml
synthetic-data build --config src/synthetic_data/yaml/config.yaml
synthetic-data export --config src/synthetic_data/yaml/config.yaml --hub-id username/dataset
```

See [synthetic_data/README.md](src/synthetic_data/README.md) for detailed usage and [docs/synthetic.md](docs/synthetic.md) for architecture.

### 2. Prepare Data for Fine-tuning

```bash
# From HuggingFace Hub (recommended for Colab)
finetune prepare --hub-id samuellimabraz/quantum-assistant --output-dir outputs/finetune

# From local dataset
finetune prepare --dataset-path outputs/final --output-dir outputs/finetune
```

Outputs ms-swift compatible JSONL format. See [finetune/README.md](src/finetune/README.md) for details.

### 3. Fine-tune VLM

```bash
swift sft \
    --model_type qwen3_vl-8b-instruct \
    --dataset outputs/finetune/train.jsonl \
    --val_dataset outputs/finetune/validation.jsonl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_rslora true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --lora_dropout 0.10 \
    --output_dir outputs/models/qwen3-vl-quantum
```

Configuration based on experiments detailed in [docs/finetune.md](docs/finetune.md).

### 4. Evaluate Models

```bash
# Qiskit HumanEval benchmark
evaluate qiskit-humaneval \
    --dataset path/to/qiskit_humaneval.json \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --num-samples 10 \
    --k-values "1,5,10"

# Synthetic dataset
evaluate synthetic \
    --dataset outputs/final \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-vl-quantum \
    --vlm \
    --split test
```

See [evaluate/README.md](src/evaluate/README.md) for evaluation framework and [docs/evaluate.md](docs/evaluate.md) for methodology.

## Dataset

The **Quantum Assistant Dataset** is publicly available on HuggingFace:

🤗 [samuellimabraz/quantum-assistant](https://huggingface.co/datasets/samuellimabraz/quantum-assistant)

<p align="center">
  <img src="assets/images/fig_overview_dashboard.png" alt="Dataset Overview" width="95%">
  <br>
  <em>Dataset composition: distribution by task type, category, modality, and test coverage</em>
</p>

### Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 8,366 |
| Multimodal Samples | 3,774 (45%) |
| Train / Val / Test | 5,837 / 1,239 / 1,290 |
| Code Samples with Tests | 5,173 (62%) |

### Task Types

- **Function Completion** (30%): Complete function body from signature + docstring
- **Code Generation** (32%): Generate full code from natural language
- **Question Answering** (38%): Conceptual explanations

### Categories

1. `circuits_and_gates` (34%)
2. `quantum_info_and_operators` (20%)
3. `algorithms_and_applications` (17%)
4. `hardware_and_providers` (10%)
5. `transpilation_and_compilation` (8%)
6. `primitives_and_execution` (6%)
7. `noise_and_error_mitigation` (5%)



### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("samuellimabraz/quantum-assistant")

# Filter by type
code_gen = dataset["train"].filter(lambda x: x["type"] == "code_generation")

# Filter multimodal only
multimodal = dataset["train"].filter(lambda x: x["image"] is not None)
```

## Models

Fine-tuned models are available in the HuggingFace collection:

🤗 [Quantum Assistant Models](https://huggingface.co/collections/samuellimabraz/quantum-assistant)

| Model | Configuration | HumanEval Pass@1 | Synthetic Pass@1 |
|-------|---------------|------------------|------------------|
| Qwen3-VL-FT (r32, 1ep) | rsLoRA r=32, 1 epoch | 40.40% | 46.61% |
| Qwen3-VL-FT (r32, 2ep) | rsLoRA r=32, 2 epochs | 43.71% | 50.50% |
| Qwen3-VL-FT (r64, 1ep) | rsLoRA r=64, 1 epoch | 38.41% | 47.74% |

Baseline: Qwen3-VL-8B-Instruct (32.45% / 32.29%)

<p align="center">
  <img src="assets/images/fig8_humaneval_combined.png" alt="HumanEval Results" width="85%">
  <br>
  <em>Qiskit HumanEval benchmark results: fine-tuned models outperform baselines by +11-17 pp</em>
</p>

<p align="center">
  <img src="assets/images/fig4_category_heatmap.png" alt="Category Heatmap" width="85%">
  <br>
  <em>Performance heatmap by category showing fine-tuned models vs baselines (Pass@1 %)</em>
</p>

## Documentation

### Usage Guides (READMEs)

- [Synthetic Data Module](src/synthetic_data/README.md) - Dataset generation pipeline
- [Fine-tune Module](src/finetune/README.md) - Data preparation for training
- [Evaluate Module](src/evaluate/README.md) - Evaluation framework

### Technical Documentation (docs/)

- [Synthetic Data Pipeline](docs/synthetic.md) - Architecture, stages, algorithms
- [Fine-tuning Experiments](docs/finetune.md) - PEFT techniques, hyperparameters
- [Evaluation Methodology](docs/evaluate.md) - Metrics, benchmarks, analysis
- [Model Utilities](docs/models.md) - Serving, batching, benchmarking

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{braz2025quantumassistant,
  title={Quantum Assistant: Especialização de Modelos Multimodais para Computação Quântica},
  author={Braz, Samuel Lima and Leite, João Paulo Reus Rodrigues},
  year={2025},
  institution={Universidade Federal de Itajubá (UNIFEI)},
  url={https://github.com/samuellimabraz/quantum-assistant}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgments

- IBM Quantum and Qiskit team for open-source documentation
- UNIFEI (Universidade Federal de Itajubá) for academic support
- Advisor: Prof. João Paulo Reus Rodrigues Leite
- The quantum computing community for educational materials

## Related Work

- [Qiskit HumanEval](https://arxiv.org/abs/2406.14712) - Evaluation benchmark for quantum code generation
- [Qiskit Code Assistant](https://arxiv.org/abs/2405.19495) - IBM's text-only quantum code assistant
- [Qiskit](https://qiskit.org/) - Open-source quantum computing framework
