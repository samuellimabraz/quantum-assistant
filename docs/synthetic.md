# Synthetic Data Generation Pipeline

> High-quality multimodal dataset generation for quantum computing VLM/LLM fine-tuning with Qiskit.

## Overview

This pipeline generates training data from open-source quantum computing documentation (Jupyter notebooks, MDX files, PDFs) with full support for multimodal samples containing circuit diagrams, charts, and visualizations.

### Target Distribution

| Type | Description | Unit Test | Image Support | Target |
|------|-------------|-----------|---------------|--------|
| `function_completion` | Stub (imports + signature + docstring + pass) | ✓ Required | Optional | 35% |
| `code_generation` | Natural language task (HumanEval Hard format) | ✓ Required | Optional | 35% |
| `qa` | Theory/concepts (explanation, summary, analysis) | ✗ Optional | Optional | 30% |

**Multimodal Ratio:** Configurable per type (default: 70% QA, 30% code types)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SYNTHETIC DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   STAGE 1    │───▶│   STAGE 2    │───▶│   STAGE 3    │───▶│   STAGE 4    │  │
│  │    Parse     │    │  Transcribe  │    │ Filter Images│    │    Chunk     │  │
│  │              │    │              │    │              │    │              │  │
│  │ .ipynb/.mdx  │    │  VLM + Auto  │    │ LLM Quality  │    │  Semantic    │  │
│  │ .pdf → Docs  │    │ Classify     │    │   Filter     │    │  Splitting   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │          │
│         ▼                   ▼                   ▼                   ▼          │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Documents with content, code blocks, images → [IMAGE:id] markers        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────────┐   │
│  │   STAGE 5    │    │                    STAGE 6                          │   │
│  │ Filter Chunks│───▶│                   Generate                          │   │
│  │              │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │ LLM Content  │    │  │ 6.1     │ │ 6.2     │ │ 6.3     │ │ 6.4     │   │   │
│  │  Quality     │    │  │ Allocate│▶│ Plan    │▶│ Filter  │▶│ Answer  │   │   │
│  └──────────────┘    │  │         │ │ Inputs  │ │ Cands   │ │ + Test  │   │   │
│                      │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│                      │                                       ┌─────────┐   │   │
│                      │                                       │ 6.5     │   │   │
│                      │                                       │ Curate  │   │   │
│                      │                                       │+Classify│   │   │
│                      │                                       └─────────┘   │   │
│                      └─────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │
│  │   STAGE 7    │───▶│   STAGE 8    │───▶│   STAGE 9    │                      │
│  │    Build     │    │    Export    │    │   Analyze    │                      │
│  │              │    │              │    │              │                      │
│  │  Stratified  │    │  HuggingFace │    │ Statistics + │                      │
│  │   Splits     │    │   Format     │    │    Plots     │                      │
│  └──────────────┘    └──────────────┘    └──────────────┘                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Parse

**Module:** `extractors/ingestion.py`, `parsers/`

Extracts content from source documents and resolves image references.

```
Input:  Source files (.ipynb, .mdx, .pdf)
Output: outputs/parsed/documents.pkl
```

**Operations:**
- Parse Jupyter notebooks preserving cell boundaries and code-output relationships
- Parse MDX files with JSX image tag support (`<Image src="/docs/..." />`)
- Parse PDFs with slide detection and section extraction
- Replace inline images with `[IMAGE:img_xxxxxxxxxxxx]` markers
- Resolve image paths to actual files on disk
- Remove unresolved image markers from content

**Parsers:**
| Parser | Formats | Features |
|--------|---------|----------|
| `JupyterParser` | `.ipynb` | Cell-aware, preserves code-output pairs, extracts embedded images |
| `MDXParser` | `.mdx` | JSX components, frontmatter, Qiskit doc format |
| `PDFParser` | `.pdf` | OCR fallback, slide detection, section headers |
| `TOCParser` | `_toc.yml` | Ordered document loading from table of contents |

---

### Stage 2: Transcribe

**Module:** `extractors/transcriber.py`

Generates detailed VLM descriptions and classifies image types.

```
Input:  outputs/parsed/documents.pkl
Output: outputs/transcribed/documents.pkl
```

**Operations:**
- Batch VLM transcription with semaphore-controlled concurrency
- Context-aware prompting (alt text, caption, surrounding code)
- Automatic image type classification

**Image Types:**
| Type | Description | Use Cases |
|------|-------------|-----------|
| `circuit` | Quantum circuit diagrams | Code generation, function completion |
| `chart` | Histograms, plots, measurement results | QA analysis |
| `bloch_sphere` | Bloch sphere visualizations | Code or QA |
| `formula` | Mathematical equations | Usually filtered |
| `diagram` | Technical/architecture diagrams | QA explanation |
| `table` | Structured data tables | QA interpretation |
| `code_output` | Execution visualizations | QA analysis |
| `decorative` | Logos, icons | Filtered out |

---

### Stage 3: Filter Images

**Module:** `utils/image_filter.py`

LLM-based quality filtering of images before chunking.

```
Input:  outputs/transcribed/documents.pkl
Output: outputs/filtered_images/documents.pkl
```

**Filter Criteria:**
- ✓ Pass: Circuits, charts, Bloch spheres, diagrams with sufficient context
- ✗ Reject: Decorative, uninformative, incomplete transcriptions
- Markers for rejected images are removed from document content

---

### Stage 4: Chunk

**Module:** `extractors/chunker.py`

Splits documents into semantic units with image references and accumulated code context.

```
Input:  outputs/filtered_images/documents.pkl
Output: outputs/chunks/chunks.pkl
```

**Chunking Strategy:**

| Document Type | Strategy |
|--------------|----------|
| Notebooks | Cell-aware: keep code + output together, max N code blocks per chunk |
| MDX | Header-based: split on `#`/`##`, accumulate `###`/`####` |
| PDFs | Slide markers (`---`) or paragraph boundaries |

**Key Features:**
- **Accumulated Code Context:** Each chunk tracks all code blocks from prior cells
- **Image References:** `[IMAGE:id]` markers (not embedded transcriptions)
- **Quality Classification:** HIGH (code+images), MEDIUM (content), LOW (imports-only)
- **Limits Enforcement:** Max code blocks, max images per chunk
- **Neighbor Context:** Previous/next chunk text for generation context

**Chunk Data Structure:**
```python
@dataclass
class Chunk:
    text: str                          # Main content with [IMAGE:id] markers
    source_path: Path                  # Source document
    chunk_id: int                      # Sequential ID
    code_blocks: list[str]             # Code in this chunk
    images: list[ImageReference]       # Image references
    accumulated_code: list[str]        # All prior code blocks
    previous_chunk_text: str           # Neighbor context
    next_chunk_text: str               # Neighbor context
    quality: ChunkQuality              # HIGH/MEDIUM/LOW
```

---

### Stage 5: Filter Chunks

**Module:** `utils/quality.py`

LLM-based filtering of chunk text content quality.

```
Input:  outputs/chunks/chunks.pkl
Output: outputs/filtered/chunks.pkl
```

**Filter Criteria:**
- ✓ Pass: Quantum computing, Qiskit, physics, mathematics content
- ✗ Reject: Trivial Python, generic content, bibliography, TOC

**Note:** Images are NOT re-evaluated at this stage (already filtered in Stage 3).

---

### Stage 6: Generate

**Modules:** `generators/stages/`

Multi-stage sample generation split into separate commands.

```
Input:  outputs/filtered/chunks.pkl
Output: outputs/generated/samples.pkl (via intermediate stages)
```

The generation stage is split into 5 separate sub-commands, each with its own checkpoint and output:

| Sub-Stage | Command | Module | Input | Output |
|-----------|---------|--------|-------|--------|
| 6a | `plan` | `stages/plan.py` | `filtered/chunks.pkl` | `planned/candidates.pkl` |
| 6b | `filter-candidates` | `stages/filter_candidates.py` | `planned/candidates.pkl` | `filtered_candidates/candidates.pkl` |
| 6c | `answer` | `stages/answer.py` | `filtered_candidates/candidates.pkl` | `answered/samples.pkl` |
| 6d | `curate` | `stages/curate.py` | `answered/samples.pkl` | `curated/samples.pkl` |
| 6e | `classify` | `stages/classify.py` | `curated/samples.pkl` | `generated/samples.pkl` |

#### 6a: Plan - Input Planning and Allocation

**Module:** `stages/plan.py`, `allocation.py`

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ALLOCATION ALGORITHM                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Build Candidates                                                    │
│     For each (chunk, image, question_type) combination:                │
│     • Score by content suitability (code presence, text length)        │
│     • Create SampleTask with chunk reference and target image          │
│                                                                         │
│  2. Calculate Targets                                                   │
│     Per (question_type, is_multimodal):                                │
│     • Base target from config ratios                                   │
│     • Over-allocate by factor (default 1.8x)                           │
│                                                                         │
│  3. Diversity-Aware Selection                                          │
│     • Track chunk and image usage counts                               │
│     • Adjust scores: score × (1 - penalty × diversity_weight)          │
│     • Process scarce combinations first                                │
│     • Select top candidates per category                               │
│                                                                         │
│  Metrics:                                                              │
│  • Chunk coverage: % of unique chunks used                             │
│  • Image coverage: % of unique images used                             │
│  • Avg chunk usage: reuse factor                                       │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**
```yaml
generation:
  target_samples: 8000
  over_allocation_factor: 1.8    # Generate 80% more candidates
  diversity_weight: 0.4          # Balance score vs diversity
  max_generation_attempts: 3     # Retry rounds if target not met
  
  type_allocations:
    qa:
      ratio: 0.30                # 30% of samples
      multimodal_ratio: 0.70     # 70% with images
    code_generation:
      ratio: 0.35
      multimodal_ratio: 0.30
    function_completion:
      ratio: 0.35
      multimodal_ratio: 0.30
```

Combines diversity-aware allocation with unified session-based question/test generation.

**Session-Based Generation Flow (per task):**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED SESSION PER TASK                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Generate Initial Question                                        │
│     • Use type-specific prompt (function_completion/code_gen/qa)    │
│     • Build context     │
│                                                                      │
│  2. Refine Question                                │
│     • Check context grounding, format compliance                    │
│     • Check multimodal alignment  │
│     • Ensure self-contained and Qiskit 2.0 compliant               │
│                                                                      │
│  3. Generate Test (code types only)                                 │
│     • Generate test based on context patterns                       │
│     • Validate: syntax, assertions, entry point, deprecated APIs    │
│                                                                      │
│  4. Test Correction Loop (up to 3 attempts)                         │
│     • On validation error: classify error type                      │
│     • Provide error feedback and request correction                 │
│     • Re-validate until pass or max attempts                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Improvements:**
- **Question Refinement:** Catches over-description of images, ensures context grounding
- **Test Correction Loop:** Up to 3 correction attempts with error feedback
- **Unified Session:** Maintains conversation context across all steps per task
- **Context Prioritization:** Code that generated images appears first in context

**Output:** `InputCandidate` objects with question, test_code, entry_point

#### 6b: Filter Candidates

**Module:** `stages/filter_candidates.py`

Optional LLM-based quality filtering of generated inputs.

**Reject Criteria:**
- Trivial or non-quantum tasks
- Question/type format mismatch
- Multimodal questions that don't reference image
- Over-description of image content

**Output:** Filtered candidate list

#### 6c: Answer - Answer Generation with Validation

**Module:** `stages/answer.py`, `sessions.py`

Session-based answer generation with test validation loop.

**Flow:**
1. Build answer sessions for each candidate
2. Generate initial answer
3. For code types: run test validation loop (max iterations)
4. On failure: provide error feedback and request correction
5. Assemble final Sample objects

**Output:** Validated samples with code that passes unit tests

#### 6d: Curate - Quality Curation

**Module:** `stages/curate.py`

Final quality check on generated Q&A pairs.

**Curation Checks:**
- Content correctness (factually accurate)
- Multimodal coherence (image is essential, not decorative)
- Context relevance (answer uses context patterns)
- Formatting standards (no URLs, pleasantries, repetition)

**Output:** Curated samples (rejects saved separately)

#### 6e: Classify - Category Classification

**Module:** `stages/classify.py`

Post-generation classification into 7 categories based on actual question+answer content:

1. `circuits_and_gates` - Circuit construction, standard gates
2. `algorithms_and_applications` - VQE, QAOA, Grover, QML
3. `transpilation_and_compilation` - Pass managers, optimization
4. `primitives_and_execution` - SamplerV2, EstimatorV2, sessions
5. `noise_and_error_mitigation` - ZNE, TREX, dynamical decoupling
6. `quantum_info_and_operators` - Statevectors, operators, fidelity
7. `hardware_and_providers` - Backend properties, ISA circuits

---

### Stage 7: Build

**Module:** `dataset/builder.py`

Creates stratified train/val/test splits.

```
Input:  outputs/generated/samples.pkl
Output: outputs/splits/{train,val,test}.pkl
```

**Stratification Dimensions:**
- Category (7 quantum computing categories)
- Question type (function_completion, code_generation, qa)
- Modality (multimodal vs text-only)

**Default Split Ratios:** 70% train / 15% val / 15% test

---

### Stage 8: Export

**Module:** `dataset/exporter.py`

Exports to HuggingFace datasets format.

```
Input:  outputs/splits/{train,val,test}.pkl
Output: outputs/final/ (HuggingFace DatasetDict)
```

**Schema:**
```python
{
    "question": str,       # Input prompt/question
    "answer": str,         # Reference solution/answer
    "category": str,       # 7 categories
    "type": str,           # function_completion, code_generation, qa
    "test_code": str,      # Unit test (code types only)
    "entry_point": str,    # Function name (code types only)
    "image": PIL.Image,    # Associated image (multimodal only)
    "source": str,         # Relative source document path
}
```

---

### Stage 9: Analyze

**Modules:** `dataset/analyzer.py`, `dataset/plotter.py`, `tools/pipeline_analyzer.py`

Generates statistics and visualizations.

```
Output: outputs/analysis/
```

**Generated Artifacts:**

| File | Description |
|------|-------------|
| `statistics.json` | Complete distribution statistics |
| `overview_dashboard.png` | Multi-panel dashboard |
| `split_distribution.png` | Samples per split |
| `type_distribution.png` | Question type distribution |
| `category_distribution.png` | Category distribution |
| `modality_distribution.png` | Multimodal vs text-only |
| `multimodal_breakdown.png` | Multimodal by split/type/category |
| `source_analysis.png` | Pipeline source statistics |
| `chunk_distribution.png` | Chunk size and content |
| `diversity_sweep.png` | Allocation diversity analysis |

---

## Image Reference System

Images use a marker-based approach for flexibility and efficiency.

### At Parse Time
```markdown
# Creating Bell States

A Bell state is a maximally entangled quantum state...

[IMAGE:img_a1b2c3d4e5f6]

This circuit creates the φ⁺ Bell state when applied to |00⟩...
```

### At Generation Time (Context Building)

For images generated by code cells, the code context is prioritized:

```markdown
[PRIORITY - Code That Generated Target Image]
```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
```
USE THIS CODE as primary reference for implementation. The image shows the OUTPUT of this code.

[Target Image Description - for visual reference only]
Bell Circuit Diagram: 2-qubit circuit with H gate on q0 and CNOT...

[Main Content]
# Creating Bell States

A Bell state is a maximally entangled quantum state...
This circuit creates the φ⁺ Bell state when applied to |00⟩...

[Prior Code Context]
```python
from qiskit import QuantumCircuit
```
```

For images without code context (diagrams, charts):
```markdown
[TARGET IMAGE: Bell Circuit Diagram]
The circuit shows a 2-qubit Bell state preparation:
- Hadamard gate on qubit q0 (creates superposition)
- CNOT gate with q0 as control and q1 as target
[END TARGET IMAGE]

[Main Content]
...
```

**Context Prioritization:**
1. **Code that generated image** - First priority for code output images
2. **Image description** - Truncated to 500 chars when code context exists
3. **Main content** - Document text
4. **Prior code context** - Accumulated code from earlier cells

**Benefits:**
- Code patterns prioritized over visual descriptions
- Prevents layer-by-layer image enumeration in questions
- Efficient: transcriptions only inserted when needed
- Clean context hierarchy for better generation quality

---

## Question Type Formats

### function_completion

**Question:** Function stub with imports, signature, docstring, and `pass`
**Answer:** ONLY the function body (lines replacing `pass`)

```python
# Question
from qiskit import QuantumCircuit

def create_bell_state() -> QuantumCircuit:
    """Create a Bell state (|00⟩ + |11⟩)/√2 and return the circuit."""
    pass

# Answer
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
return qc
```

### code_generation

**Question:** Natural language task with function specification
**Answer:** Complete code with imports and function definition

```python
# Question
Create a 3-qubit GHZ state circuit and return it.
You must implement this using a function named `create_ghz` with no arguments.

# Answer
from qiskit import QuantumCircuit

def create_ghz():
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    return ghz
```

### qa

**Question:** Theory/concept question
**Answer:** Explanation (may include code examples)

No unit test required. Code in answers verified for syntax/execution only.

---

## Configuration Reference

### Full Configuration Structure

```yaml
sources:
  - path: /path/to/data
    type: directory
    include_patterns: ["*.ipynb", "*.mdx", "*.pdf"]
    exclude_patterns: ["**/node_modules/**"]

categories:
  - name: circuits_and_gates
    description: Quantum circuits, standard gates...
  # ... 7 categories

models:
  endpoints:
    - name: vision-model
      base_url: ${VISION_MODEL_BASE_URL}
      api_key: ${VISION_MODEL_API_KEY}
      model_name: ${VISION_MODEL_NAME}
      max_tokens: 8192
      temperature: 0.1
      timeout: 300.0        # Request timeout in seconds
      max_retries: 5        # Retry attempts per request
      retry_delay: 1.0      # Initial retry delay (exponential backoff)
    - name: question-model
      # ...
    - name: answer-model
      # ...
    - name: curate-model
      # ...

prompts:
  input_generation_system: |
    # System prompt for question generation
  test_generation_prompt: |
    # Prompt for unit test generation
  answer_generation_system: |
    # System prompt for answer generation
  # ... many prompt templates

generation:
  target_samples: 8000
  
  # Model assignments
  question_model: question-model
  vision_model: vision-model
  answer_model: answer-model
  curate_model: curate-model
  
  # Concurrency
  llm_concurrency: 16
  vlm_concurrency: 16
  
  # Allocation
  over_allocation_factor: 1.8
  diversity_weight: 0.4
  max_generation_attempts: 3
  keep_extra_samples: true
  
  type_allocations:
    qa:
      ratio: 0.30
      multimodal_ratio: 0.70
    code_generation:
      ratio: 0.35
      multimodal_ratio: 0.30
    function_completion:
      ratio: 0.35
      multimodal_ratio: 0.30
  
  # Chunking
  max_context_length: 2912
  min_chunk_length: 800
  max_code_blocks_per_chunk: 3
  max_images_per_chunk: 3
  
  # Filtering
  enable_image_transcription: true
  enable_content_filtering: true
  enable_candidate_filtering: true
  enable_curate_filtering: true
  enable_deduplication: true
  similarity_threshold: 0.85
  
  # Code verification
  enable_code_verification: true
  code_verification_max_iterations: 7
  code_verification_timeout: 60
  test_validation_timeout: 60
  answer_checkpoint_interval: 50  # Save checkpoint every N answer completions

dataset:
  name: quantum-multimodal
  description: |
    High-quality multimodal dataset...
  
  parsed_dir: outputs/parsed
  generated_dir: outputs/generated
  final_dir: outputs/final
  images_dir: outputs/images
  
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  
  license: apache-2.0

seed: 42
```

---

## CLI Commands

### Pipeline Execution

```bash
# Complete pipeline
synthetic-data pipeline --config config.yaml

# Document processing stages (1-5)
synthetic-data parse --config config.yaml
synthetic-data transcribe --config config.yaml
synthetic-data filter-images --config config.yaml
synthetic-data chunk --config config.yaml
synthetic-data filter-chunks --config config.yaml

# Generation stages (6a-6e) - can run individually or together
synthetic-data plan --config config.yaml
synthetic-data filter-candidates --config config.yaml
synthetic-data answer --config config.yaml
synthetic-data curate --config config.yaml
synthetic-data classify --config config.yaml

# Or run all generation stages together
synthetic-data generate --config config.yaml

# Dataset building (7-8)
synthetic-data build --config config.yaml
synthetic-data export --config config.yaml

# With HuggingFace Hub
synthetic-data export --config config.yaml --hub-id user/dataset --analyze
```

### Analysis

```bash
# Complete analysis
synthetic-data analyze --config config.yaml --source splits

# Allocation sweep analysis
synthetic-data analyze-allocation --config config.yaml --target 8000

# Options
synthetic-data analyze --config config.yaml \
  --source splits \
  --pipeline \
  --allocation \
  --hub-id user/dataset
```

### Inspection

```bash
# Inspect pipeline artifacts
synthetic-data inspect --config config.yaml --stage chunks --count 5
synthetic-data inspect --config config.yaml --stage samples --random

# Inspect generation traces
synthetic-data inspect-traces --config config.yaml --stage answer_generation
synthetic-data inspect-traces --config config.yaml --failed --count 10
```

### Utilities

```bash
# Configuration info
synthetic-data info --config config.yaml
synthetic-data validate-config --config config.yaml

# Cache management
synthetic-data cache-info --config config.yaml
synthetic-data cache-clear --config config.yaml --stage chunks
```

---

## Output Structure

```
outputs/
├── parsed/
│   ├── documents.pkl           # Parsed documents
│   └── summary.json            # Parsing statistics
├── transcribed/
│   ├── documents.pkl           # Documents with image transcriptions
│   └── summary.json            # Transcription statistics
├── filtered_images/
│   ├── documents.pkl           # Documents with quality-filtered images
│   ├── image_filter_decisions.jsonl
│   └── summary.json
├── chunks/
│   ├── chunks.pkl              # Content chunks
│   └── summary.json
├── filtered/
│   ├── chunks.pkl              # Quality-filtered chunks
│   ├── filter_decisions.jsonl
│   └── summary.json
├── planned/
│   └── candidates.pkl          # Generated question/test candidates
├── filtered_candidates/
│   ├── candidates.pkl          # Quality-filtered candidates
│   └── filter_decisions.jsonl
├── answered/
│   ├── samples.pkl             # Samples with validated answers
│   └── failures.jsonl          # Answer generation failures
├── curated/
│   ├── samples.pkl             # Quality-curated samples
│   └── rejected.jsonl          # Curation rejections
├── generated/
│   ├── samples.pkl             # Final classified samples
│   └── samples.jsonl           # JSONL export
├── splits/
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── summary.json
├── analysis/
│   ├── statistics.json
│   └── *.png                   # Visualization plots
└── final/
    ├── train/
    ├── validation/
    ├── test/
    └── README.md               # Dataset card
```

---

## Checkpoint and Retry System

All long-running stages support fault tolerance with checkpoints and automatic retries.

### Automatic Retry Logic

All model API calls use exponential backoff retry with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | 5 | Maximum retry attempts per request |
| `retry_delay` | 1.0s | Initial delay, doubles each retry (exponential backoff) |
| `timeout` | 300s | Per-request timeout |

**Retry Triggers:**
- Network errors (`httpx.RequestError`)
- HTTP errors (`httpx.HTTPStatusError`) 
- Timeouts (`asyncio.TimeoutError`)
- Empty/None responses (`ValueError`)

**Progress Tracking:**
- Progress callbacks are called ONCE per item after all retries complete
- Retries are transparent to the progress system
- No duplicate progress updates during retry attempts

### Per-Stage Checkpoints

| Stage | Checkpoint File | Tracks | Interval |
|-------|-----------------|--------|----------|
| Transcribe | `transcribe_checkpoint.pkl` | Transcribed image IDs | Every N images (batch_size) |
| Filter Images | `filter_images_checkpoint.pkl` | Filtered image IDs + decisions | Every N images (batch_size) |
| Filter Chunks | `filter_chunks_checkpoint.pkl` | Filtered chunk IDs + results | Every N chunks (batch_size) |
| Plan (6a) | `plan_checkpoint.pkl` | Tasks + candidates + phase | Per batch |
| Filter Candidates (6b) | `filter_candidates_checkpoint.pkl` | Processed indices + decisions | Per batch |
| Answer (6c) | `answer_checkpoint.pkl` | Processed indices + results | Every N answers |
| Curate (6d) | `curate_checkpoint.pkl` | Processed indices + decisions | Per batch |
| Classify (6e) | `classify_checkpoint.pkl` | Processed indices + categories | Per batch |

### Generation Stage Checkpointing

Each generation sub-stage has **independent checkpointing**:

1. **Plan (6a):** Tracks processed task indices; each task is fully processed (question + refinement + test + corrections) before checkpoint
2. **Filter Candidates (6b):** Tracks filter decisions per candidate
3. **Answer (6c):** Incremental checkpoints every N completions
4. **Curate (6d):** Tracks curation decisions per sample
5. **Classify (6e):** Tracks classification per sample

Each stage can be resumed independently if interrupted.

### Resume Behavior

When resuming from checkpoint:
```
  Resuming from stage: answer_generation_partial
    Resuming: 487/1000 already processed
  Stage 3: Generating answers...
    [Progress bar continues from 487/1000]
```

**Features:**
- Tracks **processed item IDs/indices**, not just counts
- Skips already-processed items on resume
- Stores full objects (not dicts) for proper restoration
- Periodic saves prevent data loss
- Automatic cleanup on success
- Clear console output showing resume progress

---

## Performance Optimization

### Parallelization Strategy

All stages use the same optimized pattern:
1. **Semaphore-based concurrency** - Precise control over parallel requests
2. **No artificial batching** - All items launch concurrently (up to limit)
3. **Streaming progress** - Updates after each item
4. **Connection pooling** - HTTP clients with optimized limits

### Recommended Settings

| Workload | llm_concurrency | vlm_concurrency |
|----------|-----------------|-----------------|
| OpenAI API | 10-20 | 16-32 |
| Local (vLLM/Ollama) | 20-50 | 32-64 |
| Cloud with high limits | 50+ | 64+ |

---

## Qiskit 2.0 API Compliance

The pipeline enforces Qiskit 2.0 API patterns:

| Deprecated | Replacement |
|------------|-------------|
| `bind_parameters()` | `assign_parameters()` |
| `transpile()` | `generate_preset_pass_manager().run()` |
| `execute()` | `Sampler/Estimator` primitives |
| `Qubit.index` | `circuit.find_bit(qubit)` |
| `qiskit.primitives.Sampler` | `StatevectorSampler` |
| `qiskit.test` | `qiskit_ibm_runtime.fake_provider` |

---

## Tracing and Debugging

### Enable Tracing
```bash
synthetic-data generate --config config.yaml  # Enabled by default
synthetic-data generate --config config.yaml --no-trace  # Disable
```

### Trace Content
Each conversation includes:
1. **Input Generation:** System prompt, user prompt, generated question, refinement step
2. **Test Generation:** Conversation history, validation errors, correction loop (up to 3 attempts)
3. **Answer Generation:** System prompt, answer prompt, test validation, correction loop

### Inspecting Traces
```bash
# View summary
synthetic-data inspect-traces --config config.yaml

# Filter by stage
synthetic-data inspect-traces --config config.yaml --stage answer_generation

# View failures
synthetic-data inspect-traces --config config.yaml --failed

# Full content (no truncation)
synthetic-data inspect-traces --config config.yaml --index 5 --full
```

---

## Evaluation Alignment

Dataset aligns with Qiskit HumanEval benchmarks:

| Benchmark | Format | Our Type |
|-----------|--------|----------|
| Qiskit HumanEval | Signature + docstring | `function_completion` |
| Qiskit HumanEval Hard | Natural language | `code_generation` |
