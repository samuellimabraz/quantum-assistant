# Quantum Assistant Demo

Interactive web interface for the Quantum Assistant - a multimodal Vision-Language Model specialized for quantum computing with Qiskit.

[![Paper](https://img.shields.io/badge/Paper-ESWA%202026-00A4A6)](https://doi.org/10.1016/j.eswa.2026.133931)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.eswa.2026.133931-blue)](https://doi.org/10.1016/j.eswa.2026.133931)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Quantum%20Assistant-yellow)](https://huggingface.co/datasets/samuellimabraz/quantum-assistant)
[![Models](https://img.shields.io/badge/🤗%20Models-Collection-orange)](https://huggingface.co/collections/samuellimabraz/quantum-assistant)

## Features

- **Interactive Chat**: Ask questions about quantum computing, generate Qiskit code
- **Multimodal Support**: Upload circuit diagrams, Bloch spheres, or histograms for analysis
- **Code Execution**: Run generated Qiskit/Python code directly in the browser with sandboxed execution
- **Practice Mode**: Solve coding problems from the full dataset with:
  - **800+ coding problems** from test and validation splits
  - **Split selector**: Switch between test (1290 total) and validation (1239 total) datasets
  - **Pagination**: Navigate through problems with 25 items per page
  - **Search**: Find specific problems by keyword
  - **Filters**: Filter by task type (Function Completion, Code Generation), category, and multimodal (with/without images)
  - **AI Helper**: Get hints and guidance while solving problems
  - **Progress tracking**: Mark solved problems with local storage persistence
- **Dataset Examples**: Browse and test with samples from the Quantum Assistant test set
- **Code Highlighting**: Syntax-highlighted code blocks with copy functionality
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Access to a VLM API endpoint (vLLM, OpenAI-compatible, etc.)

### Installation

```bash
cd src/demo

npm install
uv venv
uv pip install -r pyproject.toml
```

### Configuration

Create a `.env.local` file:

```bash
# Model API Configuration
DEMO_MODEL_URL=http://localhost:8000/v1
DEMO_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
DEMO_API_KEY=your-api-key
DEMO_MAX_TOKENS=4096
DEMO_TEMPERATURE=0.1
DEMO_TIMEOUT=120

HF_TOKEN=hf_...

# Python environment for code execution (Practice mode)
PYTHON_PATH=/path/to/quantum-assistant/.venv/bin/python
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
src/demo/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat/route.ts       # Chat API endpoint
│   │   │   ├── execute/route.ts    # Code execution endpoint
│   │   │   └── examples/route.ts   # Dataset examples endpoint
│   │   ├── page.tsx                # Main page
│   │   ├── layout.tsx              # Root layout
│   │   └── globals.css             # Global styles
│   ├── components/
│   │   ├── Header/                 # Navigation header
│   │   ├── Chat/                   # Chat interface components
│   │   │   ├── ChatInterface.tsx   # Main chat component
│   │   │   ├── Message.tsx         # Message with code execution
│   │   │   └── ExecutionResult.tsx # Code execution output display
│   │   ├── Practice/               # Practice mode components
│   │   │   ├── PracticeInterface.tsx # Main practice layout
│   │   │   ├── ProblemList.tsx     # Problem list with pagination/search
│   │   │   ├── CodeEditor.tsx      # Monaco code editor
│   │   │   ├── TestRunner.tsx      # Unit test execution
│   │   │   └── AIHelper.tsx        # AI assistance sidebar
│   │   └── Examples/               # Dataset examples panel
│   ├── lib/
│   │   ├── api/vlm-client.ts       # VLM API client
│   │   └── dataset/loader.ts       # HuggingFace dataset loader
│   ├── config/constants.ts         # Project configuration
│   └── types/index.ts              # TypeScript types
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
```

## API Endpoints

### POST /api/chat

Send messages to the VLM for completion.

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "Create a Bell state circuit" }
  ]
}
```

### GET /api/examples

Fetch examples from the Quantum Assistant dataset.

Query parameters:
- `split`: train | validation | test (default: test)
- `limit`: number of examples (default: 50, max: 100)
- `offset`: pagination offset (default: 0)
- `type`: function_completion | code_generation | qa
- `category`: circuits_and_gates | quantum_info_and_operators | ...
- `hasImage`: true | false
- `codingOnly`: true | false - filter for problems with test code and entry points
- `search`: text search in question and answer

### POST /api/examples

Get dataset metadata.

```json
{
  "action": "getSplitInfo"  // Returns counts for each split
}
```

```json
{
  "action": "getCodingCount",
  "split": "test"  // Returns count of coding problems in split
}
```

### POST /api/execute

Execute Python/Qiskit code in a sandboxed environment.

```json
{
  "code": "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nprint(qc.draw())",
  "timeout": 30
}
```

Response:
```json
{
  "success": true,
  "output": "     ┌───┐     \nq_0: ┤ H ├──■──\n     └───┘┌─┴─┐\nq_1: ─────┤ X ├\n          └───┘",
  "error": "",
  "executionTime": 245,
  "hasCircuitOutput": true
}
```

**Security Features:**
- Sandboxed execution with blocked file writes
- Subprocess isolation with timeout protection
- Blocked dangerous system calls (os.system, subprocess, etc.)
- Maximum code length limit (50KB)
- Execution timeout (max 60s)

## Using with Fine-tuned Models

To use with the fine-tuned Quantum Assistant models:

1. Download a model from the [HuggingFace collection](https://huggingface.co/collections/samuellimabraz/quantum-assistant)
2. Serve with vLLM:

```bash
vllm serve samuellimabraz/qwen3-vl-8b-quantum-r32-1ep \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

3. Configure `.env.local` with the endpoint:

```bash
DEMO_MODEL_URL=http://localhost:8000/v1
DEMO_MODEL_NAME=samuellimabraz/qwen3-vl-8b-quantum-r32-1ep
```

## Author

**Samuel Lima Braz**  
Universidade Federal de Itajubá (UNIFEI)  
Advisor: Prof. João Paulo Reus Rodrigues Leite

## Citation

```bibtex
@article{braz2026quantumassistant,
  title     = {Quantum Assistant: Specialization of Multimodal Vision-Language Models for Quantum Computing},
  author    = {Braz, Samuel Lima and Leite, Jo{\~a}o Paulo Reus Rodrigues},
  journal   = {Expert Systems with Applications},
  year      = {2026},
  issn      = {0957-4174},
  doi       = {10.1016/j.eswa.2026.133931},
  url       = {https://doi.org/10.1016/j.eswa.2026.133931},
  publisher = {Elsevier}
}
```

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.

