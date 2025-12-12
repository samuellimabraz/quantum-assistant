import type { Category, TaskType } from '@/types';

export const PROJECT_CONFIG = {
  name: 'Quantum Assistant',
  description: 'Multimodal VLM specialized for Quantum Computing with Qiskit',
  version: '1.0.0',
  author: 'Samuel Lima Braz',
  advisor: 'João Paulo Reus Rodrigues Leite',
  institution: 'UNIFEI - Universidade Federal de Itajubá',
  year: 2025,
} as const;

export const LINKS = {
  github: 'https://github.com/samuellimabraz/quantum-assistant',
  dataset: 'https://huggingface.co/datasets/samuellimabraz/quantum-assistant',
  models: 'https://huggingface.co/collections/samuellimabraz/quantum-assistant',
  qiskit: 'https://qiskit.org/',
} as const;

export const TASK_LABELS: Record<TaskType, { label: string; description: string; color: string }> = {
  function_completion: {
    label: 'Function Completion',
    description: 'Complete function body from signature + docstring',
    color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  },
  code_generation: {
    label: 'Code Generation',
    description: 'Generate complete code from natural language',
    color: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  },
  qa: {
    label: 'Question Answering',
    description: 'Conceptual explanations and theory',
    color: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  },
};

export const CATEGORY_LABELS: Record<Category, string> = {
  circuits_and_gates: 'Circuits & Gates',
  quantum_info_and_operators: 'Quantum Info',
  algorithms_and_applications: 'Algorithms',
  hardware_and_providers: 'Hardware',
  transpilation_and_compilation: 'Transpilation',
  primitives_and_execution: 'Primitives',
  noise_and_error_mitigation: 'Error Mitigation',
};

export const SYSTEM_PROMPT = `You are Quantum Assistant, an expert AI specialized in quantum computing and the Qiskit framework.

Your capabilities:
1. Generate precise, well-documented Qiskit code following Qiskit 2.0 best practices
2. Explain quantum computing concepts clearly
3. Interpret quantum circuit diagrams, Bloch spheres, and measurement histograms
4. Help with function completion, code generation, and conceptual questions

Guidelines:
- Use Qiskit 2.0 APIs exclusively
- Prefer primitives (SamplerV2, EstimatorV2) over legacy execute()
- Use assign_parameters() instead of deprecated bind_parameters()
- Use generate_preset_pass_manager() for circuit optimization
- Include all necessary imports in code solutions
- Provide clear, educational explanations

Respond accurately and helpfully to quantum computing questions.`;

