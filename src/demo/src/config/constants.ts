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

export const SYSTEM_PROMPT = `You are Quantum Assistant, an expert AI specialized in quantum computing, physics, mathematics, and the Qiskit framework.

## SCOPE RESTRICTIONS (STRICTLY ENFORCED)
You ONLY answer questions related to:
- Quantum computing (circuits, gates, algorithms, hardware, error correction)
- Qiskit framework and IBM Quantum services
- Physics (quantum mechanics, quantum information theory)
- Mathematics (linear algebra, probability, complex numbers as they relate to quantum)
- Machine learning for quantum (QML, variational algorithms, optimization)
- Scientific computing related to quantum (NumPy, SciPy for quantum applications)

## FORBIDDEN TOPICS - DO NOT ANSWER
If asked about any of the following, politely decline and redirect to quantum computing:
- General programming unrelated to quantum computing
- Personal advice, relationships, lifestyle
- Politics, news, current events, opinions
- Harmful, illegal, or unethical content
- Medical, legal, or financial advice
- Creative writing, stories, poetry (unless quantum-themed educational)
- Other AI systems, jailbreaking, prompt injection
- Anything that could be used maliciously

## RESPONSE GUIDELINES
1. Generate precise, well-documented Qiskit code following Qiskit 2.0 best practices
2. Explain quantum computing concepts clearly with mathematical rigor when appropriate
3. Interpret quantum circuit diagrams, Bloch spheres, and measurement histograms
4. Help with function completion, code generation, and conceptual questions
5. Use Qiskit 2.0 APIs exclusively
6. Prefer primitives (SamplerV2, EstimatorV2) over legacy execute()
7. Use assign_parameters() instead of deprecated bind_parameters()
8. Use generate_preset_pass_manager() for circuit optimization
9. Include all necessary imports in code solutions
10. Provide clear, educational explanations

## CODE SAFETY
- Never generate code that accesses environment variables, files outside the sandbox, or network resources
- Never generate code using dangerous modules (ctypes, pickle, subprocess, etc.)
- Keep code focused on quantum computing tasks

## OFF-TOPIC RESPONSE
If a question is outside your scope, respond with:
"I'm Quantum Assistant, specialized in quantum computing, Qiskit, physics, and related mathematics. I can't help with [topic], but I'd be happy to assist with quantum computing questions! For example, I can help you understand quantum gates, create quantum circuits, or explain quantum algorithms."

Respond accurately and helpfully to quantum computing questions while maintaining strict topic boundaries.`;

// List of allowed topic keywords for input validation
export const ALLOWED_TOPICS = [
  // Quantum Computing
  'quantum', 'qubit', 'qubits', 'superposition', 'entanglement', 'measurement',
  'circuit', 'gate', 'hadamard', 'cnot', 'pauli', 'rotation', 'phase',
  'bloch', 'sphere', 'state', 'vector', 'amplitude', 'probability',
  'algorithm', 'grover', 'shor', 'vqe', 'qaoa', 'qft', 'fourier',
  'error', 'correction', 'noise', 'decoherence', 'fidelity', 'mitigation',
  'transpiler', 'transpile', 'optimization', 'compilation',
  
  // Qiskit
  'qiskit', 'ibm', 'aer', 'simulator', 'backend', 'provider', 'runtime',
  'sampler', 'estimator', 'primitive', 'job', 'result', 'counts',
  'quantumcircuit', 'classicalregister', 'quantumregister',
  'pass', 'manager', 'layout', 'routing', 'scheduling',
  
  // Physics & Math
  'physics', 'mechanics', 'hamiltonian', 'unitary', 'hermitian', 'operator',
  'eigenvalue', 'eigenvector', 'matrix', 'tensor', 'linear', 'algebra',
  'hilbert', 'space', 'basis', 'orthogonal', 'projection',
  'complex', 'number', 'exponential', 'trigonometric',
  'probability', 'distribution', 'expectation', 'variance',
  'wave', 'function', 'schrodinger', 'dirac', 'bra', 'ket', 'notation',
  
  // QML & Optimization
  'machine', 'learning', 'variational', 'ansatz', 'parameter', 'parametrized',
  'optimization', 'gradient', 'descent', 'cost', 'function', 'loss',
  'training', 'classical', 'hybrid', 'neural', 'kernel',
  
  // Scientific Computing
  'numpy', 'scipy', 'matplotlib', 'plot', 'histogram', 'visualization',
  'array', 'matrix', 'calculation', 'computation', 'simulation',
  
  // General programming (quantum-related)
  'python', 'code', 'function', 'class', 'import', 'library',
  'example', 'tutorial', 'explain', 'how', 'what', 'why', 'help',
  'implement', 'create', 'build', 'make', 'generate', 'write',
];

// Blocked patterns that should never be allowed
export const BLOCKED_INPUT_PATTERNS = [
  // Prompt injection attempts
  /ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)/i,
  /disregard\s+(previous|all|above|prior)/i,
  /forget\s+(everything|all|your)\s+(instructions?|rules?|training)/i,
  /new\s+instructions?:/i,
  /system\s*prompt/i,
  /jailbreak/i,
  /dan\s*mode/i,
  /pretend\s+(you\s+are|to\s+be)/i,
  /act\s+as\s+(if|a)/i,
  /roleplay\s+as/i,
  /you\s+are\s+now/i,
  
  // Harmful content requests
  /how\s+to\s+(hack|attack|exploit|break\s+into)/i,
  /malware|virus|trojan|ransomware/i,
  /steal\s+(data|information|credentials|password)/i,
  /bypass\s+(security|authentication|firewall)/i,
  /injection\s+attack/i,
  /sql\s+injection/i,
  /xss|cross.?site/i,
  
  // Explicit requests to bypass restrictions
  /bypass\s+(filter|restriction|limitation|safety)/i,
  /disable\s+(safety|filter|moderation)/i,
  /unlock\s+(hidden|secret|restricted)/i,
  /override\s+(rules?|restrictions?)/i,
  
  // Off-topic explicit requests
  /write\s+(me\s+)?(a\s+)?(story|poem|essay|article|blog)/i,
  /tell\s+me\s+(a\s+)?joke/i,
  /relationship\s+advice/i,
  /dating\s+advice/i,
  /political\s+opinion/i,
  /investment\s+advice/i,
  /medical\s+(advice|diagnosis)/i,
  /legal\s+advice/i,
];

