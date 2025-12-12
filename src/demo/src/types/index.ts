export type TaskType = 'function_completion' | 'code_generation' | 'qa';

export type Category =
  | 'circuits_and_gates'
  | 'quantum_info_and_operators'
  | 'algorithms_and_applications'
  | 'hardware_and_providers'
  | 'transpilation_and_compilation'
  | 'primitives_and_execution'
  | 'noise_and_error_mitigation';

export interface DatasetExample {
  id: string;
  question: string;
  answer: string;
  type: TaskType;
  category: Category;
  imageUrl?: string;
  hasImage: boolean;
  testCode?: string;
  entryPoint?: string;
  source: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  imageUrl?: string;
  imageBase64?: string;
  timestamp: Date;
  isLoading?: boolean;
}

export interface ChatRequest {
  messages: Array<{
    role: string;
    content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>;
  }>;
  image?: string;
}

export interface ChatResponse {
  content: string;
  error?: string;
}

export interface ModelConfig {
  baseUrl: string;
  modelName: string;
  apiKey: string;
  maxTokens: number;
  temperature: number;
  timeout: number;
}

export interface ExecuteRequest {
  code: string;
  timeout?: number;
}

export interface ExecuteResponse {
  success: boolean;
  output: string;
  error: string;
  executionTime: number;
  hasCircuitOutput?: boolean;
}

