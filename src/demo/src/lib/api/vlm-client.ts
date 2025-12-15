import type { ModelConfig } from '@/types';

interface MessageContent {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | MessageContent[];
}

interface VLMResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
}

interface StreamChoice {
  delta: {
    content?: string;
  };
}

interface StreamChunk {
  choices: StreamChoice[];
}

export class VLMClient {
  private config: ModelConfig;

  constructor(config: ModelConfig) {
    this.config = config;
  }

  private buildPayload(messages: ChatMessage[], stream = false): Record<string, unknown> {
    return {
      model: this.config.modelName,
      messages,
      max_tokens: this.config.maxTokens,
      temperature: this.config.temperature,
      stream,
    };
  }

  async generate(
    prompt: string,
    systemPrompt?: string,
    imageBase64?: string
  ): Promise<string> {
    const messages: ChatMessage[] = [];

    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }

    if (imageBase64) {
      messages.push({
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          {
            type: 'image_url',
            image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
          },
        ],
      });
    } else {
      messages.push({ role: 'user', content: prompt });
    }

    return this.chat(messages);
  }

  async chat(messages: ChatMessage[]): Promise<string> {
    const url = `${this.config.baseUrl}/chat/completions`;
    const headers: HeadersInit = { 'Content-Type': 'application/json' };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const payload = this.buildPayload(messages, false);

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeout * 1000
    );

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const data: VLMResponse = await response.json();
      const content = data.choices[0]?.message?.content;

      if (!content) {
        throw new Error('Empty response from model');
      }

      return content;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.config.timeout}s`);
      }
      throw error;
    }
  }

  async *chatStream(messages: ChatMessage[]): AsyncGenerator<string, void, unknown> {
    const url = `${this.config.baseUrl}/chat/completions`;
    const headers: HeadersInit = { 'Content-Type': 'application/json' };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const payload = this.buildPayload(messages, true);

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeout * 1000
    );

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      if (!response.body) {
        throw new Error('No response body for streaming');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          const data = trimmed.slice(6);
          if (data === '[DONE]') return;

          try {
            const chunk: StreamChunk = JSON.parse(data);
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              yield content;
            }
          } catch {
            // Skip malformed chunks
          }
        }
      }
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${this.config.timeout}s`);
      }
      throw error;
    }
  }
}

export function createVLMClient(): VLMClient {
  const config: ModelConfig = {
    baseUrl: process.env.DEMO_MODEL_URL || 'http://localhost:8000/v1',
    modelName: process.env.DEMO_MODEL_NAME || 'Qwen/Qwen3-VL-8B-Instruct',
    apiKey: process.env.DEMO_API_KEY || '',
    maxTokens: parseInt(process.env.DEMO_MAX_TOKENS || '4096', 10),
    temperature: parseFloat(process.env.DEMO_TEMPERATURE || '0.1'),
    timeout: parseInt(process.env.DEMO_TIMEOUT || '120', 10),
  };

  return new VLMClient(config);
}
