import { NextRequest } from 'next/server';
import { createVLMClient } from '@/lib/api/vlm-client';
import { ALLOWED_TOPICS, BLOCKED_INPUT_PATTERNS } from '@/config/constants';

export const maxDuration = 120;

interface MessageContent {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | MessageContent[];
}

interface ChatRequestBody {
  messages: ChatMessage[];
  stream?: boolean;
}

interface ContentValidation {
  valid: boolean;
  reason?: string;
  isOffTopic?: boolean;
}

/**
 * Extract text content from a message for validation
 */
function extractTextContent(content: string | MessageContent[]): string {
  if (typeof content === 'string') {
    return content;
  }
  return content
    .filter((c): c is MessageContent & { type: 'text'; text: string } => c.type === 'text' && !!c.text)
    .map(c => c.text)
    .join(' ');
}

/**
 * Validate user input for malicious patterns and topic relevance
 */
function validateUserInput(text: string): ContentValidation {
  const lowerText = text.toLowerCase();
  
  // Check for blocked patterns (prompt injection, harmful content, etc.)
  for (const pattern of BLOCKED_INPUT_PATTERNS) {
    if (pattern.test(text)) {
      return {
        valid: false,
        reason: "I can't process this request. Please ask a question related to quantum computing, Qiskit, physics, or mathematics.",
      };
    }
  }
  
  // Check message length (prevent abuse)
  if (text.length > 10000) {
    return {
      valid: false,
      reason: 'Message too long. Please keep your question under 10,000 characters.',
    };
  }
  
  // Check if the message contains any relevant topic keywords
  // Images are always allowed (circuit diagrams, Bloch spheres, etc.)
  const hasImage = text.includes('[IMAGE]') || text.length < 20; // Short messages might be follow-ups
  
  if (!hasImage) {
    const words = lowerText.split(/\s+/);
    const hasRelevantTopic = ALLOWED_TOPICS.some(topic => {
      // Check for whole word or part of compound word
      return words.some(word => 
        word.includes(topic.toLowerCase()) || 
        topic.toLowerCase().includes(word)
      );
    });
    
    // Also check for common question patterns
    const isQuestion = /^(what|how|why|when|where|can|could|would|should|is|are|do|does|explain|describe|help|show|create|implement|write|generate|build|make)/i.test(lowerText.trim());
    const hasCodeContext = /```|def\s|import\s|class\s|function|circuit/i.test(text);
    
    // Be permissive: if it's a question or has code context, allow it
    // The model will redirect off-topic questions anyway
    if (!hasRelevantTopic && !isQuestion && !hasCodeContext && text.length > 50) {
      return {
        valid: true, // Still valid, but flag as potentially off-topic
        isOffTopic: true,
      };
    }
  }
  
  return { valid: true };
}

/**
 * Create off-topic response message
 */
function createOffTopicResponse(): string {
  return `I'm **Quantum Assistant**, specialized in quantum computing, Qiskit, physics, and related mathematics.

I can help you with:
- 🔬 **Quantum Computing**: Circuits, gates, algorithms, error correction
- 💻 **Qiskit**: Code generation, debugging, best practices
- 📐 **Physics & Math**: Quantum mechanics, linear algebra, probability
- 🤖 **Quantum ML**: Variational algorithms, optimization, hybrid systems

**Please ask a question related to these topics!**

For example:
- "How do I create a Bell state in Qiskit?"
- "Explain the Grover's algorithm"
- "What is quantum entanglement?"`;
}

function isConnectionError(error: unknown): boolean {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    const cause = (error as Error & { cause?: Error })?.cause;

    if (message.includes('fetch failed') || message.includes('econnrefused')) {
      return true;
    }

    if (cause && 'code' in cause && cause.code === 'ECONNREFUSED') {
      return true;
    }
  }
  return false;
}

function createErrorMessage(isConnection: boolean): string {
  if (isConnection) {
    const modelUrl = process.env.DEMO_MODEL_URL || 'http://localhost:8000/v1';
    return `**Model Server Not Available**\n\nCould not connect to the model at:\n\`${modelUrl}\`\n\n**To use the chat feature:**\n1. Start a VLM server (vLLM, Ollama, etc.)\n2. Configure \`.env.local\` with your endpoint:\n\`\`\`\nDEMO_MODEL_URL=http://your-server:port/v1\nDEMO_MODEL_NAME=your-model-name\nDEMO_API_KEY=your-api-key\n\`\`\`\n3. Restart the demo server\n\n*Examples panel still works - try selecting a test sample!*`;
  }
  return 'An error occurred while processing your request.';
}

export async function POST(request: NextRequest) {
  try {
    const body: ChatRequestBody = await request.json();
    const { messages, stream = true } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: 'Invalid request: messages array required' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Find the last user message for validation
    const userMessages = messages.filter(m => m.role === 'user');
    const lastUserMessage = userMessages[userMessages.length - 1];
    
    if (lastUserMessage) {
      const userText = extractTextContent(lastUserMessage.content);
      const validation = validateUserInput(userText);
      
      // If input is invalid (malicious/harmful), return error
      if (!validation.valid && validation.reason) {
        const encoder = new TextEncoder();
        
        if (stream) {
          const errorStream = new ReadableStream({
            start(controller) {
              const data = JSON.stringify({ content: validation.reason, done: false });
              controller.enqueue(encoder.encode(`data: ${data}\n\n`));
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`));
              controller.close();
            },
          });
          
          return new Response(errorStream, {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              'Connection': 'keep-alive',
            },
          });
        } else {
          return new Response(
            JSON.stringify({ content: validation.reason }),
            { headers: { 'Content-Type': 'application/json' } }
          );
        }
      }
    }

    const client = createVLMClient();

    if (stream) {
      const encoder = new TextEncoder();

      const readableStream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of client.chatStream(messages)) {
              const data = JSON.stringify({ content: chunk, done: false });
              controller.enqueue(encoder.encode(`data: ${data}\n\n`));
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`));
            controller.close();
          } catch (error) {
            console.error('Stream error:', error);
            const isConnection = isConnectionError(error);
            const errorMessage = isConnection
              ? createErrorMessage(true)
              : (error instanceof Error ? error.message : 'Stream error occurred');

            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ error: errorMessage, done: true })}\n\n`)
            );
            controller.close();
          }
        },
      });

      return new Response(readableStream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    } else {
      const response = await client.chat(messages);
      return new Response(
        JSON.stringify({ content: response }),
        { headers: { 'Content-Type': 'application/json' } }
      );
    }
  } catch (error) {
    console.error('Chat API error:', error);

    if (isConnectionError(error)) {
      return new Response(
        JSON.stringify({ error: createErrorMessage(true) }),
        { status: 503, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const errorMessage =
      error instanceof Error ? error.message : 'Internal server error';

    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
