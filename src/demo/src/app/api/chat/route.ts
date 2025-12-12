import { NextRequest } from 'next/server';
import { createVLMClient } from '@/lib/api/vlm-client';

export const maxDuration = 120;

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>;
}

interface ChatRequestBody {
  messages: ChatMessage[];
  stream?: boolean;
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
