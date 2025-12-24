'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Trash2 } from 'lucide-react';
import { Message } from './Message';
import { MessageInput, MessageInputRef } from './MessageInput';
import { QubitIcon } from './QubitIcon';
import { LoadingStatus } from './LoadingStatus';
import { SYSTEM_PROMPT } from '@/config/constants';
import { resizeImageForInference, fetchAndResizeImage } from '@/lib/utils/image';
import { postProcessResponse } from '@/lib/utils/response';
import type { Message as MessageType, DatasetExample } from '@/types';

interface ChatInterfaceProps {
  selectedExample?: DatasetExample | null;
  onExampleUsed?: () => void;
}

export function ChatInterface({ selectedExample, onExampleUsed }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasStartedStreaming, setHasStartedStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<MessageInputRef>(null);
  const processedExampleRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    if (
      selectedExample &&
      selectedExample.id !== processedExampleRef.current
    ) {
      processedExampleRef.current = selectedExample.id;
      inputRef.current?.setContent(
        selectedExample.question,
        selectedExample.imageUrl
      );
      onExampleUsed?.();
    }
  }, [selectedExample, onExampleUsed]);

  const handleSendMessage = async (
    content: string,
    imageUrl?: string,
    imageBase64?: string
  ) => {
    if (!content.trim() && !imageUrl && !imageBase64) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const userMessage: MessageType = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      imageUrl,
      imageBase64,
      timestamp: new Date(),
    };

    const assistantMessageId = crypto.randomUUID();
    const loadingMessage: MessageType = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setIsLoading(true);
    setHasStartedStreaming(false);

    try {
      let imageData: string | undefined;

      if (imageBase64) {
        try {
          imageData = await resizeImageForInference(`data:image/jpeg;base64,${imageBase64}`);
        } catch (e) {
          console.error('Failed to resize image:', e);
          imageData = imageBase64;
        }
      } else if (imageUrl) {
        try {
          imageData = await fetchAndResizeImage(imageUrl);
        } catch (e) {
          console.error('Failed to fetch and resize image:', e);
        }
      }

      const userContent = imageData
        ? [
          { type: 'text', text: content },
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageData}` } },
        ]
        : content;

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            ...messages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            { role: 'user', content: userContent },
          ],
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Request failed');
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          const jsonStr = trimmed.slice(6);
          try {
            const data = JSON.parse(jsonStr);

            if (data.error) {
              throw new Error(data.error);
            }

            if (data.content) {
              // First content received - streaming has started
              if (fullContent === '') {
                setHasStartedStreaming(true);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId ? { ...m, isLoading: false } : m
                  )
                );
              }
              
              fullContent += data.content;
              const processedContent = postProcessResponse(fullContent);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: processedContent }
                    : m
                )
              );
            }
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      }

      const finalContent = postProcessResponse(fullContent);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? { ...m, content: finalContent }
            : m
        )
      );
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        return;
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? {
              ...m,
              content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
              isLoading: false,
            }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setHasStartedStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const handleClearChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setMessages([]);
    inputRef.current?.clear();
    processedExampleRef.current = null;
  };

  const handleCopyCode = (code: string) => {
    console.log('Code copied:', code.substring(0, 50) + '...');
  };

  return (
    <div className="flex flex-col h-full">
      {messages.length > 0 && (
        <div className="flex justify-end px-4 pt-2">
          <button
            onClick={handleClearChat}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-500 
                       hover:text-zinc-300 hover:bg-zinc-800/50 rounded-md transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear chat
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-16 h-16 mb-5 rounded-xl bg-zinc-800/80 border border-teal-700/30 flex items-center justify-center">
              <QubitIcon size={32} className="text-teal-400" />
            </div>
            <h2 className="text-xl font-semibold text-zinc-200 mb-2">
              Quantum Assistant
            </h2>
            <p className="text-zinc-500 max-w-md mb-8 text-sm leading-relaxed">
              Ask questions about quantum computing, generate Qiskit code, or upload circuit diagrams for analysis.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-xl">
              {[
                { label: 'Circuits', text: 'Create a Bell state circuit' },
                { label: 'Concepts', text: 'Explain Bloch sphere representation' },
                { label: 'Algorithms', text: 'Implement VQE algorithm' },
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => inputRef.current?.setContent(suggestion.text)}
                  className="bg-zinc-800/60 hover:bg-zinc-800 border border-zinc-700/50 hover:border-zinc-600/50 rounded-lg p-4 text-left group transition-all"
                >
                  <span className="text-[10px] font-mono text-teal-500/80 mb-2 block uppercase tracking-wider">
                    {suggestion.label}
                  </span>
                  <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors">
                    {suggestion.text}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <Message
              key={message.id}
              message={message}
              onCopyCode={handleCopyCode}
              loadingStatus={
                message.isLoading ? (
                  <LoadingStatus
                    isLoading={isLoading}
                    hasStartedStreaming={hasStartedStreaming}
                  />
                ) : undefined
              }
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-zinc-800/80">
        <MessageInput ref={inputRef} onSend={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}
