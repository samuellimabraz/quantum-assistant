'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Sparkles, Send, Loader2, Trash2, ChevronLeft, Copy, Check, Play } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { clsx } from 'clsx';
import type { CodingProblem } from '@/types';
import { postProcessResponse } from '@/lib/utils/response';
import { LoadingStatus } from '../Chat/LoadingStatus';

interface AIHelperProps {
  problem: CodingProblem | null;
  userCode: string;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onApplyCode?: (code: string) => void;
}

interface HelperMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// Custom matte dark theme - matching Chat component
const customTheme: { [key: string]: React.CSSProperties } = {
  'code[class*="language-"]': {
    color: '#d4d4d8',
    background: 'none',
    fontFamily: "'JetBrains Mono', Consolas, Monaco, monospace",
    fontSize: '0.8rem',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.5',
    tabSize: 4,
  },
  'pre[class*="language-"]': {
    color: '#d4d4d8',
    background: '#18181b',
    fontFamily: "'JetBrains Mono', Consolas, Monaco, monospace",
    fontSize: '0.8rem',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.5',
    tabSize: 4,
    padding: '0.75rem',
    margin: '0',
    overflow: 'auto',
    borderRadius: '0.375rem',
  },
  comment: { color: '#71717a' },
  prolog: { color: '#71717a' },
  doctype: { color: '#71717a' },
  punctuation: { color: '#a1a1aa' },
  property: { color: '#f0abfc' },
  tag: { color: '#f0abfc' },
  boolean: { color: '#c4b5fd' },
  number: { color: '#c4b5fd' },
  constant: { color: '#c4b5fd' },
  symbol: { color: '#c4b5fd' },
  selector: { color: '#86efac' },
  string: { color: '#86efac' },
  char: { color: '#86efac' },
  builtin: { color: '#86efac' },
  operator: { color: '#f0abfc' },
  variable: { color: '#d4d4d8' },
  function: { color: '#93c5fd' },
  'class-name': { color: '#93c5fd' },
  keyword: { color: '#c4b5fd' },
  regex: { color: '#fcd34d' },
  important: { color: '#fcd34d', fontWeight: 'bold' },
};

const HELPER_PROMPT = `You are a helpful coding assistant for quantum computing practice problems using Qiskit.

Your role is to:
1. Provide hints and guidance without giving away the complete solution
2. Explain quantum computing concepts when asked
3. Help debug code issues
4. Suggest improvements to the user's approach

Guidelines:
- Be encouraging and educational
- Give progressively more detailed hints if the user is stuck
- Focus on teaching, not just solving
- Reference Qiskit 2.0 best practices
- Keep responses concise and focused

Current problem context will be provided. Help the user learn while they solve the problem themselves.`;

function getSolvePrompt(problemType: 'function_completion' | 'code_generation') {
  if (problemType === 'function_completion') {
    return `You are a quantum computing expert using Qiskit.

Your task is to provide ONLY the code lines that complete the function body. Do NOT include the function signature/definition - just the implementation lines that go inside the function.

Guidelines:
- Provide ONLY the implementation code (the lines after the function definition)
- Do NOT repeat the function signature like "def function_name(...):"
- Include proper indentation for the function body
- Use Qiskit 2.0 best practices
- Add brief comments for complex steps

Example: If the function is:
\`\`\`python
def create_bell_state():
    """Create a Bell state circuit."""
    pass
\`\`\`

You should respond with ONLY:
\`\`\`python
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
\`\`\`

Format your response with ONLY the implementation code in a Python code block.`;
  }

  return `You are a quantum computing expert using Qiskit.

Your task is to provide a complete, working solution for the given problem.

Guidelines:
- Provide a complete, executable Python solution
- Include all necessary imports
- Use Qiskit 2.0 best practices
- Include brief comments explaining key steps
- Make sure the solution passes the provided tests

Format your response with the complete code in a Python code block.`;
}

function looksLikeCode(text: string): boolean {
  const codeIndicators = [
    /^from\s+/m,
    /^import\s+/m,
    /^def\s+/m,
    /^class\s+/m,
    /^\s*return\s+/m,
    /QuantumCircuit/,
    /Parameter\(/,
    /\.\w+\([^)]*\)/m,
    /^\s{4}/m, // Indented code
    /qc\.\w+/,
    /circuit\.\w+/,
  ];
  return codeIndicators.some((p) => p.test(text));
}

interface CodeBlockProps {
  code: string;
  language: string;
  onCopy: () => void;
  onApply?: () => void;
  copied: boolean;
}

function CodeBlock({ code, language, onCopy, onApply, copied }: CodeBlockProps) {
  return (
    <div className="relative group my-2">
      {/* Action buttons - always visible for better discoverability */}
      <div className="absolute right-2 top-2 flex items-center gap-1.5 z-10">
        <span className="text-[10px] text-zinc-500 bg-zinc-900/80 px-1.5 py-0.5 rounded">
          {language || 'python'}
        </span>

        <button
          onClick={onCopy}
          className="p-1 rounded bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <Check className="w-3 h-3 text-emerald-400" />
          ) : (
            <Copy className="w-3 h-3 text-zinc-400" />
          )}
        </button>

        {onApply && (
          <button
            onClick={onApply}
            className="flex items-center gap-1 px-1.5 py-1 rounded bg-teal-700/80 hover:bg-teal-600 text-teal-100 transition-colors text-[10px] font-medium"
            title="Apply code to editor"
          >
            <Play className="w-3 h-3" />
            Apply
          </button>
        )}
      </div>

      <SyntaxHighlighter
        style={customTheme}
        language={language || 'python'}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: '0.375rem',
          background: '#18181b',
          padding: '0.75rem',
          paddingTop: '2rem', // Space for buttons
          fontSize: '0.8rem',
          border: '1px solid #27272a',
          lineHeight: '1.5',
        }}
        codeTagProps={{
          style: {
            background: 'none',
            padding: 0,
          },
        }}
        wrapLongLines={false}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export function AIHelper({
  problem,
  userCode,
  isCollapsed,
  onToggleCollapse,
  onApplyCode,
}: AIHelperProps) {
  const [messages, setMessages] = useState<HelperMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hasStartedStreaming, setHasStartedStreaming] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    // Scroll only within the messages container, not the whole page
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    // Only scroll within the AI Helper panel, not the whole page
    requestAnimationFrame(() => {
      scrollToBottom();
    });
  }, [messages, scrollToBottom]);

  useEffect(() => {
    setMessages([]);
  }, [problem?.id]);

  // Fetch image as base64 for multimodal problems
  const fetchImageBase64 = async (imageUrl: string): Promise<string | null> => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          const base64Data = base64.split(',')[1] || base64;
          resolve(base64Data);
        };
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      });
    } catch {
      return null;
    }
  };

  const handleSendMessage = async (customMessage?: string, isSolveRequest = false) => {
    const messageText = customMessage || input.trim();
    if (!messageText || isLoading || !problem) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const userMessage: HelperMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };

    const assistantId = crypto.randomUUID();
    const loadingMessage: HelperMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInput('');
    setIsLoading(true);
    setHasStartedStreaming(false);

    try {
      // Build context message with problem info
      let contextMessage = `Problem: ${problem.question}`;

      if (!isSolveRequest && userCode) {
        contextMessage += `\n\nUser's current code:\n\`\`\`python\n${userCode || '# No code written yet'}\n\`\`\``;
      }

      contextMessage += `\n\nUser's request: ${messageText}`;

      // Select appropriate system prompt
      const systemPrompt = isSolveRequest
        ? getSolvePrompt(problem.type as 'function_completion' | 'code_generation')
        : HELPER_PROMPT;

      // Build messages array
      const apiMessages: Array<{
        role: 'system' | 'user' | 'assistant';
        content: string | Array<{ type: string; text?: string; image_url?: { url: string } }>;
      }> = [
          { role: 'system', content: systemPrompt },
          ...messages.map((m) => ({
            role: m.role as 'user' | 'assistant',
            content: m.content,
          })),
        ];

      // Handle multimodal problems - include image if available
      if (problem.imageUrl && problem.hasImage) {
        const imageBase64 = await fetchImageBase64(problem.imageUrl);
        if (imageBase64) {
          apiMessages.push({
            role: 'user',
            content: [
              { type: 'text', text: contextMessage },
              {
                type: 'image_url',
                image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
              },
            ],
          });
        } else {
          apiMessages.push({ role: 'user', content: contextMessage });
        }
      } else {
        apiMessages.push({ role: 'user', content: contextMessage });
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Request failed');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

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
            if (data.content) {
              // First content received - streaming has started
              if (fullContent === '') {
                setHasStartedStreaming(true);
              }
              fullContent += data.content;
              // Use postProcessResponse like ChatInterface does for proper formatting
              const processedContent = postProcessResponse(fullContent);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: processedContent } : m
                )
              );
            }
          } catch {
            continue;
          }
        }
      }

      // Apply final post-processing
      const finalContent = postProcessResponse(fullContent);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: finalContent } : m
        )
      );
    } catch (error) {
      if ((error as Error).name === 'AbortError') return;

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Error: ${error instanceof Error ? error.message : 'Failed'}` }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setHasStartedStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearMessages = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setMessages([]);
  };

  // Extract code blocks from message content, preserving indentation
  const extractCodeBlocks = (content: string): string[] => {
    const codeBlockRegex = /```(?:python)?\n?([\s\S]*?)```/g;
    const blocks: string[] = [];
    let match;
    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Preserve indentation - only trim trailing newlines, not leading whitespace
      const code = match[1].replace(/\n+$/, '');
      blocks.push(code);
    }
    return blocks;
  };

  const handleCopyCode = (code: string, messageId: string) => {
    navigator.clipboard.writeText(code);
    setCopiedId(messageId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleApplyCode = (code: string) => {
    if (onApplyCode) {
      onApplyCode(code);
    }
  };

  // Process content to add code blocks where needed
  const processContent = useCallback((content: string): string => {
    // If already has code blocks, return as-is
    if (content.includes('```')) {
      return content;
    }

    // If it looks like code, wrap it
    if (looksLikeCode(content)) {
      return '```python\n' + content + '\n```';
    }

    return content;
  }, []);

  // Collapsed view
  if (isCollapsed) {
    return (
      <button
        onClick={onToggleCollapse}
        className="h-full w-full flex flex-col items-center justify-center gap-2 bg-zinc-900/95 border-l border-zinc-800/80 hover:bg-zinc-800/50 transition-colors cursor-pointer"
        title="Expand AI Helper"
      >
        <Sparkles className="w-5 h-5 text-teal-500" />
        <span className="text-xs text-zinc-500 [writing-mode:vertical-lr] rotate-180 font-medium">
          AI Helper
        </span>
      </button>
    );
  }

  // Expanded view
  return (
    <div className="h-full flex flex-col bg-zinc-900/95 border-l border-zinc-800/80">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/80 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-teal-500" />
          <h3 className="font-semibold text-zinc-200 text-sm">AI Helper</h3>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="p-1.5 rounded-md hover:bg-zinc-800/50 transition-colors"
              title="Clear chat"
            >
              <Trash2 className="w-3.5 h-3.5 text-zinc-500" />
            </button>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-md hover:bg-zinc-800/50 transition-colors"
            title="Collapse"
          >
            <ChevronLeft className="w-4 h-4 text-zinc-500 rotate-180" />
          </button>
        </div>
      </div>

      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <Sparkles className="w-8 h-8 text-teal-500/50 mb-3" />
            <p className="text-sm text-zinc-500 mb-4">
              Need help? Ask for hints or get the solution.
            </p>
            {problem && (
              <div className="space-y-2 w-full">
                {[
                  { label: 'Give me a hint', isSolve: false },
                  { label: 'Explain the concept', isSolve: false },
                  { label: 'Solve it', isSolve: true },
                ].map(({ label, isSolve }) => (
                  <button
                    key={label}
                    onClick={() => handleSendMessage(label, isSolve)}
                    className={clsx(
                      'w-full text-left px-3 py-2 rounded-md text-xs transition-colors',
                      isSolve
                        ? 'bg-teal-900/40 hover:bg-teal-800/50 text-teal-300 hover:text-teal-200 border border-teal-700/30'
                        : 'bg-zinc-800/60 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((message) => {
            const processedContent = processContent(message.content);
            const codeBlocks = extractCodeBlocks(processedContent);
            const hasCode = codeBlocks.length > 0;

            return (
              <div
                key={message.id}
                className={clsx(
                  'flex flex-col',
                  message.role === 'user' ? 'items-end' : 'items-start'
                )}
              >
                <div
                  className={clsx(
                    'max-w-[95%] rounded-lg px-3 py-2 text-sm',
                    message.role === 'user'
                      ? 'bg-teal-700/60 text-white'
                      : 'bg-zinc-800/80 text-zinc-300'
                  )}
                >
                  {message.role === 'assistant' && !message.content ? (
                    <LoadingStatus
                      isLoading={isLoading}
                      hasStartedStreaming={hasStartedStreaming}
                    />
                  ) : (
                    <ReactMarkdown
                      components={{
                        code({ className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          const code = String(children).replace(/\n$/, '');
                          const isBlock = match || code.includes('\n') || looksLikeCode(code);

                          if (isBlock) {
                            return (
                              <CodeBlock
                                code={code}
                                language={match?.[1] || 'python'}
                                onCopy={() => handleCopyCode(code, message.id)}
                                onApply={onApplyCode ? () => handleApplyCode(code) : undefined}
                                copied={copiedId === message.id}
                              />
                            );
                          }

                          return (
                            <code className="bg-zinc-700/50 px-1 py-0.5 rounded text-xs" {...props}>
                              {children}
                            </code>
                          );
                        },
                        pre({ children }) {
                          return <>{children}</>;
                        },
                        p({ children }) {
                          return <p className="mb-2 last:mb-0">{children}</p>;
                        },
                        ul({ children }) {
                          return <ul className="list-disc ml-4 mb-2 space-y-1">{children}</ul>;
                        },
                        ol({ children }) {
                          return <ol className="list-decimal ml-4 mb-2 space-y-1">{children}</ol>;
                        },
                        li({ children }) {
                          return <li className="text-zinc-300">{children}</li>;
                        },
                        strong({ children }) {
                          return <strong className="font-semibold text-zinc-200">{children}</strong>;
                        },
                      }}
                    >
                      {processedContent}
                    </ReactMarkdown>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {problem && (
        <div className="p-3 border-t border-zinc-800/80 flex-shrink-0">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask for help..."
              disabled={isLoading}
              rows={1}
              className="flex-1 bg-zinc-800/60 border border-zinc-700/50 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-500 resize-none focus:outline-none focus:ring-1 focus:ring-teal-600/50 min-h-[40px] max-h-[100px]"
              style={{ height: 'auto' }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = `${Math.min(target.scrollHeight, 100)}px`;
              }}
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || isLoading}
              className={clsx(
                'p-2 rounded-lg transition-all',
                input.trim() && !isLoading
                  ? 'bg-teal-600 hover:bg-teal-500 text-white'
                  : 'bg-zinc-800 text-zinc-500'
              )}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
