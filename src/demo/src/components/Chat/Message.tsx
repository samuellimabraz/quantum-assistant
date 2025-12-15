'use client';

import { useMemo, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { InlineMath, BlockMath } from 'react-katex';
import { Copy, Check, Play, Square, Edit2, X } from 'lucide-react';
import Editor from '@monaco-editor/react';
import { clsx } from 'clsx';
import { QubitIcon } from './QubitIcon';
import { ExecutionResult, ExecutionResultData } from './ExecutionResult';
import type { Message as MessageType } from '@/types';

interface MessageProps {
  message: MessageType;
  onCopyCode?: (code: string) => void;
}

// Custom matte dark theme - muted, professional colors
const customTheme: { [key: string]: React.CSSProperties } = {
  'code[class*="language-"]': {
    color: '#d4d4d8',
    background: 'none',
    fontFamily: "'JetBrains Mono', Consolas, Monaco, 'Andale Mono', monospace",
    fontSize: '0.875rem',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.6',
    tabSize: 4,
    hyphens: 'none',
  },
  'pre[class*="language-"]': {
    color: '#d4d4d8',
    background: '#18181b',
    fontFamily: "'JetBrains Mono', Consolas, Monaco, 'Andale Mono', monospace",
    fontSize: '0.875rem',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    lineHeight: '1.6',
    tabSize: 4,
    hyphens: 'none',
    padding: '1rem',
    margin: '0',
    overflow: 'auto',
    borderRadius: '0.5rem',
  },
  comment: { color: '#71717a' },
  prolog: { color: '#71717a' },
  doctype: { color: '#71717a' },
  cdata: { color: '#71717a' },
  punctuation: { color: '#a1a1aa' },
  namespace: { opacity: 0.7 },
  property: { color: '#f0abfc' },
  tag: { color: '#f0abfc' },
  boolean: { color: '#c4b5fd' },
  number: { color: '#c4b5fd' },
  constant: { color: '#c4b5fd' },
  symbol: { color: '#c4b5fd' },
  deleted: { color: '#fca5a5' },
  selector: { color: '#86efac' },
  'attr-name': { color: '#fcd34d' },
  string: { color: '#86efac' },
  char: { color: '#86efac' },
  builtin: { color: '#86efac' },
  inserted: { color: '#86efac' },
  operator: { color: '#f0abfc' },
  entity: { color: '#fcd34d', cursor: 'help' },
  url: { color: '#67e8f9' },
  '.language-css .token.string': { color: '#67e8f9' },
  '.style .token.string': { color: '#67e8f9' },
  variable: { color: '#d4d4d8' },
  atrule: { color: '#93c5fd' },
  'attr-value': { color: '#86efac' },
  function: { color: '#93c5fd' },
  'class-name': { color: '#93c5fd' },
  keyword: { color: '#c4b5fd' },
  regex: { color: '#fcd34d' },
  important: { color: '#fcd34d', fontWeight: 'bold' },
  bold: { fontWeight: 'bold' },
  italic: { fontStyle: 'italic' },
};

function isPythonCode(language: string, code: string): boolean {
  if (language === 'python') return true;

  const pythonPatterns = [
    /^from\s+\w+\s+import/m,
    /^import\s+\w+/m,
    /^def\s+\w+\s*\(/m,
    /^class\s+\w+/m,
    /QuantumCircuit/,
    /qiskit/i,
  ];

  return pythonPatterns.some(p => p.test(code));
}

function CodeBlock({
  language,
  code: initialCode,
  onCopy,
}: {
  language: string;
  code: string;
  onCopy?: (code: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedCode, setEditedCode] = useState(initialCode);
  const [executionResult, setExecutionResult] = useState<ExecutionResultData | null>(null);

  // The code to use (edited or original)
  const code = isEditing ? editedCode : initialCode;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    onCopy?.(code);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExecute = useCallback(async () => {
    if (isExecuting) return;

    setIsExecuting(true);
    setExecutionResult(null);

    try {
      const response = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, timeout: 30 }),
      });

      const result = await response.json();
      setExecutionResult(result);
    } catch (error) {
      setExecutionResult({
        success: false,
        output: '',
        error: error instanceof Error ? error.message : 'Execution failed',
        executionTime: 0,
        hasCircuitOutput: false,
      });
    } finally {
      setIsExecuting(false);
    }
  }, [code, isExecuting]);

  const handleStopExecution = () => {
    setIsExecuting(false);
  };

  const handleToggleEdit = () => {
    if (isEditing) {
      // Exiting edit mode - keep the edited code
      setIsEditing(false);
    } else {
      // Entering edit mode
      setEditedCode(initialCode);
      setIsEditing(true);
    }
  };

  const handleCancelEdit = () => {
    setEditedCode(initialCode);
    setIsEditing(false);
  };

  const detectedLanguage = language || detectLanguage(code);
  const canExecute = isPythonCode(detectedLanguage, code);
  
  // Calculate editor height based on line count
  const lineCount = code.split('\n').length;
  const editorHeight = Math.min(Math.max(lineCount * 20 + 32, 100), 400);

  return (
    <div className="relative group my-3 code-block-wrapper">
      {/* Action buttons */}
      <div className="absolute right-2 top-2 flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
          {detectedLanguage || 'code'}
        </span>

        {/* Edit toggle */}
        <button
          onClick={isEditing ? handleCancelEdit : handleToggleEdit}
          className={clsx(
            'flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-all',
            isEditing
              ? 'bg-amber-600/20 text-amber-400 hover:bg-amber-600/30'
              : 'bg-zinc-700/50 text-zinc-400 hover:bg-zinc-700'
          )}
          title={isEditing ? 'Cancel editing' : 'Edit code'}
        >
          {isEditing ? (
            <>
              <X className="w-3 h-3" />
              <span>Cancel</span>
            </>
          ) : (
            <>
              <Edit2 className="w-3 h-3" />
              <span>Edit</span>
            </>
          )}
        </button>

        {canExecute && (
          <button
            onClick={isExecuting ? handleStopExecution : handleExecute}
            disabled={isExecuting}
            className={clsx(
              'flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-all',
              isExecuting
                ? 'bg-red-600/20 text-red-400 hover:bg-red-600/30'
                : 'bg-teal-600/20 text-teal-400 hover:bg-teal-600/30'
            )}
            title={isExecuting ? 'Stop execution' : 'Run code'}
          >
            {isExecuting ? (
              <>
                <Square className="w-3 h-3" />
                <span>Stop</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3" />
                <span>Run</span>
              </>
            )}
          </button>
        )}

        <button
          onClick={handleCopy}
          className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <Check className="w-3.5 h-3.5 text-emerald-400" />
          ) : (
            <Copy className="w-3.5 h-3.5 text-zinc-400" />
          )}
        </button>
      </div>

      {isEditing ? (
        // Monaco Editor for editing
        <div 
          className="rounded-lg overflow-hidden border border-amber-600/30"
          style={{ height: editorHeight }}
        >
          <Editor
            height="100%"
            language={detectedLanguage || 'python'}
            value={editedCode}
            onChange={(value) => setEditedCode(value || '')}
            theme="vs-dark"
            options={{
              fontSize: 14,
              fontFamily: "'JetBrains Mono', Consolas, Monaco, monospace",
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              lineNumbers: 'on',
              glyphMargin: false,
              folding: false,
              lineDecorationsWidth: 8,
              lineNumbersMinChars: 3,
              padding: { top: 12, bottom: 12 },
              renderLineHighlight: 'line',
              tabSize: 4,
              insertSpaces: true,
              wordWrap: 'on',
              automaticLayout: true,
            }}
          />
        </div>
      ) : (
        // Static code display
        <SyntaxHighlighter
          style={customTheme}
          language={detectedLanguage || 'python'}
          PreTag="div"
          customStyle={{
            margin: 0,
            borderRadius: '0.5rem',
            background: '#18181b',
            padding: '1rem',
            fontSize: '0.875rem',
            border: '1px solid #27272a',
            lineHeight: '1.6',
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
      )}

      {/* Execution result */}
      {(isExecuting || executionResult) && (
        <ExecutionResult
          result={executionResult || { success: false, output: '', error: '', executionTime: 0 }}
          isLoading={isExecuting}
        />
      )}
    </div>
  );
}

function detectLanguage(code: string): string {
  const pythonPatterns = [
    /^from\s+\w+\s+import/m,
    /^import\s+\w+/m,
    /^def\s+\w+\s*\(/m,
    /^class\s+\w+/m,
    /QuantumCircuit/,
    /qiskit/i,
    /\.measure/,
    /numpy|np\./,
    /print\s*\(/,
  ];

  if (pythonPatterns.some((p) => p.test(code))) {
    return 'python';
  }

  const jsPatterns = [
    /^const\s+\w+\s*=/m,
    /^let\s+\w+\s*=/m,
    /^function\s+\w+/m,
    /=>\s*{/,
    /console\.log/,
  ];

  if (jsPatterns.some((p) => p.test(code))) {
    return 'javascript';
  }

  const bashPatterns = [/^\$\s+/m, /^#!\/bin\/(ba)?sh/m, /\|\s*grep/, /apt-get|pip\s+install/];

  if (bashPatterns.some((p) => p.test(code))) {
    return 'bash';
  }

  return 'python';
}

function looksLikeCode(text: string): boolean {
  // Multi-line code indicators
  if (text.includes('\n')) {
    const codeIndicators = [
      /^from\s+/m,
      /^import\s+/m,
      /^def\s+/m,
      /^class\s+/m,
      /^\s*return\s+/m,
      /QuantumCircuit/,
      /Parameter\(/,
      /\.\w+\([^)]*\)/m, // Method calls like qc.h(), qc.cx()
    ];
    return codeIndicators.some((p) => p.test(text));
  }

  // Single-line code indicators for function completion responses
  const singleLinePatterns = [
    /^return\s+\w+/,                          // return circuit.control(...)
    /^\w+\s*=\s*\w+\([^)]*\)/,                // theta = Parameter("theta")
    /^\w+\.\w+\([^)]*\)$/,                    // circuit.control(num_ctrl_qubits)
    /\w+\s*=\s*\w+\([^)]*\)(?:\s+\w+\.|\s+\w+\s*=)/, // Multiple statements
    /QuantumCircuit\(/,
    /Parameter\(/,
    /\.control\(/,
    /\.measure\(/,
  ];
  return singleLinePatterns.some((p) => p.test(text.trim()));
}

export function Message({ message, onCopyCode }: MessageProps) {
  const isUser = message.role === 'user';
  const isLoading = message.isLoading;

  const avatar = useMemo(() => {
    if (isUser) {
      return (
        <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center border border-zinc-700/50">
          <span className="text-[10px] font-bold text-zinc-400 font-mono">YOU</span>
        </div>
      );
    }
    return (
      <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center border border-teal-700/40">
        <QubitIcon size={18} className="text-teal-400" />
      </div>
    );
  }, [isUser]);

  const imageSource =
    message.imageUrl || (message.imageBase64 ? `data:image/jpeg;base64,${message.imageBase64}` : null);

  const processedContent = useMemo(() => {
    let content = message.content;

    // Convert non-standard math delimiters to standard LaTeX format
    // Display math: [ ... ] containing LaTeX → $$ ... $$
    content = content.replace(
      /\[\s*(\\[a-zA-Z][^\]]*)\s*\]/g,
      (match, inner) => `\n$$\n${inner.trim()}\n$$\n`
    );

    // Inline math with \(...\) → $...$
    content = content.replace(
      /\\\(([^)]+)\\\)/g,
      (match, inner) => `$${inner}$`
    );

    // Inline math: (expression) containing LaTeX → $...$
    // Match parentheses containing backslash commands but not nested parens
    content = content.replace(
      /\(([^()]*(?:\\[a-zA-Z{}^_]|\\frac|\\sqrt|\\sum|\\exp|\\left|\\right|\\bigl|\\bigr|\\Bigl|\\Bigr|\|[01]\\rangle)[^()]*)\)/g,
      (match, inner) => {
        // Only convert if it really looks like math
        if (/\\[a-zA-Z]/.test(inner) || /\|[01n]\\rangle/.test(inner)) {
          return `$${inner}$`;
        }
        return match;
      }
    );

    // Code detection for non-markdown responses
    if (!content.includes('```') && !content.includes('$$') && !content.includes('$') && looksLikeCode(content)) {
      content = content
        .replace(/(\w+\s*=\s*\w+\([^)]*\))\s+(\w+\.)/g, '$1\n$2')
        .replace(/(\w+\.[a-z_]+\([^)]*\))\s+(\w+\.)/g, '$1\n$2');

      content = '```python\n' + content + '\n```';
    }

    return content;
  }, [message.content]);

  return (
    <div className={clsx('flex gap-3 animate-in', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <div className="flex-shrink-0">{avatar}</div>

      <div className={clsx('flex-1 max-w-[85%]', isUser ? 'flex flex-col items-end' : '')}>
        {imageSource && (
          <div className="mb-2 max-w-xs">
            <img
              src={imageSource}
              alt="Attached image"
              className="rounded-lg border border-zinc-700/50 max-h-64 object-contain bg-zinc-900"
            />
          </div>
        )}

        <div
          className={clsx(
            'rounded-xl px-4 py-3',
            isUser
              ? 'bg-teal-700/80 text-white rounded-tr-sm'
              : 'bg-zinc-800/90 border border-zinc-700/50 rounded-tl-sm'
          )}
        >
          {isLoading ? (
            <div className="typing-indicator py-1">
              <span />
              <span />
              <span />
            </div>
          ) : (
            <div className={clsx('markdown-content', isUser && 'text-white/90')}>
              <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const code = String(children).replace(/\n$/, '');

                    // Check if this is a math block (from remark-math)
                    if (className === 'language-math' || className === 'math-inline') {
                      try {
                        return <InlineMath math={code} />;
                      } catch {
                        return <code className="text-red-400">{code}</code>;
                      }
                    }

                    const isBlock = match || code.includes('\n') || looksLikeCode(code);

                    if (isBlock) {
                      return <CodeBlock language={match?.[1] || ''} code={code} onCopy={onCopyCode} />;
                    }

                    return (
                      <code className={clsx('bg-zinc-700/50 px-1.5 py-0.5 rounded text-sm', className)} {...props}>
                        {children}
                      </code>
                    );
                  },
                  pre({ children }) {
                    return <>{children}</>;
                  },
                  // Handle math blocks from remark-math
                  span({ className, children, ...props }) {
                    if (className === 'math math-inline') {
                      try {
                        const math = String(children);
                        return <InlineMath math={math} />;
                      } catch {
                        return <span className="text-red-400">{children}</span>;
                      }
                    }
                    return <span className={className} {...props}>{children}</span>;
                  },
                  div({ className, children, ...props }) {
                    if (className === 'math math-display') {
                      try {
                        const math = String(children);
                        return <BlockMath math={math} />;
                      } catch {
                        return <div className="text-red-400">{children}</div>;
                      }
                    }
                    return <div className={className} {...props}>{children}</div>;
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            </div>
          )}
        </div>

        <span className="text-xs text-zinc-500 mt-1 px-2">
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}
