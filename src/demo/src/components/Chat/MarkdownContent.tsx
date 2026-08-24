'use client';

import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { clsx } from 'clsx';
import { looksLikeCode, prepareMarkdownContent } from '@/lib/utils/response';

export interface CodeBlockRenderProps {
  language: string;
  code: string;
}

interface MarkdownContentProps {
  content: string;
  className?: string;
  renderCodeBlock?: (props: CodeBlockRenderProps) => ReactNode;
}

function isMathClass(className?: string): boolean {
  if (!className) return false;
  return (
    className.includes('math') ||
    className.includes('language-math') ||
    className.includes('katex')
  );
}

export function MarkdownContent({ content, className, renderCodeBlock }: MarkdownContentProps) {
  const prepared = prepareMarkdownContent(content);

  return (
    <div className={clsx('markdown-content', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore', errorColor: '#a1a1aa' }]]}
        components={{
          code({ className: codeClassName, children, ...props }) {
            if (isMathClass(codeClassName)) {
              return (
                <code className={codeClassName} {...props}>
                  {children}
                </code>
              );
            }

            const match = /language-(\w+)/.exec(codeClassName || '');
            const code = String(children).replace(/\n$/, '');
            const isBlock = Boolean(match) || code.includes('\n') || looksLikeCode(code);

            if (isBlock) {
              if (renderCodeBlock) {
                return <>{renderCodeBlock({ language: match?.[1] || '', code })}</>;
              }
              return (
                <pre className="my-3 overflow-x-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-300">
                  <code>{code}</code>
                </pre>
              );
            }

            return (
              <code
                className={clsx('bg-zinc-700/50 px-1.5 py-0.5 rounded text-sm', codeClassName)}
                {...props}
              >
                {children}
              </code>
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
        }}
      >
        {prepared}
      </ReactMarkdown>
    </div>
  );
}
