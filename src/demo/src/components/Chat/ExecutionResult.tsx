'use client';

import { useState } from 'react';
import { 
  CheckCircle2, 
  XCircle, 
  Clock, 
  ChevronDown, 
  ChevronUp,
  Terminal,
  AlertTriangle,
  Copy,
  Check
} from 'lucide-react';
import { clsx } from 'clsx';

export interface ExecutionResultData {
  success: boolean;
  output: string;
  error: string;
  executionTime: number;
  hasCircuitOutput?: boolean;
}

interface ExecutionResultProps {
  result: ExecutionResultData;
  isLoading?: boolean;
}

export function ExecutionResult({ result, isLoading }: ExecutionResultProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [copied, setCopied] = useState(false);
  
  const hasOutput = result.output.trim().length > 0;
  const hasError = result.error.trim().length > 0;
  const outputToShow = hasError ? result.error : result.output;
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(outputToShow);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  if (isLoading) {
    return (
      <div className="mt-3 rounded-lg border border-zinc-700/50 bg-zinc-900/50 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border-b border-zinc-700/50">
          <div className="w-4 h-4 border-2 border-teal-500/30 border-t-teal-500 rounded-full animate-spin" />
          <span className="text-xs font-medium text-zinc-400">Executing code...</span>
        </div>
        <div className="p-3">
          <div className="flex items-center gap-2 text-zinc-500">
            <Terminal className="w-4 h-4" />
            <span className="text-sm">Running Python with Qiskit...</span>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className={clsx(
      'mt-3 rounded-lg border overflow-hidden transition-all duration-200',
      result.success
        ? 'border-emerald-600/30 bg-emerald-950/20'
        : 'border-red-600/30 bg-red-950/20'
    )}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={clsx(
          'w-full flex items-center justify-between px-3 py-2 transition-colors',
          result.success 
            ? 'bg-emerald-900/30 hover:bg-emerald-900/40' 
            : 'bg-red-900/30 hover:bg-red-900/40'
        )}
      >
        <div className="flex items-center gap-2">
          {result.success ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : (
            <XCircle className="w-4 h-4 text-red-400" />
          )}
          <span className={clsx(
            'text-xs font-medium',
            result.success ? 'text-emerald-300' : 'text-red-300'
          )}>
            {result.success ? 'Execution Successful' : 'Execution Failed'}
          </span>
          
          <span className="flex items-center gap-1 text-xs text-zinc-500 ml-2">
            <Clock className="w-3 h-3" />
            {result.executionTime}ms
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          {(hasOutput || hasError) && (
            <span className="text-xs text-zinc-500">
              {isExpanded ? 'Hide' : 'Show'} output
            </span>
          )}
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-zinc-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          )}
        </div>
      </button>
      
      {/* Output */}
      {isExpanded && (hasOutput || hasError) && (
        <div className="relative">
          <div className="absolute right-2 top-2 z-10">
            <button
              onClick={handleCopy}
              className="p-1.5 rounded bg-zinc-800/80 hover:bg-zinc-700 transition-colors"
              title="Copy output"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5 text-zinc-400" />
              )}
            </button>
          </div>
          
          <div className={clsx(
            'p-3 font-mono text-sm overflow-x-auto',
            result.success ? 'bg-zinc-900/50' : 'bg-zinc-900/50'
          )}>
            {hasError && (
              <div className="flex items-start gap-2 mb-2 text-red-400">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <pre className="whitespace-pre-wrap break-words text-red-300">{result.error}</pre>
              </div>
            )}
            
            {hasOutput && (
              <pre className={clsx(
                'whitespace-pre-wrap break-words',
                result.hasCircuitOutput ? 'text-teal-300' : 'text-zinc-300'
              )}>
                {result.output}
              </pre>
            )}
            
            {!hasOutput && !hasError && result.success && (
              <span className="text-zinc-500 italic">
                Code executed successfully with no output
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

