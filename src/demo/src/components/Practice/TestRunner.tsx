'use client';

import { useState } from 'react';
import {
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Loader2,
  Terminal,
  FileCode,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { TestResult } from '@/types';

interface TestRunnerProps {
  userCode: string;
  testCode: string;
  entryPoint: string;
  onTestComplete: (result: TestResult) => void;
  initialResult?: TestResult | null;
}

export function TestRunner({
  userCode,
  testCode,
  entryPoint,
  onTestComplete,
  initialResult = null,
}: TestRunnerProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<TestResult | null>(initialResult);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showTraceback, setShowTraceback] = useState(false);

  const runTests = async () => {
    if (isRunning) return;

    setIsRunning(true);
    setResult(null);
    setShowTraceback(false);

    try {
      const response = await fetch('/api/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userCode,
          testCode,
          entryPoint,
          timeout: 30,
        }),
      });

      const data = await response.json();
      setResult(data);
      onTestComplete(data);
      // Auto-expand on failure
      if (!data.passed) {
        setIsExpanded(true);
      }
    } catch (error) {
      const errorResult: TestResult = {
        passed: false,
        total: 0,
        failed: 0,
        details: [],
        executionTime: 0,
        error: error instanceof Error ? error.message : 'Failed to run tests',
      };
      setResult(errorResult);
      onTestComplete(errorResult);
      setIsExpanded(true);
    } finally {
      setIsRunning(false);
    }
  };

  const hasDetails = result && (result.error || result.details.length > 0 || result.traceback || result.output);
  const passedCount = result?.details?.filter(t => t.passed).length ?? 0;
  const totalCount = result?.total ?? result?.details?.length ?? 0;

  return (
    <div className="flex flex-col">
      {/* Compact header with Run button and status */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/50">
        <div className="flex items-center gap-3">
          <button
            onClick={runTests}
            disabled={isRunning || !userCode.trim()}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-md font-medium text-sm transition-all',
              isRunning || !userCode.trim()
                ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                : 'bg-teal-600 hover:bg-teal-500 text-white'
            )}
          >
            {isRunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                Run Tests
              </>
            )}
          </button>

          {result && (
            <div className="flex items-center gap-2 text-sm">
              {result.passed ? (
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Passed</span>
                  {totalCount > 0 && (
                    <span className="text-emerald-400/70 text-xs font-normal">
                      ({passedCount}/{totalCount} tests)
                    </span>
                  )}
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-red-400">
                  <XCircle className="w-4 h-4" />
                  <span>Failed</span>
                  {totalCount > 0 && (
                    <span className="text-red-400/70 text-xs font-normal">
                      ({passedCount}/{totalCount} tests)
                    </span>
                  )}
                </span>
              )}
              <span className="flex items-center gap-1 text-zinc-500 text-xs">
                <Clock className="w-3 h-3" />
                {result.executionTime}ms
              </span>
            </div>
          )}
        </div>

        {result && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {isExpanded ? 'Hide' : 'Show'} details
            {isExpanded ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        )}
      </div>

      {/* Expandable details section */}
      {isExpanded && result && (
        <div className="px-4 py-3 bg-zinc-900/30 border-t border-zinc-800/50 max-h-64 overflow-y-auto">
          {/* Summary for passed tests */}
          {result.passed && !result.error && (
            <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-800/30 mb-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm text-emerald-300 font-medium">
                  All {totalCount} test{totalCount !== 1 ? 's' : ''} passed successfully!
                </span>
              </div>
            </div>
          )}

          {/* Main error message */}
          {result.error && (
            <div className="p-3 rounded-lg bg-red-950/30 border border-red-800/40 mb-3">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <pre className="text-xs text-red-200/80 whitespace-pre-wrap font-mono flex-1 break-all">
                  {result.error}
                </pre>
              </div>
            </div>
          )}

          {/* Traceback toggle and display */}
          {result.traceback && result.traceback !== result.error && (
            <div className="mb-3">
              <button
                onClick={() => setShowTraceback(!showTraceback)}
                className="flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-200 transition-colors mb-2"
              >
                <FileCode className="w-3.5 h-3.5" />
                {showTraceback ? 'Hide' : 'Show'} full traceback
                {showTraceback ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>
              {showTraceback && (
                <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-700/50 overflow-x-auto">
                  <pre className="text-[11px] text-zinc-300 whitespace-pre font-mono">
                    {result.traceback}
                  </pre>
                </div>
              )}
            </div>
          )}

          {/* Output display */}
          {result.output && (
            <div className="mb-3">
              <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1.5">
                <Terminal className="w-3.5 h-3.5" />
                <span>Output</span>
              </div>
              <div className="p-2 rounded-lg bg-zinc-900/60 border border-zinc-800/50">
                <pre className="text-[11px] text-zinc-400 whitespace-pre-wrap font-mono">
                  {result.output}
                </pre>
              </div>
            </div>
          )}

          {/* Test details - always show */}
          {result.details && result.details.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs text-zinc-500 mb-2">Test Results:</div>
              {result.details.map((test, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'p-2 rounded-md border text-xs',
                    test.passed
                      ? 'bg-emerald-950/20 border-emerald-800/30'
                      : 'bg-red-950/20 border-red-800/30'
                  )}
                >
                  <div className="flex items-center gap-2">
                    {test.passed ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-red-400" />
                    )}
                    <span
                      className={clsx(
                        'font-medium',
                        test.passed ? 'text-emerald-300' : 'text-red-300'
                      )}
                    >
                      {test.name}
                    </span>
                  </div>

                  {!test.passed && (test.expected || test.actual || test.error) && (
                    <div className="ml-5 mt-1 space-y-0.5 font-mono text-[11px]">
                      {test.expected && (
                        <p className="text-zinc-400">
                          <span className="text-zinc-500">Expected: </span>
                          <span className="text-emerald-300">{test.expected}</span>
                        </p>
                      )}
                      {test.actual && (
                        <p className="text-zinc-400">
                          <span className="text-zinc-500">Actual: </span>
                          <span className="text-red-300">{test.actual}</span>
                        </p>
                      )}
                      {test.error && !result.error && (
                        <p className="text-red-300/80">{test.error}</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
