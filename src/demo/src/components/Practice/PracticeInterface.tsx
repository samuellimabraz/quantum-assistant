'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import { ChevronLeft, ChevronRight, FileText, Lightbulb, Image as ImageIcon } from 'lucide-react';
import { clsx } from 'clsx';
import { CodeEditor } from './CodeEditor';
import { ProblemList } from './ProblemList';
import { TestRunner } from './TestRunner';
import { AIHelper } from './AIHelper';
import { TASK_LABELS, CATEGORY_LABELS } from '@/config/constants';
import { extractCodeFromResponse, normalizeIndentation } from '@/lib/utils/response';
import type { CodingProblem, TestResult } from '@/types';

interface PracticeInterfaceProps {
  className?: string;
}

export function PracticeInterface({ className }: PracticeInterfaceProps) {
  const [selectedProblem, setSelectedProblem] = useState<CodingProblem | null>(null);
  const [userCode, setUserCode] = useState('');
  const [solvedProblems, setSolvedProblems] = useState<Set<string>>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('solvedProblems');
      if (stored) {
        try {
          return new Set(JSON.parse(stored));
        } catch {
          return new Set();
        }
      }
    }
    return new Set();
  });

  const [isProblemListCollapsed, setIsProblemListCollapsed] = useState(false);
  const [isAIHelperCollapsed, setIsAIHelperCollapsed] = useState(true);
  const [problemListWidth, setProblemListWidth] = useState(320);
  const [aiHelperWidth, setAIHelperWidth] = useState(320);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('solvedProblems', JSON.stringify([...solvedProblems]));
    }
  }, [solvedProblems]);

  // Extract code from question for function_completion problems
  const extractCodeFromQuestion = useCallback((question: string): { description: string; code: string | null } => {
    const codeBlockMatch = question.match(/```python\n([\s\S]*?)```/);
    if (codeBlockMatch) {
      // Remove the code block from the description
      const description = question.replace(/```python\n[\s\S]*?```/, '').trim();
      return { description, code: codeBlockMatch[1].trim() };
    }
    return { description: question, code: null };
  }, []);

  // Get display description (without code block for function_completion)
  const displayDescription = useMemo(() => {
    if (!selectedProblem) return '';
    if (selectedProblem.type === 'function_completion') {
      const { description } = extractCodeFromQuestion(selectedProblem.question);
      return description || 'Complete the function below:';
    }
    return selectedProblem.question;
  }, [selectedProblem, extractCodeFromQuestion]);

  // Get the function signature for function_completion problems
  // Returns imports + def line + docstring (everything before 'pass')
  const getFunctionSignature = useCallback((question: string): string | null => {
    const { code } = extractCodeFromQuestion(question);
    if (!code) return null;
    
    const lines = code.split('\n');
    const signatureLines: string[] = [];
    let foundDef = false;
    let inDocstring = false;
    let docstringChar = '';
    let docstringComplete = false;
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Check if this is the def line
      if (!foundDef && trimmed.startsWith('def ')) {
        foundDef = true;
        signatureLines.push(line);
        continue;
      }
      
      // If we haven't found def yet, this is an import or other preamble - include it
      if (!foundDef) {
        signatureLines.push(line);
        continue;
      }
      
      // After def line, check for docstring
      if (!inDocstring && !docstringComplete && (line.includes('"""') || line.includes("'''"))) {
        signatureLines.push(line);
        docstringChar = line.includes('"""') ? '"""' : "'''";
        // Check if docstring starts and ends on same line
        const count = (line.match(new RegExp(docstringChar.replace(/"/g, '\\"'), 'g')) || []).length;
        if (count >= 2) {
          // Docstring complete on one line
          docstringComplete = true;
          continue;
        }
        inDocstring = true;
        continue;
      }
      
      // Check for docstring end (multi-line docstring)
      if (inDocstring && line.includes(docstringChar)) {
        signatureLines.push(line);
        inDocstring = false;
        docstringComplete = true;
        continue;
      }
      
      // Still inside multi-line docstring
      if (inDocstring) {
        signatureLines.push(line);
        continue;
      }
      
      // After docstring is complete, stop at 'pass' or any actual code
      if (docstringComplete || foundDef) {
        if (trimmed === 'pass' || trimmed === '' || trimmed.startsWith('#')) {
          // Skip 'pass', empty lines, and comments after docstring
          continue;
        }
        // Found actual implementation code - stop here
        break;
      }
    }
    
    return signatureLines.join('\n');
  }, [extractCodeFromQuestion]);

  const handleSelectProblem = useCallback((problem: CodingProblem) => {
    setSelectedProblem(problem);
    
    // Set initial code template based on problem type
    if (problem.type === 'function_completion') {
      const { code } = extractCodeFromQuestion(problem.question);
      if (code) {
        setUserCode(code + '\n    # Your code here\n    pass');
      } else {
        setUserCode('# Write your solution here\n');
      }
    } else {
      setUserCode('# Write your solution here\n');
    }
  }, [extractCodeFromQuestion]);

  const handleTestComplete = useCallback((result: TestResult) => {
    if (result.passed && selectedProblem) {
      setSolvedProblems((prev) => new Set([...prev, selectedProblem.id]));
    }
  }, [selectedProblem]);

  const toggleAIHelper = useCallback(() => {
    setIsAIHelperCollapsed(prev => !prev);
  }, []);

  // Handler to apply code from AI Helper to the editor
  const handleApplyCode = useCallback((code: string) => {
    if (!selectedProblem) {
      setUserCode(code);
      return;
    }

    // Extract actual code from markdown code blocks if present
    const extractedCode = extractCodeFromResponse(code, selectedProblem.entryPoint);

    if (selectedProblem.type === 'function_completion') {
      // For function completion, combine the function signature with the generated body
      const signature = getFunctionSignature(selectedProblem.question);
      if (signature) {
        // Check if the AI response already includes the full function definition
        const hasFullFunction = extractedCode.match(/^\s*def\s+\w+\s*\(/m);
        
        if (hasFullFunction) {
          // AI returned full function - use it directly
          const normalized = normalizeIndentation(extractedCode, 0);
          setUserCode(normalized);
        } else {
          // AI returned only the body - combine with signature
          // Normalize the body code to have consistent 4-space indentation for function body
          const normalizedBody = normalizeIndentation(extractedCode, 4);
          setUserCode(signature + '\n' + normalizedBody);
        }
      } else {
        setUserCode(extractedCode);
      }
    } else {
      // For code generation, replace the entire code
      const normalized = normalizeIndentation(extractedCode, 0);
      setUserCode(normalized);
    }
  }, [selectedProblem, getFunctionSignature]);

  return (
    <div className={clsx('h-full flex overflow-hidden', className)}>
      {/* Problem List Sidebar */}
      <div
        className={clsx(
          'flex-shrink-0 transition-all duration-200 relative h-full',
          isProblemListCollapsed ? 'w-12' : ''
        )}
        style={{ width: isProblemListCollapsed ? 48 : problemListWidth }}
      >
        {isProblemListCollapsed ? (
          <div className="h-full flex flex-col items-center pt-4 bg-zinc-900/95 border-r border-zinc-800/80">
            <button
              onClick={() => setIsProblemListCollapsed(false)}
              className="p-2 rounded-md hover:bg-zinc-800/50 transition-colors"
              title="Expand problems"
            >
              <span className="text-xs text-zinc-500 [writing-mode:vertical-lr] font-medium">
                Problems
              </span>
            </button>
          </div>
        ) : (
          <>
            <ProblemList
              onSelectProblem={handleSelectProblem}
              selectedProblemId={selectedProblem?.id}
              solvedProblems={solvedProblems}
            />
            <button
              onClick={() => setIsProblemListCollapsed(true)}
              className="absolute -right-3 top-4 w-6 h-6 rounded-full bg-zinc-800 border border-zinc-700/50 flex items-center justify-center hover:bg-zinc-700 transition-colors z-50"
              title="Collapse problems"
            >
              <ChevronLeft className="w-4 h-4 text-zinc-400" />
            </button>
            <div
              className="absolute top-0 bottom-0 -right-0.5 w-1 cursor-col-resize hover:bg-teal-500/50 transition-colors z-40"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = problemListWidth;

                const handleMouseMove = (moveEvent: MouseEvent) => {
                  const newWidth = Math.min(500, Math.max(240, startWidth + moveEvent.clientX - startX));
                  setProblemListWidth(newWidth);
                };

                const handleMouseUp = () => {
                  document.removeEventListener('mousemove', handleMouseMove);
                  document.removeEventListener('mouseup', handleMouseUp);
                  document.body.style.cursor = '';
                  document.body.style.userSelect = '';
                };

                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
              }}
            />
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 bg-zinc-950 h-full overflow-hidden">
        {selectedProblem ? (
          <>
            {/* Problem Description - compact header */}
            <div className="flex-shrink-0 border-b border-zinc-800/80 bg-zinc-900/50">
              <div className="px-4 py-3">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span
                    className={clsx(
                      'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border',
                      selectedProblem.type === 'function_completion'
                        ? 'bg-emerald-900/30 text-emerald-400 border-emerald-700/30'
                        : 'bg-blue-900/30 text-blue-400 border-blue-700/30'
                    )}
                  >
                    {TASK_LABELS[selectedProblem.type].label}
                  </span>
                  <span className="text-xs text-zinc-500">
                    {CATEGORY_LABELS[selectedProblem.category]}
                  </span>
                  {selectedProblem.hasImage && (
                    <span className="flex items-center gap-1 text-xs text-zinc-500">
                      <ImageIcon className="w-3.5 h-3.5" />
                      Has image
                    </span>
                  )}
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-zinc-300 leading-relaxed">
                      {displayDescription}
                    </p>
                  </div>
                  {selectedProblem.imageUrl && (
                    <div className="flex-shrink-0">
                      <img
                        src={selectedProblem.imageUrl}
                        alt="Problem illustration"
                        className="max-w-[160px] max-h-24 rounded-lg border border-zinc-700/50 bg-zinc-900 object-contain"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Test Runner - at top, compact */}
            <div className="flex-shrink-0 border-b border-zinc-800/80">
              <TestRunner
                userCode={userCode}
                testCode={selectedProblem.testCode}
                entryPoint={selectedProblem.entryPoint}
                onTestComplete={handleTestComplete}
              />
            </div>

            {/* Code Editor - takes remaining space */}
            <div className="flex-1 min-h-0">
              <CodeEditor
                value={userCode}
                onChange={setUserCode}
                language="python"
                height="calc(100vh - 220px)"
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center px-4">
            <div className="w-16 h-16 mb-5 rounded-xl bg-zinc-800/80 border border-teal-700/30 flex items-center justify-center">
              <FileText className="w-8 h-8 text-teal-400" />
            </div>
            <h2 className="text-xl font-semibold text-zinc-200 mb-2">
              Practice Mode
            </h2>
            <p className="text-zinc-500 max-w-md mb-6 text-sm leading-relaxed">
              Select a coding problem from the sidebar to start practicing.
              Solve problems and run unit tests to verify your solutions.
            </p>
            <div className="flex items-center gap-2 text-xs text-zinc-600">
              <Lightbulb className="w-4 h-4" />
              <span>Use the AI Helper for hints and guidance</span>
            </div>
          </div>
        )}
      </div>

      {/* AI Helper Sidebar */}
      <div
        className={clsx(
          'flex-shrink-0 transition-all duration-200 relative h-full'
        )}
        style={{ width: isAIHelperCollapsed ? 48 : aiHelperWidth }}
      >
        {!isAIHelperCollapsed && (
          <>
            <button
              onClick={toggleAIHelper}
              className="absolute -left-3 top-4 w-6 h-6 rounded-full bg-zinc-800 border border-zinc-700/50 flex items-center justify-center hover:bg-zinc-700 transition-colors z-50"
              title="Collapse AI Helper"
            >
              <ChevronRight className="w-4 h-4 text-zinc-400" />
            </button>
            <div
              className="absolute top-0 bottom-0 -left-0.5 w-1 cursor-col-resize hover:bg-teal-500/50 transition-colors z-40"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = aiHelperWidth;

                const handleMouseMove = (moveEvent: MouseEvent) => {
                  const newWidth = Math.min(500, Math.max(240, startWidth - (moveEvent.clientX - startX)));
                  setAIHelperWidth(newWidth);
                };

                const handleMouseUp = () => {
                  document.removeEventListener('mousemove', handleMouseMove);
                  document.removeEventListener('mouseup', handleMouseUp);
                  document.body.style.cursor = '';
                  document.body.style.userSelect = '';
                };

                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
              }}
            />
          </>
        )}
        <AIHelper
          problem={selectedProblem}
          userCode={userCode}
          isCollapsed={isAIHelperCollapsed}
          onToggleCollapse={toggleAIHelper}
          onApplyCode={handleApplyCode}
        />
      </div>
    </div>
  );
}
