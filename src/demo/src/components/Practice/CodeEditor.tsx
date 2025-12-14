'use client';

import { useRef, useCallback } from 'react';
import Editor, { OnMount, OnChange } from '@monaco-editor/react';
import type * as Monaco from 'monaco-editor';
import { Loader2 } from 'lucide-react';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  readOnly?: boolean;
  height?: string;
  className?: string;
}

export function CodeEditor({
  value,
  onChange,
  language = 'python',
  readOnly = false,
  height = '100%',
  className,
}: CodeEditorProps) {
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);

  const handleEditorMount: OnMount = useCallback((editor) => {
    editorRef.current = editor;
    editor.focus();
  }, []);

  const handleChange: OnChange = useCallback((val) => {
    onChange(val || '');
  }, [onChange]);

  return (
    <div className={className} style={{ height, minHeight: '300px' }}>
      <Editor
        height="100%"
        width="100%"
        language={language}
        value={value}
        onChange={handleChange}
        onMount={handleEditorMount}
        theme="quantum-dark"
        loading={
          <div className="flex items-center justify-center h-full bg-zinc-900">
            <Loader2 className="w-6 h-6 animate-spin text-teal-500" />
          </div>
        }
        beforeMount={(monaco) => {
          monaco.editor.defineTheme('quantum-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
              { token: 'comment', foreground: '71717a', fontStyle: 'italic' },
              { token: 'keyword', foreground: 'c4b5fd' },
              { token: 'string', foreground: '86efac' },
              { token: 'number', foreground: 'c4b5fd' },
              { token: 'type', foreground: '93c5fd' },
              { token: 'function', foreground: '93c5fd' },
              { token: 'variable', foreground: 'd4d4d8' },
              { token: 'operator', foreground: 'f0abfc' },
              { token: 'delimiter', foreground: 'a1a1aa' },
            ],
            colors: {
              'editor.background': '#18181b',
              'editor.foreground': '#d4d4d8',
              'editor.lineHighlightBackground': '#27272a',
              'editor.selectionBackground': '#0d9488aa',
              'editor.inactiveSelectionBackground': '#27272a',
              'editorCursor.foreground': '#14b8a6',
              'editorLineNumber.foreground': '#52525b',
              'editorLineNumber.activeForeground': '#a1a1aa',
              'editorIndentGuide.background': '#27272a',
              'editorIndentGuide.activeBackground': '#3f3f46',
              'editor.selectionHighlightBackground': '#0d94882a',
              'editorBracketMatch.background': '#0d94884a',
              'editorBracketMatch.border': '#14b8a6',
              'scrollbar.shadow': '#00000000',
              'scrollbarSlider.background': '#3f3f4680',
              'scrollbarSlider.hoverBackground': '#52525b80',
              'scrollbarSlider.activeBackground': '#71717a80',
            },
          });
        }}
        options={{
          readOnly,
          fontSize: 14,
          fontFamily: "'JetBrains Mono', Consolas, 'Courier New', monospace",
          fontLigatures: true,
          lineHeight: 1.6,
          padding: { top: 16, bottom: 16 },
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 4,
          insertSpaces: true,
          wordWrap: 'on',
          lineNumbers: 'on',
          glyphMargin: false,
          folding: true,
          lineDecorationsWidth: 8,
          lineNumbersMinChars: 4,
          renderLineHighlight: 'line',
          cursorBlinking: 'smooth',
          cursorSmoothCaretAnimation: 'on',
          smoothScrolling: true,
          contextmenu: true,
          quickSuggestions: true,
          suggestOnTriggerCharacters: true,
          acceptSuggestionOnEnter: 'on',
          formatOnPaste: true,
          formatOnType: true,
          bracketPairColorization: { enabled: true },
        }}
      />
    </div>
  );
}

