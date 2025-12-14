'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { PanelRightOpen, PanelRightClose, ChevronRight, ChevronLeft, MessageSquare, Code } from 'lucide-react';
import { clsx } from 'clsx';
import { Header, ChatInterface, ExamplesPanel, PracticeInterface } from '@/components';
import { PROJECT_CONFIG } from '@/config/constants';
import type { DatasetExample, AppMode } from '@/types';

export default function HomePage() {
  const [selectedExample, setSelectedExample] = useState<DatasetExample | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  const [panelWidth, setPanelWidth] = useState(320);
  const [mode, setMode] = useState<AppMode>('chat');
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleSelectExample = useCallback((example: DatasetExample) => {
    setSelectedExample(example);
  }, []);

  const handleExampleUsed = useCallback(() => {
    setSelectedExample(null);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX;
    startWidthRef.current = panelWidth;
  }, [panelWidth]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const diff = startXRef.current - e.clientX;
      const newWidth = Math.min(500, Math.max(240, startWidthRef.current + diff));
      setPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        localStorage.setItem('examplesPanelWidth', panelWidth.toString());
      }
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, panelWidth]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('examplesPanelWidth');
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed) && parsed >= 240 && parsed <= 500) {
          setPanelWidth(parsed);
        }
      }
    }
  }, []);

  const currentWidth = isPanelCollapsed ? 48 : panelWidth;

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950">
      <Header mode={mode} onModeChange={setMode} />

      <main className="flex-1 flex overflow-hidden">
        {mode === 'chat' ? (
          <>
            <div
              className={clsx(
                'flex-1 flex flex-col',
                'transition-[margin]',
                isDragging ? 'duration-0' : 'duration-300',
                isPanelOpen && !isPanelCollapsed ? '' : '',
                isPanelOpen ? '' : ''
              )}
              style={{
                marginRight: isPanelOpen ? currentWidth : 0,
              }}
            >
              <div className="flex items-center justify-end px-4 py-2 border-b border-zinc-800/80 lg:hidden">
                <button
                  onClick={() => setIsPanelOpen(!isPanelOpen)}
                  className="p-2 rounded-md hover:bg-zinc-800/50 transition-colors"
                >
                  {isPanelOpen ? (
                    <PanelRightClose className="w-5 h-5 text-zinc-500" />
                  ) : (
                    <PanelRightOpen className="w-5 h-5 text-zinc-500" />
                  )}
                </button>
              </div>

              <div className="flex-1 overflow-hidden">
                <ChatInterface
                  selectedExample={selectedExample}
                  onExampleUsed={handleExampleUsed}
                />
              </div>
            </div>

            <aside
              className={clsx(
                'fixed right-0 top-[57px] bottom-0 bg-zinc-900/95 backdrop-blur-sm border-l border-zinc-800/80',
                'transform z-40',
                'transition-[transform,width]',
                isDragging ? 'duration-0' : 'duration-300',
                'lg:translate-x-0',
                isPanelOpen ? 'translate-x-0' : 'translate-x-full'
              )}
              style={{ width: currentWidth }}
            >
              {/* Resize handle */}
              {!isPanelCollapsed && (
                <div
                  onMouseDown={handleMouseDown}
                  className={clsx(
                    'absolute top-0 bottom-0 -left-0.5 w-1 cursor-col-resize z-50',
                    'hover:bg-teal-500/50 transition-colors',
                    isDragging && 'bg-teal-500/70'
                  )}
                >
                  <div
                    className={clsx(
                      'absolute top-1/2 -translate-y-1/2 w-4 h-16 -left-1.5',
                      'flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity',
                      isDragging && 'opacity-100'
                    )}
                  >
                    <div className="w-1 h-8 rounded-full bg-zinc-600 flex flex-col items-center justify-center gap-1">
                      <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
                      <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
                      <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={() => setIsPanelCollapsed(!isPanelCollapsed)}
                className="hidden lg:flex absolute -left-3 top-4 w-6 h-6 rounded-full bg-zinc-800 border border-zinc-700/50 items-center justify-center hover:bg-zinc-700 transition-colors z-50"
                title={isPanelCollapsed ? 'Expand panel' : 'Collapse panel'}
              >
                {isPanelCollapsed ? (
                  <ChevronLeft className="w-4 h-4 text-zinc-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-zinc-400" />
                )}
              </button>

              {isPanelCollapsed ? (
                <div className="h-full flex flex-col items-center pt-8">
                  <button
                    onClick={() => setIsPanelCollapsed(false)}
                    className="p-2 rounded-md hover:bg-zinc-800/50 transition-colors"
                    title="Expand examples"
                  >
                    <span className="text-xs text-zinc-500 [writing-mode:vertical-lr] rotate-180 font-medium">
                      Test Examples
                    </span>
                  </button>
                </div>
              ) : (
                <ExamplesPanel onSelectExample={handleSelectExample} />
              )}
            </aside>

            {isPanelOpen && (
              <div
                className="fixed inset-0 bg-black/50 z-30 lg:hidden"
                onClick={() => setIsPanelOpen(false)}
              />
            )}
          </>
        ) : (
          <PracticeInterface className="flex-1" />
        )}
      </main>

      <footer className="bg-zinc-900/95 border-t border-zinc-800/80 py-3 px-4 text-center text-xs text-zinc-500">
        <p>
          {PROJECT_CONFIG.name} - {PROJECT_CONFIG.year} |{' '}
          <span>{PROJECT_CONFIG.institution}</span> |{' '}
          Apache 2.0 License
        </p>
      </footer>
    </div>
  );
}
