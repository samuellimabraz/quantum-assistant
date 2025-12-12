'use client';

import { useState, useCallback } from 'react';
import { PanelRightOpen, PanelRightClose, ChevronRight, ChevronLeft } from 'lucide-react';
import { clsx } from 'clsx';
import { Header, ChatInterface, ExamplesPanel } from '@/components';
import { PROJECT_CONFIG } from '@/config/constants';
import type { DatasetExample } from '@/types';

export default function HomePage() {
  const [selectedExample, setSelectedExample] = useState<DatasetExample | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);

  const handleSelectExample = useCallback((example: DatasetExample) => {
    setSelectedExample(example);
  }, []);

  const handleExampleUsed = useCallback(() => {
    setSelectedExample(null);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-zinc-950">
      <Header />

      <main className="flex-1 flex overflow-hidden">
        <div
          className={clsx(
            'flex-1 flex flex-col transition-all duration-300',
            isPanelOpen && !isPanelCollapsed ? 'lg:mr-80' : isPanelOpen && isPanelCollapsed ? 'lg:mr-12' : ''
          )}
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
            'transform transition-all duration-300 z-40',
            'lg:translate-x-0',
            isPanelOpen ? 'translate-x-0' : 'translate-x-full',
            isPanelCollapsed ? 'w-12' : 'w-80'
          )}
        >
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
