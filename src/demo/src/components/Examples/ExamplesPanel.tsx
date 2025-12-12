'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Database,
  Filter,
  RefreshCw,
  Image as ImageIcon,
  FileText,
  Loader2,
  ChevronDown,
} from 'lucide-react';
import { clsx } from 'clsx';
import { ExampleCard } from './ExampleCard';
import { TASK_LABELS, CATEGORY_LABELS } from '@/config/constants';
import type { DatasetExample, TaskType, Category } from '@/types';

interface ExamplesPanelProps {
  onSelectExample: (example: DatasetExample) => void;
}

type ModalityFilter = 'all' | 'multimodal' | 'text-only';

export function ExamplesPanel({ onSelectExample }: ExamplesPanelProps) {
  const [examples, setExamples] = useState<DatasetExample[]>([]);
  const [filteredExamples, setFilteredExamples] = useState<DatasetExample[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [typeFilter, setTypeFilter] = useState<TaskType | 'all'>('all');
  const [categoryFilter, setCategoryFilter] = useState<Category | 'all'>('all');
  const [modalityFilter, setModalityFilter] = useState<ModalityFilter>('all');
  const [showFilters, setShowFilters] = useState(false);

  const loadExamples = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/examples?split=test&limit=100');
      if (!response.ok) throw new Error('Failed to load examples');

      const data = await response.json();
      setExamples(data.examples);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load examples');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadExamples();
  }, [loadExamples]);

  useEffect(() => {
    let filtered = [...examples];

    if (typeFilter !== 'all') {
      filtered = filtered.filter((e) => e.type === typeFilter);
    }

    if (categoryFilter !== 'all') {
      filtered = filtered.filter((e) => e.category === categoryFilter);
    }

    if (modalityFilter === 'multimodal') {
      filtered = filtered.filter((e) => e.hasImage);
    } else if (modalityFilter === 'text-only') {
      filtered = filtered.filter((e) => !e.hasImage);
    }

    setFilteredExamples(filtered);
  }, [examples, typeFilter, categoryFilter, modalityFilter]);

  const stats = {
    total: examples.length,
    multimodal: examples.filter((e) => e.hasImage).length,
    textOnly: examples.filter((e) => !e.hasImage).length,
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-zinc-800/80">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-teal-500" />
            <h2 className="font-semibold text-zinc-200">
              Test Examples
            </h2>
          </div>
          <button
            onClick={loadExamples}
            disabled={isLoading}
            className="p-1.5 rounded-md hover:bg-zinc-800/50 transition-colors"
            title="Refresh examples"
          >
            <RefreshCw
              className={clsx(
                'w-4 h-4 text-zinc-500',
                isLoading && 'animate-spin'
              )}
            />
          </button>
        </div>

        <div className="flex items-center gap-2 text-xs text-zinc-500 mb-3">
          <span className="flex items-center gap-1">
            <ImageIcon className="w-3.5 h-3.5" />
            {stats.multimodal} multimodal
          </span>
          <span className="text-zinc-600">|</span>
          <span className="flex items-center gap-1">
            <FileText className="w-3.5 h-3.5" />
            {stats.textOnly} text-only
          </span>
        </div>

        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          <Filter className="w-4 h-4" />
          <span>Filters</span>
          <ChevronDown
            className={clsx(
              'w-4 h-4 transition-transform',
              showFilters && 'rotate-180'
            )}
          />
        </button>

        {showFilters && (
          <div className="mt-3 space-y-3 animate-in">
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">
                Task Type
              </label>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value as TaskType | 'all')}
                className="w-full bg-zinc-800/80 border border-zinc-700/50 rounded-md px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:ring-1 focus:ring-teal-600/50"
              >
                <option value="all">All Types</option>
                {Object.entries(TASK_LABELS).map(([key, config]) => (
                  <option key={key} value={key}>
                    {config.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-zinc-500 mb-1 block">
                Category
              </label>
              <select
                value={categoryFilter}
                onChange={(e) =>
                  setCategoryFilter(e.target.value as Category | 'all')
                }
                className="w-full bg-zinc-800/80 border border-zinc-700/50 rounded-md px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:ring-1 focus:ring-teal-600/50"
              >
                <option value="all">All Categories</option>
                {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
                  <option key={key} value={key}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-zinc-500 mb-1 block">
                Modality
              </label>
              <div className="flex gap-2">
                {(['all', 'multimodal', 'text-only'] as ModalityFilter[]).map(
                  (mode) => (
                    <button
                      key={mode}
                      onClick={() => setModalityFilter(mode)}
                      className={clsx(
                        'flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all font-mono',
                        modalityFilter === mode
                          ? 'bg-teal-700/80 text-white'
                          : 'bg-zinc-800/80 text-zinc-400 hover:text-zinc-200'
                      )}
                    >
                      {mode === 'all'
                        ? 'All'
                        : mode === 'multimodal'
                        ? 'MM'
                        : 'Text'}
                    </button>
                  )
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-40 text-zinc-500">
            <Loader2 className="w-6 h-6 animate-spin mb-2" />
            <span className="text-sm">Loading examples...</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-40 text-center px-4">
            <p className="text-red-400 text-sm mb-3">{error}</p>
            <button onClick={loadExamples} className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md text-sm transition-colors">
              Retry
            </button>
          </div>
        ) : filteredExamples.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-zinc-500 text-center">
            <Filter className="w-6 h-6 mb-2 opacity-50" />
            <p className="text-sm">No examples match your filters</p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 px-1 mb-2">
              {filteredExamples.length} examples
            </p>
            {filteredExamples.map((example) => (
              <ExampleCard
                key={example.id}
                example={example}
                onSelect={onSelectExample}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
