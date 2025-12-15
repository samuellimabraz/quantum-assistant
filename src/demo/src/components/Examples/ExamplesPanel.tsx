'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import {
  Database,
  Filter,
  Image as ImageIcon,
  FileText,
  Loader2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
} from 'lucide-react';
import { clsx } from 'clsx';
import { ExampleCard } from './ExampleCard';
import { useDataset } from '@/lib/dataset/DatasetProvider';
import { TASK_LABELS, CATEGORY_LABELS } from '@/config/constants';
import type { DatasetExample, TaskType, Category } from '@/types';

interface ExamplesPanelProps {
  onSelectExample: (example: DatasetExample) => void;
}

type ModalityFilter = 'all' | 'multimodal' | 'text-only';
type Split = 'train' | 'validation' | 'test';

const ITEMS_PER_PAGE = 25;

export function ExamplesPanel({ onSelectExample }: ExamplesPanelProps) {
  const { isLoading: isDatasetLoading, loadedSplits, splitCounts, filterExamples, loadSplit } = useDataset();

  const [typeFilter, setTypeFilter] = useState<TaskType | 'all'>('all');
  const [categoryFilter, setCategoryFilter] = useState<Category | 'all'>('all');
  const [modalityFilter, setModalityFilter] = useState<ModalityFilter>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [split, setSplit] = useState<Split>('test');
  const [currentPage, setCurrentPage] = useState(0);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Load train split if selected (not loaded by default)
  useEffect(() => {
    if (split === 'train' && !loadedSplits.has('train')) {
      loadSplit('train');
    }
  }, [split, loadedSplits, loadSplit]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
      setCurrentPage(0);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(0);
  }, [split, typeFilter, categoryFilter, modalityFilter, debouncedSearch]);

  // Filter locally loaded data
  const { examples, totalExamples } = useMemo(() => {
    if (!loadedSplits.has(split)) {
      return { examples: [], totalExamples: 0 };
    }

    const filters: {
      type?: TaskType;
      category?: Category;
      hasImage?: boolean;
      search?: string;
    } = {};

    if (typeFilter !== 'all') filters.type = typeFilter;
    if (categoryFilter !== 'all') filters.category = categoryFilter;
    if (modalityFilter === 'multimodal') filters.hasImage = true;
    else if (modalityFilter === 'text-only') filters.hasImage = false;
    if (debouncedSearch) filters.search = debouncedSearch;

    const result = filterExamples(split, filters, ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE);

    return { examples: result.examples, totalExamples: result.total };
  }, [loadedSplits, split, filterExamples, typeFilter, categoryFilter, modalityFilter, debouncedSearch, currentPage]);

  // Scroll to top on page change
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = 0;
    }
  }, [currentPage]);

  const totalPages = Math.ceil(totalExamples / ITEMS_PER_PAGE);

  const stats = useMemo(() => ({
    total: totalExamples,
    displayed: examples.length,
  }), [examples, totalExamples]);

  const clearSearch = () => {
    setSearchQuery('');
    searchInputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      clearSearch();
    }
  };

  const isLoading = isDatasetLoading || !loadedSplits.has(split);

  return (
    <div className="h-full flex flex-col bg-zinc-900/95 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800/80 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-teal-500" />
            <h2 className="font-semibold text-zinc-200">Dataset Examples</h2>
          </div>
          {isLoading && (
            <Loader2 className="w-4 h-4 animate-spin text-zinc-500" />
          )}
        </div>

        {/* Split Selector */}
        <div className="flex gap-1 mb-3 bg-zinc-800/50 p-1 rounded-lg">
          {(['train', 'validation', 'test'] as Split[]).map((s) => (
            <button
              key={s}
              onClick={() => setSplit(s)}
              className={clsx(
                'flex-1 px-2 py-1.5 text-xs font-medium rounded-md transition-all',
                split === s
                  ? 'bg-teal-600/80 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50'
              )}
            >
              <div className="flex items-center justify-center gap-1">
                <span className="capitalize">{s}</span>
                {splitCounts[s] && (
                  <span className="text-[10px] opacity-70">({splitCounts[s]})</span>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search examples..."
            className="w-full bg-zinc-800/80 border border-zinc-700/50 rounded-lg pl-9 pr-8 py-2 text-sm text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-teal-600/50 focus:border-teal-700/50"
          />
          {searchQuery && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Stats */}
        <div className="flex items-center gap-2 text-xs text-zinc-500 mb-3">
          <span className="flex items-center gap-1">
            <FileText className="w-3.5 h-3.5" />
            {stats.total} examples
          </span>
          {debouncedSearch && (
            <>
              <span className="text-zinc-600">|</span>
              <span className="text-teal-400 truncate max-w-[100px]">
                &quot;{debouncedSearch}&quot;
              </span>
            </>
          )}
        </div>

        {/* Filters Toggle */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          <Filter className="w-4 h-4" />
          <span>Filters</span>
          {(typeFilter !== 'all' || categoryFilter !== 'all' || modalityFilter !== 'all') && (
            <span className="px-1.5 py-0.5 text-[10px] bg-teal-600/30 text-teal-400 rounded">
              Active
            </span>
          )}
          <ChevronDown
            className={clsx(
              'w-4 h-4 transition-transform',
              showFilters && 'rotate-180'
            )}
          />
        </button>

        {/* Filter Options */}
        {showFilters && (
          <div className="mt-3 space-y-3 animate-in slide-in-from-top-2 duration-200">
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
                onChange={(e) => setCategoryFilter(e.target.value as Category | 'all')}
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
                {(['all', 'multimodal', 'text-only'] as ModalityFilter[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setModalityFilter(mode)}
                    className={clsx(
                      'flex-1 py-1.5 px-2 rounded-md text-xs font-medium transition-all',
                      modalityFilter === mode
                        ? 'bg-teal-700/80 text-white'
                        : 'bg-zinc-800/80 text-zinc-400 hover:text-zinc-200'
                    )}
                  >
                    {mode === 'all' ? 'All' : mode === 'multimodal' ? 'Multimodal' : 'Text'}
                  </button>
                ))}
              </div>
            </div>

            {/* Clear Filters */}
            {(typeFilter !== 'all' || categoryFilter !== 'all' || modalityFilter !== 'all') && (
              <button
                onClick={() => {
                  setTypeFilter('all');
                  setCategoryFilter('all');
                  setModalityFilter('all');
                }}
                className="w-full py-2 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 rounded-md transition-colors"
              >
                Clear all filters
              </button>
            )}
          </div>
        )}
      </div>

      {/* Examples List */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-3 scroll-smooth min-h-0">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-40 text-zinc-500">
            <Loader2 className="w-6 h-6 animate-spin mb-2" />
            <span className="text-sm">Loading examples...</span>
          </div>
        ) : examples.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-zinc-500 text-center">
            <Filter className="w-6 h-6 mb-2 opacity-50" />
            <p className="text-sm">No examples match your filters</p>
            {debouncedSearch && (
              <button
                onClick={clearSearch}
                className="mt-2 text-teal-400 hover:text-teal-300 text-sm"
              >
                Clear search
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 px-1 mb-2">
              Showing {currentPage * ITEMS_PER_PAGE + 1}–{Math.min((currentPage + 1) * ITEMS_PER_PAGE, totalExamples)} of {totalExamples}
            </p>
            {examples.map((example) => (
              <ExampleCard
                key={example.id}
                example={example}
                onSelect={onSelectExample}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && !isLoading && (
        <div className="p-2 border-t border-zinc-800/80 flex-shrink-0">
          <div className="flex items-center justify-between gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className={clsx(
                'flex items-center gap-0.5 px-2 py-1 text-xs font-medium rounded-md transition-colors flex-shrink-0',
                currentPage === 0
                  ? 'text-zinc-600 cursor-not-allowed'
                  : 'text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100'
              )}
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Prev</span>
            </button>

            <div className="flex items-center gap-0.5 overflow-hidden flex-1 justify-center min-w-0">
              {(() => {
                const maxVisible = 3;
                const pages: (number | 'ellipsis')[] = [];

                if (totalPages <= maxVisible + 2) {
                  for (let i = 0; i < totalPages; i++) pages.push(i);
                } else {
                  pages.push(0);

                  if (currentPage > 2) {
                    pages.push('ellipsis');
                  }

                  const start = Math.max(1, currentPage - 1);
                  const end = Math.min(totalPages - 2, currentPage + 1);

                  for (let i = start; i <= end; i++) {
                    if (!pages.includes(i)) pages.push(i);
                  }

                  if (currentPage < totalPages - 3) {
                    pages.push('ellipsis');
                  }

                  if (!pages.includes(totalPages - 1)) {
                    pages.push(totalPages - 1);
                  }
                }

                return pages.map((page, idx) => {
                  if (page === 'ellipsis') {
                    return (
                      <span key={`ellipsis-${idx}`} className="text-zinc-600 px-0.5 text-xs">
                        …
                      </span>
                    );
                  }

                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={clsx(
                        'w-6 h-6 text-[11px] font-medium rounded transition-colors flex-shrink-0',
                        currentPage === page
                          ? 'bg-teal-600/80 text-white'
                          : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                      )}
                    >
                      {page + 1}
                    </button>
                  );
                });
              })()}
            </div>

            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={currentPage >= totalPages - 1}
              className={clsx(
                'flex items-center gap-0.5 px-2 py-1 text-xs font-medium rounded-md transition-colors flex-shrink-0',
                currentPage >= totalPages - 1
                  ? 'text-zinc-600 cursor-not-allowed'
                  : 'text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100'
              )}
            >
              <span className="hidden sm:inline">Next</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
