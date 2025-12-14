'use client';

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';
import { datasetLoader, FilterOptions, LoadExamplesResult } from './loader';
import type { DatasetExample, CodingProblem } from '@/types';

type Split = 'train' | 'validation' | 'test';

interface DatasetContextValue {
  isLoading: boolean;
  loadedSplits: Set<Split>;
  splitCounts: Record<string, number>;
  loadSplit: (split: Split) => Promise<void>;
  filterExamples: (
    split: Split,
    filters: FilterOptions,
    limit?: number,
    offset?: number
  ) => LoadExamplesResult;
  getCodingProblems: (split: Split) => CodingProblem[];
  getAllExamples: (split: Split) => DatasetExample[];
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

interface DatasetProviderProps {
  children: ReactNode;
  initialSplits?: Split[];
}

export function DatasetProvider({
  children,
  initialSplits = ['test', 'validation'],
}: DatasetProviderProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [loadedSplits, setLoadedSplits] = useState<Set<Split>>(new Set());
  const [splitCounts, setSplitCounts] = useState<Record<string, number>>({});

  // Load initial splits on mount
  useEffect(() => {
    const loadInitialData = async () => {
      setIsLoading(true);
      try {
        // Load split info first
        const info = await datasetLoader.getSplitInfo();
        setSplitCounts(info);

        // Load initial splits in parallel
        await Promise.all(
          initialSplits.map(async (split) => {
            await datasetLoader.preloadSplit(split);
            setLoadedSplits((prev) => new Set([...prev, split]));
          })
        );
      } catch (error) {
        console.error('Failed to load dataset:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialData();
  }, []);

  const loadSplit = useCallback(async (split: Split) => {
    if (datasetLoader.isLoaded(split)) {
      setLoadedSplits((prev) => new Set([...prev, split]));
      return;
    }

    await datasetLoader.preloadSplit(split);
    setLoadedSplits((prev) => new Set([...prev, split]));

    // Update counts after loading
    const examples = datasetLoader.getAllExamples(split);
    setSplitCounts((prev) => ({ ...prev, [split]: examples.length }));
  }, []);

  const filterExamples = useCallback(
    (
      split: Split,
      filters: FilterOptions,
      limit: number = 50,
      offset: number = 0
    ): LoadExamplesResult => {
      if (!datasetLoader.isLoaded(split)) {
        return { examples: [], total: 0 };
      }
      return datasetLoader.filterExamples(split, filters, limit, offset);
    },
    []
  );

  const getCodingProblems = useCallback((split: Split): CodingProblem[] => {
    return datasetLoader.getCodingProblems(split);
  }, []);

  const getAllExamples = useCallback((split: Split): DatasetExample[] => {
    return datasetLoader.getAllExamples(split);
  }, []);

  return (
    <DatasetContext.Provider
      value={{
        isLoading,
        loadedSplits,
        splitCounts,
        loadSplit,
        filterExamples,
        getCodingProblems,
        getAllExamples,
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error('useDataset must be used within a DatasetProvider');
  }
  return context;
}

