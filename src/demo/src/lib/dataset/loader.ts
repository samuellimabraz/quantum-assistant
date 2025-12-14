import type { DatasetExample, TaskType, Category, CodingProblem } from '@/types';

interface HFImage {
  src: string;
  height: number;
  width: number;
}

interface HFDatasetRow {
  question: string;
  answer: string;
  type: string;
  category: string;
  image: HFImage | null;
  test_code: string | null;
  entry_point: string | null;
  source: string;
}

interface HFDatasetResponse {
  rows: Array<{ row: HFDatasetRow; row_idx: number }>;
  num_rows_total: number;
}

interface HFSplitInfo {
  num_examples: number;
}

interface HFDatasetInfo {
  dataset_info?: {
    default?: {
      splits?: Record<string, HFSplitInfo>;
    };
  };
}

export interface LoadExamplesResult {
  examples: DatasetExample[];
  total: number;
}

export interface FilterOptions {
  type?: TaskType;
  category?: Category;
  hasImage?: boolean;
  search?: string;
  codingOnly?: boolean;
}

const HF_DATASET_API = 'https://datasets-server.huggingface.co';
const DATASET_ID = 'samuellimabraz/quantum-assistant';
const MAX_FETCH_LIMIT = 100;

export class DatasetLoader {
  private splitData: Map<string, DatasetExample[]> = new Map();
  private splitInfo: Record<string, number> = {};
  private isLoading: Map<string, Promise<void>> = new Map();

  /**
   * Preload all examples for a split (fetches all data at once)
   */
  async preloadSplit(split: 'train' | 'validation' | 'test'): Promise<void> {
    if (this.splitData.has(split)) {
      return;
    }

    // Prevent duplicate loading
    if (this.isLoading.has(split)) {
      return this.isLoading.get(split);
    }

    const loadPromise = this.fetchAllExamples(split);
    this.isLoading.set(split, loadPromise);

    try {
      await loadPromise;
    } finally {
      this.isLoading.delete(split);
    }
  }

  private async fetchAllExamples(split: 'train' | 'validation' | 'test'): Promise<void> {
    const allExamples: DatasetExample[] = [];
    let offset = 0;
    let total = 0;

    // First request to get total count
    const firstBatch = await this.fetchBatch(split, 0, MAX_FETCH_LIMIT);
    allExamples.push(...firstBatch.examples);
    total = firstBatch.total;
    offset = firstBatch.examples.length;

    // Fetch remaining batches
    while (offset < total) {
      const batch = await this.fetchBatch(split, offset, MAX_FETCH_LIMIT);
      allExamples.push(...batch.examples);
      offset += batch.examples.length;

      if (batch.examples.length < MAX_FETCH_LIMIT) break;
    }

    this.splitData.set(split, allExamples);
    this.splitInfo[split] = allExamples.length;
  }

  private async fetchBatch(
    split: string,
    offset: number,
    limit: number
  ): Promise<{ examples: DatasetExample[]; total: number }> {
    const url = `${HF_DATASET_API}/rows?dataset=${encodeURIComponent(DATASET_ID)}&config=default&split=${split}&offset=${offset}&length=${limit}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load dataset: ${response.status}`);
    }

    const data: HFDatasetResponse = await response.json();

    const examples: DatasetExample[] = data.rows.map((item) => {
      const row = item.row;
      return {
        id: `${split}-${item.row_idx}`,
        question: row.question,
        answer: row.answer,
        type: row.type as TaskType,
        category: row.category as Category,
        imageUrl: row.image?.src || undefined,
        hasImage: row.image !== null,
        testCode: row.test_code || undefined,
        entryPoint: row.entry_point || undefined,
        source: row.source,
      };
    });

    return { examples, total: data.num_rows_total };
  }

  /**
   * Check if a split is loaded
   */
  isLoaded(split: 'train' | 'validation' | 'test'): boolean {
    return this.splitData.has(split);
  }

  /**
   * Get loading progress (for UI feedback)
   */
  isCurrentlyLoading(split: 'train' | 'validation' | 'test'): boolean {
    return this.isLoading.has(split);
  }

  /**
   * Get all examples for a split (must be preloaded first)
   */
  getAllExamples(split: 'train' | 'validation' | 'test'): DatasetExample[] {
    return this.splitData.get(split) || [];
  }

  /**
   * Get coding problems from loaded data
   */
  getCodingProblems(split: 'train' | 'validation' | 'test'): CodingProblem[] {
    const examples = this.splitData.get(split) || [];
    return examples.filter(
      (e): e is CodingProblem =>
        e.testCode !== undefined &&
        e.entryPoint !== undefined &&
        (e.type === 'function_completion' || e.type === 'code_generation')
    );
  }

  /**
   * Filter and paginate locally loaded data
   */
  filterExamples(
    split: 'train' | 'validation' | 'test',
    filters: FilterOptions,
    limit: number = 50,
    offset: number = 0
  ): LoadExamplesResult {
    let examples = filters.codingOnly
      ? this.getCodingProblems(split)
      : this.getAllExamples(split);

    // Apply filters
    if (filters.type) {
      examples = examples.filter((e) => e.type === filters.type);
    }
    if (filters.category) {
      examples = examples.filter((e) => e.category === filters.category);
    }
    if (filters.hasImage !== undefined) {
      examples = examples.filter((e) => e.hasImage === filters.hasImage);
    }
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      examples = examples.filter(
        (e) =>
          e.question.toLowerCase().includes(searchLower) ||
          e.answer.toLowerCase().includes(searchLower)
      );
    }

    const total = examples.length;
    const paginated = examples.slice(offset, offset + limit);

    return { examples: paginated, total };
  }

  /**
   * Get split information
   */
  async getSplitInfo(): Promise<Record<string, number>> {
    // Return cached if available
    if (Object.keys(this.splitInfo).length > 0) {
      return this.splitInfo;
    }

    const url = `${HF_DATASET_API}/info?dataset=${encodeURIComponent(DATASET_ID)}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        return { train: 8366, validation: 1247, test: 1291 };
      }

      const data: HFDatasetInfo = await response.json();
      const splits = data.dataset_info?.default?.splits || {};

      const result: Record<string, number> = {};
      for (const [name, info] of Object.entries(splits)) {
        result[name] = info.num_examples || 0;
      }

      this.splitInfo = result;
      return result;
    } catch {
      return { train: 8366, validation: 1247, test: 1291 };
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.splitData.clear();
    this.splitInfo = {};
  }
}

export const datasetLoader = new DatasetLoader();
