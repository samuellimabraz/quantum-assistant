import type { DatasetExample, TaskType, Category } from '@/types';

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

const HF_DATASET_API = 'https://datasets-server.huggingface.co';
const DATASET_ID = 'samuellimabraz/quantum-assistant';

export class DatasetLoader {
  private cache: Map<string, DatasetExample[]> = new Map();

  async loadExamples(
    split: 'train' | 'validation' | 'test' = 'test',
    limit: number = 50,
    offset: number = 0
  ): Promise<DatasetExample[]> {
    const cacheKey = `${split}-${offset}-${limit}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

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

    this.cache.set(cacheKey, examples);
    return examples;
  }

  async loadFilteredExamples(
    filters: {
      type?: TaskType;
      category?: Category;
      hasImage?: boolean;
    },
    split: 'train' | 'validation' | 'test' = 'test',
    limit: number = 100
  ): Promise<DatasetExample[]> {
    const allExamples = await this.loadExamples(split, limit);

    return allExamples.filter((example) => {
      if (filters.type && example.type !== filters.type) return false;
      if (filters.category && example.category !== filters.category) return false;
      if (filters.hasImage !== undefined && example.hasImage !== filters.hasImage) return false;
      return true;
    });
  }

  async getDatasetInfo(): Promise<{ totalSamples: number; splits: string[] }> {
    const url = `${HF_DATASET_API}/info?dataset=${encodeURIComponent(DATASET_ID)}`;
    const response = await fetch(url);

    if (!response.ok) {
      return { totalSamples: 8366, splits: ['train', 'validation', 'test'] };
    }

    const data = await response.json();
    const splits = Object.keys(data.dataset_info?.default?.splits || {});
    const totalSamples = Object.values(data.dataset_info?.default?.splits || {}).reduce(
      (acc: number, split: unknown) => acc + ((split as { num_examples: number }).num_examples || 0),
      0
    );

    return { totalSamples: totalSamples as number, splits };
  }
}

export const datasetLoader = new DatasetLoader();

