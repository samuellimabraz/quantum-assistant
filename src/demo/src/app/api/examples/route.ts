import { NextRequest, NextResponse } from 'next/server';
import { datasetLoader } from '@/lib/dataset/loader';
import type { TaskType, Category } from '@/types';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    
    const split = (searchParams.get('split') as 'train' | 'validation' | 'test') || 'test';
    const limit = parseInt(searchParams.get('limit') || '50', 10);
    const offset = parseInt(searchParams.get('offset') || '0', 10);
    const type = searchParams.get('type') as TaskType | null;
    const category = searchParams.get('category') as Category | null;
    const hasImage = searchParams.get('hasImage');

    const filters: {
      type?: TaskType;
      category?: Category;
      hasImage?: boolean;
    } = {};

    if (type) filters.type = type;
    if (category) filters.category = category;
    if (hasImage !== null) filters.hasImage = hasImage === 'true';

    let examples;
    
    if (Object.keys(filters).length > 0) {
      examples = await datasetLoader.loadFilteredExamples(filters, split, limit + offset);
      examples = examples.slice(offset, offset + limit);
    } else {
      examples = await datasetLoader.loadExamples(split, limit, offset);
    }

    return NextResponse.json({
      examples,
      total: examples.length,
      split,
      offset,
      limit,
    });
  } catch (error) {
    console.error('Examples API error:', error);
    
    const errorMessage =
      error instanceof Error ? error.message : 'Failed to load examples';

    return NextResponse.json(
      { error: errorMessage, examples: [] },
      { status: 500 }
    );
  }
}

