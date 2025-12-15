import { NextRequest, NextResponse } from 'next/server';
import { datasetLoader } from '@/lib/dataset/loader';
import type { TaskType, Category } from '@/types';

// Server-side loader for API route (used for split info and fallback)
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const split = (searchParams.get('split') as 'train' | 'validation' | 'test') || 'test';
    const limit = Math.min(parseInt(searchParams.get('limit') || '50', 10), 100);
    const offset = parseInt(searchParams.get('offset') || '0', 10);
    const type = searchParams.get('type') as TaskType | null;
    const category = searchParams.get('category') as Category | null;
    const hasImage = searchParams.get('hasImage');
    const search = searchParams.get('search') || undefined;
    const codingOnly = searchParams.get('codingOnly') === 'true';

    // Ensure the split is loaded
    if (!datasetLoader.isLoaded(split)) {
      await datasetLoader.preloadSplit(split);
    }

    // Build filters
    const filters: {
      type?: TaskType;
      category?: Category;
      hasImage?: boolean;
      search?: string;
      codingOnly?: boolean;
    } = { codingOnly };

    if (type) filters.type = type;
    if (category) filters.category = category;
    if (hasImage !== null) filters.hasImage = hasImage === 'true';
    if (search) filters.search = search;

    const result = datasetLoader.filterExamples(split, filters, limit, offset);

    return NextResponse.json({
      examples: result.examples,
      total: result.total,
      split,
      offset,
      limit,
      hasMore: offset + result.examples.length < result.total,
    });
  } catch (error) {
    console.error('Examples API error:', error);

    const errorMessage =
      error instanceof Error ? error.message : 'Failed to load examples';

    return NextResponse.json(
      { error: errorMessage, examples: [], total: 0 },
      { status: 500 }
    );
  }
}

// Endpoint to get split info
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    if (body.action === 'getSplitInfo') {
      const splitInfo = await datasetLoader.getSplitInfo();
      return NextResponse.json({ splitInfo });
    }
    
    if (body.action === 'getCodingCount') {
      const split = (body.split || 'test') as 'train' | 'validation' | 'test';
      
      // Ensure the split is loaded
      if (!datasetLoader.isLoaded(split)) {
        await datasetLoader.preloadSplit(split);
      }
      
      const codingProblems = datasetLoader.getCodingProblems(split);
      return NextResponse.json({ count: codingProblems.length, split });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (error) {
    console.error('Examples API POST error:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}
