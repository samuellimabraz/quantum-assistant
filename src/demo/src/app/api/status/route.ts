import { NextResponse } from 'next/server';

export interface RunPodHealth {
  jobs: {
    completed: number;
    failed: number;
    inProgress: number;
    inQueue: number;
    retried: number;
  };
  workers: {
    idle: number;
    initializing: number;
    running: number;
    throttled: number;
  };
}

export interface StatusResponse {
  status: 'ready' | 'cold_start' | 'initializing' | 'processing' | 'unavailable';
  message: string;
  workers: {
    idle: number;
    running: number;
    initializing: number;
  };
  queue: {
    inProgress: number;
    inQueue: number;
  };
  estimatedWait?: number; // seconds
}

/**
 * Check RunPod endpoint health to provide user feedback during cold starts
 */
export async function GET(): Promise<NextResponse<StatusResponse>> {
  const baseUrl = process.env.DEMO_MODEL_URL || 'http://localhost:8000/v1';
  const apiKey = process.env.DEMO_API_KEY || '';

  // Extract RunPod endpoint URL from the vLLM base URL
  // vLLM URL format: https://api.runpod.ai/v2/{endpoint_id}/openai/v1
  // Health URL format: https://api.runpod.ai/v2/{endpoint_id}/health
  const runpodMatch = baseUrl.match(/https:\/\/api\.runpod\.ai\/v2\/([^/]+)/);
  
  if (!runpodMatch) {
    // Not a RunPod endpoint, assume it's always ready (local/other provider)
    return NextResponse.json({
      status: 'ready',
      message: 'Model server ready',
      workers: { idle: 1, running: 0, initializing: 0 },
      queue: { inProgress: 0, inQueue: 0 },
    });
  }

  const endpointId = runpodMatch[1];
  const healthUrl = `https://api.runpod.ai/v2/${endpointId}/health`;

  try {
    const response = await fetch(healthUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      // Short timeout for health check
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return NextResponse.json({
        status: 'unavailable',
        message: 'Unable to check model status',
        workers: { idle: 0, running: 0, initializing: 0 },
        queue: { inProgress: 0, inQueue: 0 },
      });
    }

    const health: RunPodHealth = await response.json();
    
    const totalWorkers = health.workers.idle + health.workers.running + (health.workers.initializing || 0);
    const hasActiveWorkers = totalWorkers > 0;
    const hasIdleWorkers = health.workers.idle > 0;
    const isInitializing = (health.workers.initializing || 0) > 0;
    const hasQueuedJobs = health.jobs.inQueue > 0;
    const hasRunningJobs = health.jobs.inProgress > 0;

    let status: StatusResponse['status'];
    let message: string;
    let estimatedWait: number | undefined;

    if (hasIdleWorkers) {
      status = 'ready';
      message = 'Model ready';
    } else if (isInitializing) {
      status = 'initializing';
      message = 'Model loading...';
      estimatedWait = 30; // Typical vLLM model load time
    } else if (health.workers.running > 0) {
      status = 'processing';
      message = hasQueuedJobs 
        ? `Processing (${health.jobs.inQueue} in queue)` 
        : 'Processing request...';
      estimatedWait = hasQueuedJobs ? health.jobs.inQueue * 15 : undefined;
    } else if (!hasActiveWorkers && (hasQueuedJobs || hasRunningJobs)) {
      status = 'cold_start';
      message = 'Starting worker...';
      estimatedWait = 45; // Cold start + model load
    } else if (!hasActiveWorkers) {
      status = 'cold_start';
      message = 'Workers scaled to zero, will start on request';
      estimatedWait = 45;
    } else {
      status = 'ready';
      message = 'Model ready';
    }

    return NextResponse.json({
      status,
      message,
      workers: {
        idle: health.workers.idle,
        running: health.workers.running,
        initializing: health.workers.initializing || 0,
      },
      queue: {
        inProgress: health.jobs.inProgress,
        inQueue: health.jobs.inQueue,
      },
      estimatedWait,
    });
  } catch (error) {
    console.error('Health check error:', error);
    
    // Network error might indicate cold start
    return NextResponse.json({
      status: 'cold_start',
      message: 'Connecting to model server...',
      workers: { idle: 0, running: 0, initializing: 0 },
      queue: { inProgress: 0, inQueue: 0 },
      estimatedWait: 45,
    });
  }
}

