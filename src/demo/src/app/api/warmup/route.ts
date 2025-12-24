import { NextResponse } from 'next/server';

export async function POST(): Promise<NextResponse> {
  const baseUrl = process.env.DEMO_MODEL_URL || 'http://localhost:8000/v1';
  const apiKey = process.env.DEMO_API_KEY || '';
  const modelName = process.env.DEMO_MODEL_NAME || 'default';

  console.log('[Warmup] Starting warmup...');
  console.log('[Warmup] Base URL:', baseUrl);

  const runpodMatch = baseUrl.match(/https:\/\/api\.runpod\.ai\/v2\/([^/]+)/);
  
  if (!runpodMatch) {
    console.log('[Warmup] Not a RunPod endpoint, skipping');
    return NextResponse.json({ 
      status: 'skipped', 
      message: 'Not a RunPod endpoint',
    });
  }

  const endpointId = runpodMatch[1];
  console.log('[Warmup] Endpoint ID:', endpointId);

  try {
    const healthUrl = `https://api.runpod.ai/v2/${endpointId}/health`;
    let healthData = null;
    
    try {
      const healthResponse = await fetch(healthUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
        signal: AbortSignal.timeout(5000),
      });

      if (healthResponse.ok) {
        healthData = await healthResponse.json();
        console.log('[Warmup] Health:', JSON.stringify(healthData));
        
        if (healthData.workers?.idle > 0) {
          console.log('[Warmup] Idle workers available');
          return NextResponse.json({
            status: 'ready',
            message: 'Workers already available',
            workers: healthData.workers,
          });
        }
        
        if (healthData.workers?.initializing > 0) {
          console.log('[Warmup] Workers already initializing');
          return NextResponse.json({
            status: 'warming',
            message: 'Workers already starting',
            workers: healthData.workers,
          });
        }
      }
    } catch (e) {
      console.log('[Warmup] Health check error:', e);
    }

    const openaiUrl = `${baseUrl}/chat/completions`;
    console.log('[Warmup] Sending to OpenAI endpoint:', openaiUrl);
    
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), 5000);
    
    try {
      const warmupResponse = await fetch(openaiUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelName,
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 1,
          stream: false,
        }),
        signal: abortController.signal,
      });

      clearTimeout(timeoutId);
      
      console.log('[Warmup] Response status:', warmupResponse.status);
      
      return NextResponse.json({
        status: warmupResponse.status === 200 ? 'ready' : 'warming',
        message: warmupResponse.status === 200 
          ? 'Model responded (was ready)' 
          : 'Request queued, worker starting',
        httpStatus: warmupResponse.status,
        workers: healthData?.workers,
      });
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      if ((fetchError as Error).name === 'AbortError') {
        console.log('[Warmup] Request sent (aborted wait - worker starting)');
        return NextResponse.json({
          status: 'warming',
          message: 'Request sent, worker starting',
          workers: healthData?.workers,
        });
      }
      
      throw fetchError;
    }

  } catch (error) {
    console.error('[Warmup] Error:', error);
    return NextResponse.json({
      status: 'error',
      message: error instanceof Error ? error.message : 'Warmup failed',
    }, { status: 500 });
  }
}

export async function GET(): Promise<NextResponse> {
  const baseUrl = process.env.DEMO_MODEL_URL || 'http://localhost:8000/v1';
  const apiKey = process.env.DEMO_API_KEY || '';

  const runpodMatch = baseUrl.match(/https:\/\/api\.runpod\.ai\/v2\/([^/]+)/);
  
  if (!runpodMatch) {
    return NextResponse.json({ 
      ready: true,
      message: 'Not a RunPod endpoint' 
    });
  }

  const endpointId = runpodMatch[1];
  const healthUrl = `https://api.runpod.ai/v2/${endpointId}/health`;

  try {
    const response = await fetch(healthUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(10000), 
    });

    if (!response.ok) {
      console.log('[Warmup GET] Health check failed:', response.status);
      return NextResponse.json({ ready: false, message: 'Health check failed' });
    }

    const health = await response.json();
    console.log('[Warmup GET] Health:', JSON.stringify(health));
    
    const idleWorkers = health.workers?.idle || 0;
    const readyWorkers = health.workers?.ready || 0;
    const runningWorkers = health.workers?.running || 0;
    const initializingWorkers = health.workers?.initializing || 0;
    const throttledWorkers = health.workers?.throttled || 0;
    
    const isReady = idleWorkers > 0 || readyWorkers > 0;
    const isWarming = initializingWorkers > 0;
    const isBusy = runningWorkers > 0 && !isReady;
    const jobsInQueue = health.jobs?.inQueue || 0;
    const jobsInProgress = health.jobs?.inProgress || 0;

    return NextResponse.json({
      ready: isReady,
      warming: isWarming,
      busy: isBusy,
      jobsInQueue,
      jobsInProgress,
      workers: {
        idle: idleWorkers,
        ready: readyWorkers,
        running: runningWorkers,
        initializing: initializingWorkers,
        throttled: throttledWorkers,
      },
    });
  } catch (error) {
    const isTimeout = error instanceof Error && error.name === 'TimeoutError';
    if (!isTimeout) {
      console.error('[Warmup GET] Error:', error);
    }
    return NextResponse.json({ 
      ready: false, 
      warming: true, 
      message: isTimeout ? 'Health check timed out' : 'Check failed'
    });
  }
}
