'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

export type WarmupStatus = 'idle' | 'checking' | 'warming' | 'ready' | 'error';

interface WarmupState {
  status: WarmupStatus;
  message: string;
  workers: {
    idle: number;
    ready?: number;
    running: number;
    initializing: number;
  } | null;
}

interface UseWarmupReturn extends WarmupState {
  triggerWarmup: () => Promise<void>;
  checkStatus: () => Promise<void>;
}

let warmupInProgress = false;
let lastWarmupTime = 0;
const WARMUP_COOLDOWN = 5000; 

/**
 * Hook to pre-warm RunPod workers when user lands on the page.
 * Automatically triggers warmup on mount and polls for readiness.
 */
export function useWarmup(autoWarmup = true): UseWarmupReturn {
  const [state, setState] = useState<WarmupState>({
    status: 'idle',
    message: '',
    workers: null,
  });
  const mountedRef = useRef(true);

  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/warmup', { method: 'GET' });
      if (response.ok && mountedRef.current) {
        const data = await response.json();
        
        if (data.ready) {
          setState({
            status: 'ready',
            message: 'Model ready',
            workers: data.workers || null,
          });
          return;
        }
        
        if (data.warming) {
          setState({
            status: 'warming',
            message: 'Model starting...',
            workers: data.workers || null,
          });
          return;
        }

        if (data.busy) {
          setState({
            status: 'warming',
            message: 'Model busy...',
            workers: data.workers || null,
          });
          return;
        }

        setState((prev) => ({
          ...prev,
          workers: data.workers || null,
        }));
      }
    } catch {
      // Silently fail status checks
    }
  }, []);

  const triggerWarmup = useCallback(async () => {
    // Prevent duplicate warmups
    const now = Date.now();
    if (warmupInProgress || (now - lastWarmupTime) < WARMUP_COOLDOWN) {
      await checkStatus();
      return;
    }
    
    warmupInProgress = true;
    lastWarmupTime = now;
    
    setState((prev) => ({ ...prev, status: 'checking', message: 'Checking model status...' }));

    try {
      const response = await fetch('/api/warmup', { method: 'POST' });
      const data = await response.json();
      
      if (!mountedRef.current) return;

      if (data.status === 'ready') {
        setState({
          status: 'ready',
          message: 'Model ready',
          workers: data.workers || null,
        });
      } else if (data.status === 'warming') {
        setState({
          status: 'warming',
          message: data.message || 'Starting model...',
          workers: data.workers || null,
        });
      } else if (data.status === 'skipped') {
        setState({
          status: 'ready',
          message: '',
          workers: null,
        });
      } else if (data.status === 'error') {
        setState({
          status: 'error',
          message: data.message || 'Warmup failed',
          workers: null,
        });
      }
    } catch (error) {
      if (mountedRef.current) {
        setState({
          status: 'error',
          message: error instanceof Error ? error.message : 'Warmup request failed',
          workers: null,
        });
      }
    } finally {
      warmupInProgress = false;
    }
  }, [checkStatus]);

  useEffect(() => {
    mountedRef.current = true;
    
    if (autoWarmup) {
      triggerWarmup();
    }
    
    return () => {
      mountedRef.current = false;
    };
  }, [autoWarmup, triggerWarmup]);

  useEffect(() => {
    if (state.status !== 'warming') return;

    const pollInterval = setInterval(() => {
      checkStatus();
    }, 5000); 

    const timeout = setTimeout(() => {
      clearInterval(pollInterval);
    }, 180000);

    return () => {
      clearInterval(pollInterval);
      clearTimeout(timeout);
    };
  }, [state.status, checkStatus]);

  return {
    ...state,
    triggerWarmup,
    checkStatus,
  };
}

