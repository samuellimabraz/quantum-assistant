'use client';

import { useEffect, useState, useCallback } from 'react';
import { Loader2, Server, Cpu, Zap, Clock, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import type { StatusResponse } from '@/app/api/status/route';

type LoadingPhase = 
  | 'sending'      // Initial request sent
  | 'cold_start'   // Workers starting up
  | 'initializing' // Model loading
  | 'processing'   // Actively generating
  | 'streaming';   // Receiving response

interface LoadingStatusProps {
  isLoading: boolean;
  hasStartedStreaming: boolean;
  onStatusChange?: (status: StatusResponse | null) => void;
}

const PHASE_CONFIG: Record<LoadingPhase, { 
  icon: React.ElementType; 
  label: string; 
  color: string;
  pulseColor: string;
}> = {
  sending: {
    icon: Zap,
    label: 'Sending request...',
    color: 'text-blue-400',
    pulseColor: 'bg-blue-400',
  },
  cold_start: {
    icon: Server,
    label: 'Starting worker...',
    color: 'text-amber-400',
    pulseColor: 'bg-amber-400',
  },
  initializing: {
    icon: Cpu,
    label: 'Loading model...',
    color: 'text-purple-400',
    pulseColor: 'bg-purple-400',
  },
  processing: {
    icon: Loader2,
    label: 'Generating response...',
    color: 'text-teal-400',
    pulseColor: 'bg-teal-400',
  },
  streaming: {
    icon: Zap,
    label: 'Receiving...',
    color: 'text-emerald-400',
    pulseColor: 'bg-emerald-400',
  },
};

export function LoadingStatus({ isLoading, hasStartedStreaming, onStatusChange }: LoadingStatusProps) {
  const [phase, setPhase] = useState<LoadingPhase>('sending');
  const [elapsedTime, setElapsedTime] = useState(0);
  const [estimatedWait, setEstimatedWait] = useState<number | undefined>();
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Poll for status while loading
  const checkStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const status: StatusResponse = await response.json();
        onStatusChange?.(status);
        
        // Map status to phase
        if (status.status === 'cold_start') {
          setPhase('cold_start');
          setStatusMessage(status.message);
        } else if (status.status === 'initializing') {
          setPhase('initializing');
          setStatusMessage(status.message);
        } else if (status.status === 'processing') {
          setPhase('processing');
          setStatusMessage(status.message);
        }
        
        if (status.estimatedWait) {
          setEstimatedWait(status.estimatedWait);
        }
      }
    } catch {
      // Silently fail, keep current phase
    }
  }, [onStatusChange]);

  // Start polling when loading starts
  useEffect(() => {
    if (!isLoading) {
      setPhase('sending');
      setElapsedTime(0);
      setEstimatedWait(undefined);
      setStatusMessage('');
      return;
    }

    // Initial status check
    checkStatus();

    // Poll every 2 seconds while loading and not streaming
    const statusInterval = setInterval(() => {
      if (!hasStartedStreaming) {
        checkStatus();
      }
    }, 2000);

    // Track elapsed time
    const timeInterval = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(timeInterval);
    };
  }, [isLoading, hasStartedStreaming, checkStatus]);

  // Update phase based on streaming state
  useEffect(() => {
    if (hasStartedStreaming) {
      setPhase('streaming');
    }
  }, [hasStartedStreaming]);

  // After 3 seconds without response, likely a cold start
  useEffect(() => {
    if (isLoading && !hasStartedStreaming && elapsedTime >= 3 && phase === 'sending') {
      setPhase('cold_start');
    }
  }, [isLoading, hasStartedStreaming, elapsedTime, phase]);

  if (!isLoading) return null;

  const config = PHASE_CONFIG[phase];
  const Icon = config.icon;
  const showEstimate = estimatedWait && phase !== 'streaming';
  
  // Calculate progress percentage
  const progress = estimatedWait ? Math.min((elapsedTime / estimatedWait) * 100, 95) : undefined;

  return (
    <div className="flex flex-col gap-2">
      {/* Main status indicator */}
      <div className="flex items-center gap-3">
        {/* Animated icon */}
        <div className="relative">
          <div className={clsx(
            'w-8 h-8 rounded-lg flex items-center justify-center',
            'bg-zinc-800/80 border border-zinc-700/50'
          )}>
            <Icon className={clsx(
              'w-4 h-4',
              config.color,
              phase !== 'streaming' && 'animate-pulse'
            )} />
          </div>
          {/* Pulse effect for cold start */}
          {(phase === 'cold_start' || phase === 'initializing') && (
            <span className={clsx(
              'absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full',
              config.pulseColor,
              'animate-ping opacity-75'
            )} />
          )}
        </div>

        {/* Status text */}
        <div className="flex-1">
          <div className={clsx('text-sm font-medium', config.color)}>
            {statusMessage || config.label}
          </div>
          
          {/* Time indicator */}
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Clock className="w-3 h-3" />
            <span>{formatTime(elapsedTime)}</span>
            {showEstimate && (
              <>
                <span className="text-zinc-600">•</span>
                <span>~{formatTime(Math.max(0, estimatedWait - elapsedTime))} remaining</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Progress bar for cold start */}
      {showEstimate && progress !== undefined && (
        <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
          <div 
            className={clsx(
              'h-full rounded-full transition-all duration-1000',
              phase === 'cold_start' && 'bg-amber-500/50',
              phase === 'initializing' && 'bg-purple-500/50',
              phase === 'processing' && 'bg-teal-500/50'
            )}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Cold start explanation */}
      {phase === 'cold_start' && elapsedTime >= 5 && (
        <div className="flex items-start gap-2 p-2 bg-amber-500/5 border border-amber-500/20 rounded-lg text-xs">
          <AlertCircle className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
          <p className="text-zinc-400">
            <span className="text-amber-400 font-medium">Cold start detected.</span>{' '}
            The model is scaling up from zero. This typically takes 30-60 seconds on first request.
          </p>
        </div>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

