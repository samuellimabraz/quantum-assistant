'use client';

import { useWarmup, WarmupStatus } from '@/lib/hooks/useWarmup';
import { Cpu, Check, Loader2, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';

interface WarmupIndicatorProps {
  className?: string;
  showWhenReady?: boolean;
}

const STATUS_CONFIG: Record<WarmupStatus, {
  icon: React.ElementType;
  label: string;
  color: string;
  bgColor: string;
  animate?: boolean;
}> = {
  idle: {
    icon: Cpu,
    label: 'Checking model...',
    color: 'text-zinc-400',
    bgColor: 'bg-zinc-800/50',
  },
  checking: {
    icon: Loader2,
    label: 'Checking model...',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    animate: true,
  },
  warming: {
    icon: Cpu,
    label: 'Pre-warming model...',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    animate: true,
  },
  ready: {
    icon: Check,
    label: 'Model ready',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
  },
  error: {
    icon: AlertCircle,
    label: 'Warmup failed',
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
  },
};

/**
 * Small indicator showing the pre-warming status of the model.
 * Appears in the header or corner of the chat interface.
 */
export function WarmupIndicator({ className, showWhenReady = false }: WarmupIndicatorProps) {
  const { status, workers } = useWarmup(true);

  if (status === 'ready' && !showWhenReady) {
    return null;
  }

  if (status === 'idle') {
    return null;
  }

  const config = STATUS_CONFIG[status];
  const Icon = config.icon;

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium transition-all duration-300',
        config.bgColor,
        config.color,
        className
      )}
    >
      <Icon
        className={clsx(
          'w-3 h-3',
          config.animate && 'animate-spin'
        )}
      />
      <span>{config.label}</span>
      {workers && status === 'warming' && workers.initializing > 0 && (
        <span className="text-zinc-500">
          ({workers.initializing} starting)
        </span>
      )}
    </div>
  );
}

export function WarmupIndicatorCompact({ className }: { className?: string }) {
  const { status, message } = useWarmup(true);

  if (status === 'ready' || status === 'idle') {
    return null;
  }

  const config = STATUS_CONFIG[status];
  const Icon = config.icon;

  return (
    <div
      className={clsx(
        'relative group',
        className
      )}
      title={message || config.label}
    >
      <div
        className={clsx(
          'p-1.5 rounded-full',
          config.bgColor
        )}
      >
        <Icon
          className={clsx(
            'w-3.5 h-3.5',
            config.color,
            config.animate && 'animate-spin'
          )}
        />
      </div>
      
      {/* Pulse effect for warming */}
      {status === 'warming' && (
        <span className="absolute inset-0 rounded-full bg-amber-400/30 animate-ping" />
      )}
    </div>
  );
}

