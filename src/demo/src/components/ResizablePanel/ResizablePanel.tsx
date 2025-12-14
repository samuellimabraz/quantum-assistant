'use client';

import { useState, useRef, useCallback, useEffect, ReactNode } from 'react';
import { clsx } from 'clsx';

interface ResizablePanelProps {
  children: ReactNode;
  defaultWidth: number;
  minWidth: number;
  maxWidth: number;
  side: 'left' | 'right';
  isOpen: boolean;
  isCollapsed: boolean;
  collapsedWidth: number;
  storageKey?: string;
  className?: string;
}

export function ResizablePanel({
  children,
  defaultWidth,
  minWidth,
  maxWidth,
  side,
  isOpen,
  isCollapsed,
  collapsedWidth,
  storageKey,
  className,
}: ResizablePanelProps) {
  const [width, setWidth] = useState(() => {
    if (typeof window !== 'undefined' && storageKey) {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed) && parsed >= minWidth && parsed <= maxWidth) {
          return parsed;
        }
      }
    }
    return defaultWidth;
  });

  const [isDragging, setIsDragging] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX;
    startWidthRef.current = width;
  }, [width]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;

    const diff = side === 'right'
      ? startXRef.current - e.clientX
      : e.clientX - startXRef.current;

    const newWidth = Math.min(maxWidth, Math.max(minWidth, startWidthRef.current + diff));
    setWidth(newWidth);
  }, [isDragging, minWidth, maxWidth, side]);

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      if (storageKey) {
        localStorage.setItem(storageKey, width.toString());
      }
    }
  }, [isDragging, storageKey, width]);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const currentWidth = isCollapsed ? collapsedWidth : width;

  return (
    <div
      ref={panelRef}
      style={{ width: currentWidth }}
      className={clsx(
        'relative flex-shrink-0 transition-[width]',
        isDragging ? 'duration-0' : 'duration-200',
        !isOpen && 'hidden lg:flex',
        className
      )}
    >
      {children}

      {!isCollapsed && isOpen && (
        <div
          onMouseDown={handleMouseDown}
          className={clsx(
            'absolute top-0 bottom-0 w-1 cursor-col-resize z-50',
            'hover:bg-teal-500/50 transition-colors',
            isDragging && 'bg-teal-500/70',
            side === 'right' ? '-left-0.5' : '-right-0.5'
          )}
        >
          <div
            className={clsx(
              'absolute top-1/2 -translate-y-1/2 w-4 h-16 -left-1.5',
              'flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity',
              isDragging && 'opacity-100'
            )}
          >
            <div className="w-1 h-8 rounded-full bg-zinc-600 flex flex-col items-center justify-center gap-1">
              <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
              <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
              <div className="w-0.5 h-1 rounded-full bg-zinc-400" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function useResizableWidth(
  defaultWidth: number,
  minWidth: number,
  maxWidth: number,
  storageKey?: string
) {
  const [width, setWidth] = useState(() => {
    if (typeof window !== 'undefined' && storageKey) {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed) && parsed >= minWidth && parsed <= maxWidth) {
          return parsed;
        }
      }
    }
    return defaultWidth;
  });

  const updateWidth = useCallback((newWidth: number) => {
    const clamped = Math.min(maxWidth, Math.max(minWidth, newWidth));
    setWidth(clamped);
    if (storageKey) {
      localStorage.setItem(storageKey, clamped.toString());
    }
  }, [minWidth, maxWidth, storageKey]);

  return [width, updateWidth] as const;
}

