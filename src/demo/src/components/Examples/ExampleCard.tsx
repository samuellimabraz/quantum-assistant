'use client';

import { Image as ImageIcon, Code, FileQuestion, ChevronRight } from 'lucide-react';
import { clsx } from 'clsx';
import { TASK_LABELS, CATEGORY_LABELS } from '@/config/constants';
import type { DatasetExample } from '@/types';

interface ExampleCardProps {
  example: DatasetExample;
  onSelect: (example: DatasetExample) => void;
  isSelected?: boolean;
}

export function ExampleCard({ example, onSelect, isSelected }: ExampleCardProps) {
  const taskConfig = TASK_LABELS[example.type];
  const categoryConfig = CATEGORY_LABELS[example.category];

  const getTaskIcon = () => {
    switch (example.type) {
      case 'function_completion':
        return <Code className="w-3.5 h-3.5" />;
      case 'code_generation':
        return <Code className="w-3.5 h-3.5" />;
      case 'qa':
        return <FileQuestion className="w-3.5 h-3.5" />;
    }
  };

  const truncateText = (text: string, maxLength: number) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
  };

  // Muted badge colors
  const badgeColors: Record<string, string> = {
    function_completion: 'bg-emerald-900/30 text-emerald-400 border-emerald-700/30',
    code_generation: 'bg-blue-900/30 text-blue-400 border-blue-700/30',
    qa: 'bg-amber-900/30 text-amber-400 border-amber-700/30',
  };

  return (
    <button
      onClick={() => onSelect(example)}
      className={clsx(
        'w-full text-left p-3 rounded-lg transition-all duration-200 group',
        'border hover:border-teal-700/40',
        isSelected
          ? 'bg-teal-900/20 border-teal-700/40'
          : 'bg-zinc-800/50 border-zinc-700/30 hover:bg-zinc-800/80'
      )}
    >
      <div className="flex items-start gap-3">
        {example.hasImage && (
          <div className="flex-shrink-0 w-14 h-14 rounded-md bg-zinc-800 border border-zinc-700/50 flex items-center justify-center overflow-hidden">
            {example.imageUrl ? (
              <img
                src={example.imageUrl}
                alt=""
                className="w-full h-full object-cover"
                loading="lazy"
              />
            ) : (
              <ImageIcon className="w-5 h-5 text-zinc-500" />
            )}
          </div>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border', badgeColors[example.type])}>
              {getTaskIcon()}
              <span className="ml-1">{taskConfig.label}</span>
            </span>
            <span className="text-[10px] text-zinc-500">
              {categoryConfig}
            </span>
          </div>

          <p className="text-sm text-zinc-300 leading-snug font-mono">
            {truncateText(example.question, 120)}
          </p>
        </div>

        <ChevronRight
          className={clsx(
            'w-4 h-4 flex-shrink-0 transition-transform duration-200',
            'text-zinc-600 group-hover:text-teal-500',
            'group-hover:translate-x-1'
          )}
        />
      </div>
    </button>
  );
}
