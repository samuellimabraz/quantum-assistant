'use client';

import { Github, Database, Boxes, ExternalLink, MessageSquare, Code } from 'lucide-react';
import { clsx } from 'clsx';
import { PROJECT_CONFIG, LINKS } from '@/config/constants';
import type { AppMode } from '@/types';

interface HeaderProps {
  mode?: AppMode;
  onModeChange?: (mode: AppMode) => void;
}

interface BadgeProps {
  href: string;
  icon: React.ReactNode;
  label: string;
  variant: 'default' | 'accent' | 'highlight';
}

function Badge({ href, icon, label, variant }: BadgeProps) {
  const variantStyles = {
    default: 'bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-300 border-zinc-700/50',
    accent: 'bg-teal-900/30 hover:bg-teal-800/40 text-teal-400 border-teal-700/40',
    highlight: 'bg-amber-900/30 hover:bg-amber-800/40 text-amber-400 border-amber-700/40',
  };

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium 
                  transition-all duration-200 hover:scale-[1.02] border ${variantStyles[variant]}`}
    >
      {icon}
      <span>{label}</span>
      <ExternalLink className="w-3 h-3 opacity-50" />
    </a>
  );
}

interface ModeToggleProps {
  mode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
  return (
    <div className="flex items-center bg-zinc-800/60 rounded-lg p-1 border border-zinc-700/50">
      <button
        onClick={() => onModeChange('chat')}
        className={clsx(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
          mode === 'chat'
            ? 'bg-teal-600 text-white shadow-sm'
            : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50'
        )}
      >
        <MessageSquare className="w-3.5 h-3.5" />
        <span>Chat</span>
      </button>
      <button
        onClick={() => onModeChange('practice')}
        className={clsx(
          'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
          mode === 'practice'
            ? 'bg-teal-600 text-white shadow-sm'
            : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50'
        )}
      >
        <Code className="w-3.5 h-3.5" />
        <span>Practice</span>
      </button>
    </div>
  );
}

export function Header({ mode = 'chat', onModeChange }: HeaderProps) {
  return (
    <header className="bg-zinc-900/95 backdrop-blur-sm border-b border-zinc-800/80 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="text-lg font-semibold text-zinc-100 tracking-tight">
                {PROJECT_CONFIG.name}
              </h1>
              <p className="text-xs text-zinc-500">
                {PROJECT_CONFIG.description}
              </p>
            </div>
            {onModeChange && (
              <ModeToggle mode={mode} onModeChange={onModeChange} />
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Badge
              href={LINKS.github}
              icon={<Github className="w-3.5 h-3.5" />}
              label="GitHub"
              variant="default"
            />
            <Badge
              href={LINKS.dataset}
              icon={<Database className="w-3.5 h-3.5" />}
              label="Dataset"
              variant="highlight"
            />
            <Badge
              href={LINKS.models}
              icon={<Boxes className="w-3.5 h-3.5" />}
              label="Models"
              variant="accent"
            />
          </div>
        </div>
      </div>
    </header>
  );
}
