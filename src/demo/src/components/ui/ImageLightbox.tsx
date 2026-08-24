'use client';

import { useCallback, useEffect, useState } from 'react';
import { Download, X, ZoomIn } from 'lucide-react';
import { clsx } from 'clsx';

interface ImageLightboxProps {
  src: string;
  alt?: string;
  onClose: () => void;
  onDownload?: () => void;
}

export function ImageLightbox({ src, alt = 'Expanded image', onClose, onDownload }: ImageLightboxProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey, true);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey, true);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={alt}
    >
      <div className="relative max-w-5xl max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <div className="bg-zinc-100 rounded-lg p-3 sm:p-4 max-h-[85vh] overflow-auto">
          <img
            src={src}
            alt={alt}
            className="max-w-full max-h-[80vh] object-contain mx-auto rounded"
          />
        </div>
        <div className="absolute top-2 right-2 flex gap-2">
          {onDownload && (
            <button
              type="button"
              onClick={onDownload}
              className="p-2 rounded-lg bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
              title="Download image"
            >
              <Download className="w-4 h-4 text-zinc-300" />
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
            title="Close"
          >
            <X className="w-4 h-4 text-zinc-300" />
          </button>
        </div>
      </div>
    </div>
  );
}

interface ExpandableImageProps {
  src: string;
  alt?: string;
  className?: string;
  wrapperClassName?: string;
}

export function ExpandableImage({
  src,
  alt = 'Image',
  className,
  wrapperClassName,
}: ExpandableImageProps) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  return (
    <>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        onKeyDown={(e) => e.stopPropagation()}
        className={clsx(
          'relative group block overflow-hidden rounded-lg border border-zinc-700/50 bg-zinc-900',
          'cursor-zoom-in focus:outline-none focus-visible:ring-1 focus-visible:ring-teal-600/60',
          wrapperClassName
        )}
        title="View full size"
      >
        <img src={src} alt={alt} loading="lazy" className={className} />
        <span className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/40 transition-colors">
          <ZoomIn className="w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
        </span>
      </button>
      {open && <ImageLightbox src={src} alt={alt} onClose={close} />}
    </>
  );
}
