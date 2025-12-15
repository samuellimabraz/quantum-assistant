'use client';

import { useState } from 'react';
import {
    CheckCircle2,
    XCircle,
    Clock,
    ChevronDown,
    ChevronUp,
    Terminal,
    AlertTriangle,
    Copy,
    Check,
    Image as ImageIcon,
    Download,
    ZoomIn,
} from 'lucide-react';
import { clsx } from 'clsx';

export interface ExecutionResultData {
    success: boolean;
    output: string;
    error: string;
    executionTime: number;
    hasCircuitOutput?: boolean;
    images?: string[]; // Base64 encoded images
}

interface ExecutionResultProps {
    result: ExecutionResultData;
    isLoading?: boolean;
}

function ImageViewer({ images }: { images: string[] }) {
    const [selectedImage, setSelectedImage] = useState<number | null>(null);

    const handleDownload = (base64: string, index: number) => {
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${base64}`;
        link.download = `quantum_output_${index + 1}.png`;
        link.click();
    };

    return (
        <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs text-zinc-500 mb-2">
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Generated Figures ({images.length})</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {images.map((base64, idx) => (
                    <div
                        key={idx}
                        className="relative group rounded-lg overflow-hidden border border-zinc-700/50 bg-zinc-900"
                    >
                        <img
                            src={`data:image/png;base64,${base64}`}
                            alt={`Output figure ${idx + 1}`}
                            className="w-full h-auto cursor-pointer hover:opacity-90 transition-opacity"
                            onClick={() => setSelectedImage(idx)}
                        />
                        <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                onClick={() => setSelectedImage(idx)}
                                className="p-1.5 rounded bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
                                title="View full size"
                            >
                                <ZoomIn className="w-3.5 h-3.5 text-zinc-300" />
                            </button>
                            <button
                                onClick={() => handleDownload(base64, idx)}
                                className="p-1.5 rounded bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
                                title="Download image"
                            >
                                <Download className="w-3.5 h-3.5 text-zinc-300" />
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {/* Full-size image modal */}
            {selectedImage !== null && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
                    onClick={() => setSelectedImage(null)}
                >
                    <div className="relative max-w-4xl max-h-[90vh] overflow-auto">
                        <img
                            src={`data:image/png;base64,${images[selectedImage]}`}
                            alt={`Output figure ${selectedImage + 1}`}
                            className="max-w-full h-auto rounded-lg"
                            onClick={(e) => e.stopPropagation()}
                        />
                        <div className="absolute top-2 right-2 flex gap-2">
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleDownload(images[selectedImage], selectedImage);
                                }}
                                className="p-2 rounded-lg bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
                                title="Download image"
                            >
                                <Download className="w-4 h-4 text-zinc-300" />
                            </button>
                            <button
                                onClick={() => setSelectedImage(null)}
                                className="p-2 rounded-lg bg-zinc-800/90 hover:bg-zinc-700 transition-colors"
                                title="Close"
                            >
                                <XCircle className="w-4 h-4 text-zinc-300" />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export function ExecutionResult({ result, isLoading }: ExecutionResultProps) {
    const [isExpanded, setIsExpanded] = useState(true);
    const [copied, setCopied] = useState(false);

    const hasOutput = result.output.trim().length > 0;
    const hasError = result.error.trim().length > 0;
    const hasImages = result.images && result.images.length > 0;
    const outputToShow = hasError ? result.error : result.output;

    const handleCopy = async () => {
        await navigator.clipboard.writeText(outputToShow);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    if (isLoading) {
        return (
            <div className="mt-3 rounded-lg border border-zinc-700/50 bg-zinc-900/50 overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800/50 border-b border-zinc-700/50">
                    <div className="w-4 h-4 border-2 border-teal-500/30 border-t-teal-500 rounded-full animate-spin" />
                    <span className="text-xs font-medium text-zinc-400">Executing code...</span>
                </div>
                <div className="p-3">
                    <div className="flex items-center gap-2 text-zinc-500">
                        <Terminal className="w-4 h-4" />
                        <span className="text-sm">Running Python with Qiskit...</span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className={clsx(
            'mt-3 rounded-lg border overflow-hidden transition-all duration-200',
            result.success
                ? 'border-emerald-600/30 bg-emerald-950/20'
                : 'border-red-600/30 bg-red-950/20'
        )}>
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className={clsx(
                    'w-full flex items-center justify-between px-3 py-2 transition-colors',
                    result.success
                        ? 'bg-emerald-900/30 hover:bg-emerald-900/40'
                        : 'bg-red-900/30 hover:bg-red-900/40'
                )}
            >
                <div className="flex items-center gap-2">
                    {result.success ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                        <XCircle className="w-4 h-4 text-red-400" />
                    )}
                    <span className={clsx(
                        'text-xs font-medium',
                        result.success ? 'text-emerald-300' : 'text-red-300'
                    )}>
                        {result.success ? 'Execution Successful' : 'Execution Failed'}
                    </span>

                    <span className="flex items-center gap-1 text-xs text-zinc-500 ml-2">
                        <Clock className="w-3 h-3" />
                        {result.executionTime}ms
                    </span>

                    {hasImages && (
                        <span className="flex items-center gap-1 text-xs text-teal-400 ml-2">
                            <ImageIcon className="w-3 h-3" />
                            {result.images?.length} figure{result.images?.length !== 1 ? 's' : ''}
                        </span>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    {(hasOutput || hasError || hasImages) && (
                        <span className="text-xs text-zinc-500">
                            {isExpanded ? 'Hide' : 'Show'} output
                        </span>
                    )}
                    {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-zinc-500" />
                    ) : (
                        <ChevronDown className="w-4 h-4 text-zinc-500" />
                    )}
                </div>
            </button>

            {/* Output */}
            {isExpanded && (hasOutput || hasError || hasImages) && (
                <div className="relative">
                    {(hasOutput || hasError) && (
                        <div className="absolute right-2 top-2 z-10">
                            <button
                                onClick={handleCopy}
                                className="p-1.5 rounded bg-zinc-800/80 hover:bg-zinc-700 transition-colors"
                                title="Copy output"
                            >
                                {copied ? (
                                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                                ) : (
                                    <Copy className="w-3.5 h-3.5 text-zinc-400" />
                                )}
                            </button>
                        </div>
                    )}

                    <div className={clsx(
                        'p-3 font-mono text-sm',
                        result.success ? 'bg-zinc-900/50' : 'bg-zinc-900/50'
                    )}>
                        {hasError && (
                            <div className="flex items-start gap-2 mb-3 text-red-400">
                                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                <pre className="whitespace-pre-wrap break-words text-red-300">{result.error}</pre>
                            </div>
                        )}

                        {hasOutput && (
                            <pre className={clsx(
                                'whitespace-pre-wrap break-words mb-3',
                                result.hasCircuitOutput ? 'text-teal-300' : 'text-zinc-300'
                            )}>
                                {result.output}
                            </pre>
                        )}

                        {!hasOutput && !hasError && !hasImages && result.success && (
                            <span className="text-zinc-500 italic">
                                Code executed successfully with no output
                            </span>
                        )}

                        {/* Display generated images */}
                        {hasImages && (
                            <ImageViewer images={result.images!} />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
