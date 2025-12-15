'use client';

import { useState, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Send, Image as ImageIcon, X, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface MessageInputProps {
  onSend: (message: string, imageUrl?: string, imageBase64?: string) => void;
  isLoading: boolean;
  placeholder?: string;
}

export interface MessageInputRef {
  setContent: (text: string, imageUrl?: string) => void;
  clear: () => void;
}

export const MessageInput = forwardRef<MessageInputRef, MessageInputProps>(
  function MessageInput(
    {
      onSend,
      isLoading,
      placeholder = 'Ask about quantum computing, Qiskit, or upload a circuit diagram...',
    },
    ref
  ) {
    const [message, setMessage] = useState('');
    const [imageBase64, setImageBase64] = useState<string | null>(null);
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useImperativeHandle(ref, () => ({
      setContent: (text: string, url?: string) => {
        setMessage(text);
        if (url) {
          setImageUrl(url);
          setImagePreview(url);
          setImageBase64(null);
        }
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
          setTimeout(() => {
            if (textareaRef.current) {
              textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
            }
          }, 0);
        }
      },
      clear: () => {
        setMessage('');
        setImageBase64(null);
        setImageUrl(null);
        setImagePreview(null);
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
        }
      },
    }));

    const handleSubmit = useCallback(() => {
      if ((!message.trim() && !imageBase64 && !imageUrl) || isLoading) return;

      onSend(message.trim(), imageUrl || undefined, imageBase64 || undefined);
      setMessage('');
      setImageBase64(null);
      setImageUrl(null);
      setImagePreview(null);

      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }, [message, imageBase64, imageUrl, isLoading, onSend]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    };

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        const base64 = result.split(',')[1];
        setImageBase64(base64);
        setImageUrl(null);
        setImagePreview(result);
      };
      reader.readAsDataURL(file);
    };

    const removeImage = () => {
      setImageBase64(null);
      setImageUrl(null);
      setImagePreview(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    };

    const adjustTextareaHeight = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const textarea = e.target;
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
      setMessage(textarea.value);
    };

    const hasContent = message.trim() || imageBase64 || imageUrl;

    return (
      <div className="bg-zinc-800/60 border border-zinc-700/50 rounded-xl p-3">
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <img
              src={imagePreview}
              alt="Upload preview"
              className="h-24 rounded-lg border border-zinc-700/50 object-contain bg-zinc-900"
            />
            <button
              onClick={removeImage}
              className="absolute -top-2 -right-2 p-1 bg-red-600/80 rounded-full hover:bg-red-600 transition-colors"
            >
              <X className="w-3 h-3 text-white" />
            </button>
          </div>
        )}

        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className={clsx(
              'p-3 rounded-lg transition-all duration-200',
              'hover:bg-zinc-700/50 text-zinc-500 hover:text-zinc-300',
              isLoading && 'opacity-50 cursor-not-allowed'
            )}
            title="Upload image"
          >
            <ImageIcon className="w-5 h-5" />
          </button>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={adjustTextareaHeight}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            rows={1}
            className={clsx(
              'flex-1 bg-transparent border-none outline-none resize-none',
              'text-zinc-200 placeholder:text-zinc-500',
              'min-h-[44px] max-h-[200px] py-3',
              isLoading && 'opacity-50'
            )}
          />

          <button
            onClick={handleSubmit}
            disabled={!hasContent || isLoading}
            className={clsx(
              'p-3 rounded-lg transition-all duration-200',
              hasContent
                ? 'bg-teal-700/80 hover:bg-teal-600/80 text-white'
                : 'bg-zinc-700/50 text-zinc-500',
              isLoading && 'opacity-50 cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    );
  }
);
