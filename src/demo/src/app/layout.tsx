import type { Metadata } from 'next';
import { PROJECT_CONFIG } from '@/config/constants';
import 'katex/dist/katex.min.css';
import './globals.css';

export const metadata: Metadata = {
  title: `${PROJECT_CONFIG.name} | Demo`,
  description: PROJECT_CONFIG.description,
  authors: [{ name: PROJECT_CONFIG.author }],
  keywords: [
    'quantum computing',
    'qiskit',
    'vision language model',
    'code generation',
    'multimodal AI',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950">
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
