/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  output: 'standalone',

  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'huggingface.co',
      },
      {
        protocol: 'https',
        hostname: '*.hf.space',
      },
      {
        protocol: 'https',
        hostname: 'datasets-server.huggingface.co',
      },
    ],
    unoptimized: process.env.NODE_ENV === 'production',
  },

  env: {
    NEXT_PUBLIC_HF_SPACE: process.env.NEXT_PUBLIC_HF_SPACE || '',
  },
};

module.exports = nextConfig;

