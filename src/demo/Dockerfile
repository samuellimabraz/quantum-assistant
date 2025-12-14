FROM node:20-slim AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci --legacy-peer-deps

COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM python:3.11-slim AS runner

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    git-lfs \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml ./

RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -e . && \
    rm -rf /root/.cache/uv

COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static
COPY --from=builder --chown=app:app /app/public ./public

RUN chown -R app:app /app

USER app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=7860
ENV HOSTNAME=0.0.0.0
ENV PYTHON_PATH=/app/.venv/bin/python

EXPOSE 7860

CMD ["node", "server.js"]

