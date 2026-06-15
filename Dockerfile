# Distil — single-image build for local Docker and Railway.
# Local embeddings (the default) are baked in at BUILD time, never written to the runtime
# volume (Railway mounts /data only at runtime — ARCHITECTURE §8).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY distil ./distil
COPY web ./web

# Core + provider + local embeddings + vector search. Drop embed-local if hosting tiny
# (set DISTIL_EMBEDDER=api) to shrink the image.
RUN pip install --upgrade pip && \
    pip install ".[anthropic,embed-local,vec,web]"

# Pre-download the default local embedding model at BUILD time so the runtime volume is never
# written during build and cold starts are fast. Override DISTIL_EMBED_MODEL as needed.
ARG DISTIL_EMBED_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${DISTIL_EMBED_MODEL}')" || true

# Data + KB live on a mounted volume at runtime (compose/Railway), not in the image.
ENV DISTIL_DB_PATH=/data/distil.db \
    DISTIL_KB_DIR=/data/kb

EXPOSE 8000

# Bind 0.0.0.0 and the injected $PORT (Railway) — default 8000 locally.
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
