# ============================================================
# Dockerfile — Multi-stage build for HuggingFace Spaces
# ============================================================
# Stage 1: Build React frontend
# Stage 2: Install Python deps, run ingestion, serve the app
#
# HuggingFace Spaces reads this Dockerfile, builds the image on
# their servers, and runs the container at a public URL.
#
# Local usage:
#   docker build -t study-companion .
#   docker run -p 8000:8000 --env-file .env study-companion
# ============================================================

# ── Stage 1: Build the React frontend ────────────────
FROM node:24-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python production image ─────────────────
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for building Python C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch (smaller than GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Copy the built React frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy config files
COPY pytest.ini ./

# Run data ingestion during build — this embeds all PDFs from
# data/raw/ into ChromaDB so the index is baked into the image.
# No need to run ingestion at startup or commit binary DB files.
ENV HF_HUB_OFFLINE=0
RUN python scripts/ingest_data.py
ENV HF_HUB_OFFLINE=1

# Create non-root user with write access to data and logs dirs
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app/data \
    && mkdir -p /app/logs && chown -R appuser:appuser /app/logs
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
