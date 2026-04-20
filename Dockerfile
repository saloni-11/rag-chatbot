# ============================================================
# Dockerfile — Multi-stage build
# ============================================================
# This Dockerfile builds your entire app in two stages:
#
# Stage 1 (frontend-build): Installs Node.js, builds the React app
#   into static HTML/CSS/JS files (the 'dist' folder).
#
# Stage 2 (production): Installs Python dependencies, copies the
#   built frontend files, and runs FastAPI serving both the API
#   and the static frontend from a single container.
#
# WHY MULTI-STAGE?
#   The frontend build needs Node.js (~300 MB) but the final
#   container doesn't. Multi-stage builds let you use Node.js in
#   stage 1, then copy only the output (tiny dist/ folder) into
#   stage 2. The final image has Python but no Node.js — much smaller.
#
# HOW TO BUILD AND RUN:
#   docker build -t rag-chatbot .
#   docker run -p 8000:8000 --env-file .env rag-chatbot
#   Then open http://localhost:8000
# ============================================================

# ── Stage 1: Build the React frontend ────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

# Copy package files first (Docker caches this layer if they don't change)
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the frontend source
COPY frontend/ ./

# Build the production bundle (outputs to dist/)
RUN npm run build


# ── Stage 2: Python production image ─────────────────
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# (done before copying code so this layer is cached when only code changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Copy the built frontend from stage 1
# FastAPI will serve these static files at the root URL
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy config files
COPY pytest.ini ./

# Create a non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

# Expose the port FastAPI runs on
EXPOSE 8000

# Health check — Docker/Kubernetes will ping this to know if the
# container is alive. If it returns non-200, the container is restarted.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible from outside the container
# (127.0.0.1 would only be accessible inside the container)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
