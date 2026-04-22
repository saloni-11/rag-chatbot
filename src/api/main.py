"""
FastAPI App — Phase 8 Update
==============================
Updated to serve the React frontend's built files in production.

In development:
  You run two servers — Vite (:5173) serves the frontend,
  FastAPI (:8000) serves the API. Vite proxies API calls.

In production (Docker):
  There's no Vite server. FastAPI serves BOTH:
    - /api/* routes → RAG pipeline (as before)
    - Everything else → React's built static files (index.html, JS, CSS)

  The React build (npm run build) outputs to frontend/dist/.
  We mount that folder with FastAPI's StaticFiles middleware.

How to run:
  Dev:        uvicorn src.api.main:app --reload
  Production: docker-compose up --build
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Setup ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


# ── Lifespan ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Starting RAG Chatbot API server...")
    logger.info("=" * 50)

    from src.api.routes import set_pipeline
    from src.rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline()
        set_pipeline(pipeline)
        logger.info("RAG Pipeline initialised and injected into routes")
    except Exception as e:
        logger.error(f"Failed to initialise pipeline: {e}")
        logger.error("Server will start but /api/query will return 500")

    yield

    logger.info("Shutting down RAG Chatbot API server")


# ── FastAPI App ──────────────────────────────────────

app = FastAPI(
    title="AI/ML Study Companion",
    description=(
        "A RAG-powered study companion for AI/ML learning. "
        "Ask questions about machine learning, deep learning, NLP, "
        "and data analytics — answers are grounded in source documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API Routes ───────────────────────────────
from src.api.routes import router  # noqa: E402

app.include_router(router)


# ── Root Endpoint ────────────────────────────────────
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "AI/ML Interview Prep RAG Chatbot API",
        "docs": "/docs",
        "health": "/api/health",
    }


# ── Serve React Frontend (production only) ───────────
# In Docker, the built React files live in frontend/dist/.
# We mount them AFTER the API routes so /api/* takes priority.
# The 'html=True' flag means requests to '/' serve index.html.
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dist), html=True),
        name="frontend",
    )
    logger.info(f"Serving frontend from {frontend_dist}")
