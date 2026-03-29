"""
FastAPI App — Phase 6: FastAPI Backend
=======================================
The entry point for the API server. This file:
  1. Creates the FastAPI app
  2. Configures CORS (so the React frontend can call the API)
  3. Initialises the RAG pipeline once at startup
  4. Includes the API routes

How to run:
    uvicorn src.api.main:app --reload

    This tells uvicorn:
      - "src.api.main" → look in src/api/main.py
      - ":app"         → find the variable called 'app'
      - "--reload"     → restart when you change code (dev only)

What is CORS?
  Cross-Origin Resource Sharing. When your React frontend runs on
  http://localhost:5173 and the API runs on http://localhost:8000,
  the browser blocks requests between them by default (security).
  CORS tells the browser: "requests from localhost:5173 are allowed."

  Without CORS middleware, your React app would get:
    "Access to fetch at 'http://localhost:8000/api/query' from origin
     'http://localhost:5173' has been blocked by CORS policy"

What is a lifespan?
  FastAPI's lifespan context manager lets you run code at startup
  and shutdown. We use it to initialise the RAG pipeline ONCE when
  the server starts, rather than on the first request. This means:
    - The first user doesn't wait 10 seconds for initialization
    - If initialization fails, the server fails to start (clear error)
    - The pipeline object is shared across all requests (singleton)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ── Setup ────────────────────────────────────────────

# Add project root to path (needed when running with uvicorn)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


# ── Lifespan (startup / shutdown) ────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup (before 'yield') and once at
    shutdown (after 'yield').

    This is a Python async context manager — the same pattern as:
        async with open("file") as f:
            ...
    But applied to the server lifecycle.

    'yield' is the dividing line:
      - Code before yield = startup (initialise pipeline)
      - Code after yield  = shutdown (cleanup if needed)
    """
    logger.info("=" * 50)
    logger.info("Starting RAG Chatbot API server...")
    logger.info("=" * 50)

    # Import here (not at top) so the server shows a clear error
    # if dependencies are missing, rather than crashing on import
    from src.api.routes import set_pipeline
    from src.rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline()
        set_pipeline(pipeline)
        logger.info("RAG Pipeline initialised and injected into routes ✅")
    except Exception as e:
        logger.error(f"Failed to initialise pipeline: {e}")
        logger.error("Server will start but /api/query will return 500")

    yield  # ← server is running and handling requests between here...

    # ...and here (shutdown)
    logger.info("Shutting down RAG Chatbot API server")


# ── FastAPI App ──────────────────────────────────────

app = FastAPI(
    title="AI/ML Interview Prep RAG Chatbot",
    description=(
        "A RAG-powered chatbot for AI/ML interview preparation. "
        "Ask questions about machine learning, deep learning, NLP, "
        "and data analytics — answers are grounded in source documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── CORS Middleware ──────────────────────────────────

# Allow the React frontend to call this API.
# In production, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"] means ANY frontend can call this API.
    # For production, restrict to your domain:
    #   allow_origins=["https://your-app.hf.space"]
    allow_origins=["*"],
    allow_credentials=True,
    # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_methods=["*"],
    # Allow all headers (Content-Type, Authorization, etc.)
    allow_headers=["*"],
)


# ── Include Routes ───────────────────────────────────

from src.api.routes import router

app.include_router(router)


# ── Root endpoint ────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint — a quick confirmation the server is running.

    Visiting http://localhost:8000/ in a browser will show this.
    """
    return {
        "message": "AI/ML Interview Prep RAG Chatbot API",
        "docs": "/docs",
        "health": "/api/health",
    }