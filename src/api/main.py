"""
FastAPI App — with production logging and static file serving
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

# ── Logging setup ────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
# File logging — JSON format for structured log analysis.
# Rotates at 10 MB, keeps 7 days of history.
# This gives you an audit trail of every query and guardrail action.
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    str(log_dir / "app.log"),
    rotation="10 MB",
    retention="7 days",
    serialize=True,
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Starting AI/ML Study Companion API server...")
    logger.info("=" * 50)

    from src.api.routes import set_pipeline
    from src.rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline()
        set_pipeline(pipeline)
        logger.info("RAG Pipeline initialised and injected into routes")
    except Exception as e:
        logger.error(f"Failed to initialise pipeline: {e}")
        logger.error("Server will start but /api/query will return 503")

    yield

    logger.info("Shutting down AI/ML Study Companion API server")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes import router  # noqa: E402

app.include_router(router)


# Serve React frontend in production (Docker), or show API info in development.
# In Docker, 'npm run build' creates frontend/dist/ — so if that folder exists,
# we serve the React app at '/'. If it doesn't exist (local dev), we show a
# simple JSON endpoint instead (the React app is served by Vite on :5173).
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dist), html=True),
        name="frontend",
    )
    logger.info(f"Serving frontend from {frontend_dist}")
else:

    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": "AI/ML Study Companion API",
            "docs": "/docs",
            "health": "/api/health",
        }
