"""
API Routes — Phase 6: FastAPI Backend
======================================
Maps URL paths to functions that call the RAG pipeline.

What is a router?
  A router groups related endpoints together. Instead of putting
  all endpoints directly on the app, you define them on a router
  and then include the router in the app. This keeps things organised
  as the API grows — you could have separate routers for /api/chat,
  /api/admin, /api/evaluation, etc.

What is @router.post()?
  A decorator that tells FastAPI: "when someone sends a POST request
  to this URL path, call this function." POST is used for queries
  because we're SENDING data (the question) to the server.

  GET is used for /health because we're just READING status — no
  data is being sent.

What does 'async def' mean?
  FastAPI is async-native. When your endpoint does:
    result = pipeline.query(question)
  and the pipeline calls Groq's API, the server is WAITING for Groq
  to respond. With 'async', the server can handle other requests
  during that wait instead of blocking. For a chatbot that calls
  an external API, this is important for handling multiple users.

  Note: our pipeline.query() is actually synchronous (blocking) because
  LlamaIndex's query engine is sync. FastAPI handles this gracefully
  by running it in a thread pool. True async would need async LlamaIndex
  calls — a future optimisation.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.rag.pipeline import RAGPipeline

# ── Router setup ─────────────────────────────────────
# prefix="/api" means all routes below start with /api
# tags=["chat"] groups them in the auto-generated docs
router = APIRouter(prefix="/api", tags=["chat"])

# ── Pipeline (set by main.py at startup) ─────────────
# This is a module-level variable that main.py populates once
# the pipeline is initialised. Routes access it as a shared singleton.
#
# Why not create it here?
#   Because pipeline initialisation is slow (~10 sec) and can fail
#   (e.g., missing API key, no ChromaDB data). We want to handle
#   that in the startup lifecycle, not when the first request arrives.
_pipeline: RAGPipeline = None


def set_pipeline(pipeline: RAGPipeline):
    """Called by main.py during startup to inject the pipeline."""
    global _pipeline
    _pipeline = pipeline


# ── Endpoints ────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Send a question to the RAG chatbot. Returns an answer "
    "grounded in the source documents, along with the source chunks used.",
)
async def query(request: QueryRequest):
    """
    Main chat endpoint.

    The flow:
      1. FastAPI validates the request body against QueryRequest
      2. We check the pipeline is ready
      3. We call pipeline.query() which runs:
         scope check → retrieve → confidence check → LLM → response
      4. FastAPI validates our response against QueryResponse
      5. The response is serialized to JSON and sent back

    HTTP status codes:
      200 — success (including guardrail rejections — those are valid responses)
      422 — invalid request (Pydantic validation failed)
      500 — server error (pipeline not ready, unexpected exception)
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline not initialised. Check server logs.",
        )

    try:
        logger.info(f"API query: {request.question[:100]}")
        result = _pipeline.query(request.question)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            guardrail_action=result["guardrail_action"],
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and the index is loaded. "
    "Used by deployment platforms to monitor service health.",
)
async def health_check():
    """
    Health check endpoint.

    Returns the service status, which model is loaded, and whether
    the ChromaDB index is available. Deployment platforms (HuggingFace
    Spaces, Docker health checks, Kubernetes liveness probes) call
    this periodically to decide whether to restart the container.
    """
    return HealthResponse(
        status="healthy" if _pipeline is not None else "unhealthy",
        model=_pipeline.model if _pipeline else "not loaded",
        index_loaded=_pipeline is not None and _pipeline.index is not None,
    )
