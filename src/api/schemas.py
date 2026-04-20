"""
API Schemas — Phase 6: FastAPI Backend
=======================================
Pydantic models that define the shape of API requests and responses.

What is Pydantic?
  Pydantic is a data validation library. When you define a model like
  QueryRequest with a 'question' field of type str, Pydantic will:
    1. Reject requests that don't have a 'question' field
    2. Reject requests where 'question' isn't a string
    3. Auto-generate JSON schema for API documentation

Why not just use plain dicts?
  Dicts don't validate anything — request["question"] could be missing,
  could be an integer, could be None. With Pydantic, if the data doesn't
  match the schema, FastAPI returns a 422 error with a clear message
  BEFORE your code even runs. This is defensive programming — catch
  bad input at the door, not deep inside your pipeline.

How FastAPI uses these:
  When you write `async def query(request: QueryRequest)`, FastAPI:
    1. Reads the incoming JSON body
    2. Validates it against QueryRequest
    3. If valid, passes it to your function as a typed object
    4. If invalid, returns 422 with details of what's wrong
  For responses, FastAPI uses response_model to:
    1. Filter out any extra fields your code might return
    2. Validate the response matches the schema
    3. Generate accurate API docs showing the response shape
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# ── Request Models ───────────────────────────────────


class QueryRequest(BaseModel):
    """
    What the client sends when asking a question.

    Field(...) means the field is required — no default value.
    min_length=1 prevents empty strings from passing validation.
    The 'examples' show up in the auto-generated API docs.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the RAG chatbot",
        examples=["What is self-attention in transformers?"],
    )


# ── Response Models ──────────────────────────────────


class SourceChunk(BaseModel):
    """
    A single source chunk that contributed to the answer.

    This is what the user sees in the "sources" panel of the UI —
    it lets them verify the answer is grounded in real documents.
    """

    text: str = Field(description="Preview of the chunk content (first 500 chars)")
    file_name: str = Field(description="Name of the source document")
    score: Optional[float] = Field(
        default=None,
        description="Similarity score (0-1, higher = more relevant)",
    )


class QueryResponse(BaseModel):
    """
    What the API returns after processing a question.

    guardrail_action tells the frontend what happened:
      - "passed":          normal answer from the RAG pipeline
      - "scope_rejected":  question was outside AI/ML scope
      - "low_confidence":  retrieved chunks weren't relevant enough
      - "empty_query":     user sent a blank question
    """

    answer: str = Field(description="The generated answer")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )
    guardrail_action: str = Field(
        description="What the guardrails decided: passed, scope_rejected, "
        "low_confidence, or empty_query",
    )


class HealthResponse(BaseModel):
    """
    Response for the health check endpoint.

    Health checks are used by deployment platforms (HuggingFace Spaces,
    Docker, Kubernetes) to know if the service is running and ready.
    A load balancer pings /health periodically — if it returns non-200,
    the platform restarts the container.
    """

    status: str = Field(description="Service status: 'healthy' or 'unhealthy'")
    model: str = Field(description="The LLM model being used")
    index_loaded: bool = Field(
        description="Whether the ChromaDB index is loaded and ready"
    )
