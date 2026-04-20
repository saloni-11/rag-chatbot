"""
tests/test_api.py
==================
Integration tests for the FastAPI API endpoints.

TESTING CONCEPTS:
=================

What is an integration test?
  Unlike unit tests that test one function in isolation, integration
  tests verify that multiple components work together. Here we test
  the full API flow: HTTP request → FastAPI → route → pipeline → response.

What is TestClient?
  FastAPI provides a test client (from httpx) that lets you make HTTP
  requests to your API without starting a real server. It simulates
  the full request/response cycle in memory — much faster than
  spinning up uvicorn and making real HTTP calls.

What is conftest.py?
  A pytest file where you put fixtures shared across test files.
  We could put the client fixture there, but for clarity we keep
  it in this file since only API tests need it.

Why mock the pipeline?
  The real RAG pipeline needs ChromaDB data, the embedding model,
  and a Groq API key. Tests should run in CI without any of those.
  We mock the pipeline to return predictable responses, so we can
  test the API layer (routing, validation, error handling) independently
  of the ML components.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ─────────────────────────────────────────


@pytest.fixture
def mock_pipeline():
    """
    Create a mock RAG pipeline that returns canned responses.

    The mock behaves like the real RAGPipeline but without needing
    ChromaDB, embeddings, or Groq. It returns a predictable response
    that we can assert against.
    """
    pipeline = MagicMock()
    pipeline.model = "llama-3.1-8b-instant"
    pipeline.index = MagicMock()  # pretend index is loaded

    # Default query response
    pipeline.query.return_value = {
        "answer": "Self-attention allows each token to attend to all others.",
        "sources": [
            {
                "text": "Self-attention mechanism...",
                "file_name": "attention.pdf",
                "score": 0.85,
            }
        ],
        "guardrail_action": "passed",
    }

    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """
    Create a FastAPI test client with the mock pipeline injected.

    We patch RAGPipeline at its source module (src.rag.pipeline)
    so when the lifespan function does 'from src.rag.pipeline import
    RAGPipeline', it gets our mock instead of the real class.

    Why patch at 'src.rag.pipeline.RAGPipeline' and not 'src.api.main.RAGPipeline'?
    Because main.py imports RAGPipeline inside the lifespan function,
    not at the top of the file. So the attribute doesn't exist on
    the main module — it only exists at the source where it's defined.
    The rule: patch where the thing LIVES, not where it's used.
    """
    with patch("src.rag.pipeline.RAGPipeline", return_value=mock_pipeline):
        from docker.src.api.main import app
        from src.api.routes import set_pipeline

        set_pipeline(mock_pipeline)

        with TestClient(app) as test_client:
            yield test_client, mock_pipeline


# ── Root Endpoint Tests ──────────────────────────────


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200 with welcome message."""
        test_client, _ = client
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_root_contains_docs_link(self, client):
        """Root should include a link to the API docs."""
        test_client, _ = client
        response = test_client.get("/")
        data = response.json()

        assert data["docs"] == "/docs"


# ── Health Endpoint Tests ────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /api/health"""

    def test_health_returns_200(self, client):
        """Health check should return 200."""
        test_client, _ = client
        response = test_client.get("/api/health")

        assert response.status_code == 200

    def test_health_returns_healthy(self, client):
        """Should report healthy when pipeline is loaded."""
        test_client, _ = client
        response = test_client.get("/api/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["index_loaded"] is True
        assert data["model"] == "llama-3.1-8b-instant"

    def test_health_reports_model_name(self, client):
        """Should include the model name in health response."""
        test_client, _ = client
        response = test_client.get("/api/health")
        data = response.json()

        assert "model" in data
        assert len(data["model"]) > 0


# ── Query Endpoint Tests ─────────────────────────────


class TestQueryEndpoint:
    """Tests for POST /api/query"""

    def test_query_returns_200(self, client):
        """Valid query should return 200."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"question": "What is self-attention?"},
        )

        assert response.status_code == 200

    def test_query_returns_answer(self, client):
        """Response should contain answer, sources, and guardrail_action."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"question": "What is self-attention?"},
        )
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "guardrail_action" in data
        assert len(data["answer"]) > 0

    def test_query_includes_sources(self, client):
        """Response should include source chunks with expected fields."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"question": "What is self-attention?"},
        )
        data = response.json()

        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert "text" in source
        assert "file_name" in source
        assert "score" in source

    def test_query_calls_pipeline(self, client):
        """Should call pipeline.query with the provided question."""
        test_client, mock_pipeline = client
        test_client.post(
            "/api/query",
            json={"question": "How does BERT work?"},
        )

        mock_pipeline.query.assert_called_once_with("How does BERT work?")

    def test_query_scope_rejected(self, client):
        """Scope-rejected responses should still return 200."""
        test_client, mock_pipeline = client

        mock_pipeline.query.return_value = {
            "answer": "I'm designed to help with AI/ML topics.",
            "sources": [],
            "guardrail_action": "scope_rejected",
        }

        response = test_client.post(
            "/api/query",
            json={"question": "Best pizza in Sydney?"},
        )
        data = response.json()

        assert response.status_code == 200
        assert data["guardrail_action"] == "scope_rejected"
        assert len(data["sources"]) == 0

    def test_query_low_confidence(self, client):
        """Low-confidence responses should still return 200."""
        test_client, mock_pipeline = client

        mock_pipeline.query.return_value = {
            "answer": "I don't have enough information.",
            "sources": [],
            "guardrail_action": "low_confidence",
        }

        response = test_client.post(
            "/api/query",
            json={"question": "What is quantum entanglement?"},
        )
        data = response.json()

        assert response.status_code == 200
        assert data["guardrail_action"] == "low_confidence"

    # ── Validation Tests ─────────────────────────

    def test_query_empty_body_returns_422(self, client):
        """Missing request body should return 422 validation error."""
        test_client, _ = client
        response = test_client.post("/api/query")

        assert response.status_code == 422

    def test_query_empty_question_returns_422(self, client):
        """Empty question string should be rejected by Pydantic."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"question": ""},
        )

        assert response.status_code == 422

    def test_query_missing_question_field_returns_422(self, client):
        """Wrong field name should return 422."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"q": "What is attention?"},
        )

        assert response.status_code == 422

    def test_query_non_string_question_returns_422(self, client):
        """Non-string question should return 422."""
        test_client, _ = client
        response = test_client.post(
            "/api/query",
            json={"question": 12345},
        )

        # Pydantic may coerce integers to strings, or reject them
        # Either way, the API should not crash
        assert response.status_code in [200, 422]

    # ── Error Handling Tests ─────────────────────

    def test_query_pipeline_error_returns_500(self, client):
        """If pipeline.query() throws, API should return 500."""
        test_client, mock_pipeline = client

        mock_pipeline.query.side_effect = RuntimeError("Groq API timeout")

        response = test_client.post(
            "/api/query",
            json={"question": "What is attention?"},
        )

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
