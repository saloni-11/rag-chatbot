"""
tests/test_guardrails.py
=========================
Unit tests for the guardrails module (scope check, confidence, filtering).

TESTING CONCEPTS:
=================

What is mocking?
  When your code depends on something expensive or external (like an
  embedding model that takes seconds to load), you replace it with a
  fake version in tests. This is called mocking.

  unittest.mock.patch replaces a function/class with a mock object
  for the duration of the test. After the test, the original is restored.

Why mock the embedding model?
  1. Speed — loading sentence-transformers takes ~5 seconds. Tests should
     run in milliseconds.
  2. Isolation — we're testing guardrails logic, not the embedding model.
     If the test fails, we want to know the guardrails code is broken,
     not that the model download failed.
  3. Determinism — real embeddings can vary slightly across runs. Mock
     embeddings are always the same, so tests are reproducible.

What is @pytest.fixture?
  A setup function that pytest injects into tests. We use it to create
  a Guardrails instance with mocked embeddings so every test starts
  from the same known state.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

# ── Helpers ──────────────────────────────────────────


def make_mock_embedding(dim=384, seed=42):
    """Create a deterministic fake embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


def make_node_with_score(text, file_name, score):
    """Create a NodeWithScore for testing confidence checks."""
    node = TextNode(text=text, metadata={"file_name": file_name})
    return NodeWithScore(node=node, score=score)


# ── Fixtures ─────────────────────────────────────────


@pytest.fixture
def guardrails():
    """
    Create a Guardrails instance with mocked embedding model.

    We patch get_embedding_model so it doesn't load the real
    sentence-transformers model (which would be slow and require
    the model to be downloaded).

    The mock returns:
      - get_text_embedding_batch: returns random vectors for reference phrases
      - get_query_embedding: returns a vector we control per test
    """
    with patch("src.rag.guardrails.get_embedding_model") as mock_get_model:
        mock_model = MagicMock()

        # Return fake embeddings for the reference phrases
        # (called during Guardrails.__init__)
        from src.rag.guardrails import SCOPE_REFERENCE_PHRASES

        num_refs = len(SCOPE_REFERENCE_PHRASES)
        fake_ref_embeddings = [make_mock_embedding(seed=i) for i in range(num_refs)]
        mock_model.get_text_embedding_batch.return_value = fake_ref_embeddings

        mock_get_model.return_value = mock_model

        from src.rag.guardrails import Guardrails

        g = Guardrails()
        g._embed_model = mock_model

        yield g, mock_model


# ── Scope Check Tests ────────────────────────────────


class TestScopeCheck:
    """Tests for the scope checking guardrail (Layer 1)."""

    def test_in_scope_question_passes(self, guardrails):
        """An AI/ML question should pass scope check."""
        g, mock_model = guardrails

        # Return an embedding very similar to one of the reference phrases
        # by returning the first reference embedding itself
        mock_model.get_query_embedding.return_value = g._reference_matrix[0].tolist()

        in_scope, message = g.check_scope("What is backpropagation?")

        assert in_scope is True
        assert message is None

    def test_out_of_scope_question_rejected(self, guardrails):
        """A non-AI/ML question should be rejected."""
        g, mock_model = guardrails

        # Return an embedding that's orthogonal to all references
        # (zero similarity = completely unrelated)
        orthogonal = np.zeros(384, dtype=np.float32)
        orthogonal[0] = 1.0  # point in a direction no reference points
        mock_model.get_query_embedding.return_value = orthogonal.tolist()

        in_scope, message = g.check_scope("Best pizza in Sydney?")

        assert in_scope is False
        assert message is not None
        assert "AI/ML" in message

    def test_empty_question_handling(self, guardrails):
        """Scope check should handle empty-ish strings."""
        g, mock_model = guardrails

        # Even for an empty-like query, the embedding model is called
        mock_model.get_query_embedding.return_value = make_mock_embedding(seed=999)

        # Should not crash — returns either True or False
        in_scope, message = g.check_scope("   ")
        assert isinstance(in_scope, bool)


# ── Confidence Check Tests ───────────────────────────


class TestConfidenceCheck:
    """Tests for the confidence and source filtering guardrails (Layers 2+3)."""

    def test_high_confidence_passes(self, guardrails):
        """Chunks with high scores should pass confidence check."""
        g, _ = guardrails

        nodes = [
            make_node_with_score("Self-attention allows...", "paper.pdf", 0.85),
            make_node_with_score("The transformer uses...", "paper.pdf", 0.72),
            make_node_with_score("BERT is a model...", "bert.pdf", 0.65),
        ]

        passed, filtered, message = g.check_confidence(nodes)

        assert passed is True
        assert message is None
        assert len(filtered) == 3  # all should pass source filtering too

    def test_low_confidence_rejected(self, guardrails):
        """Chunks with scores below threshold should trigger rejection."""
        g, _ = guardrails

        nodes = [
            make_node_with_score("Unrelated text...", "doc.pdf", 0.2),
            make_node_with_score("More unrelated...", "doc.pdf", 0.15),
        ]

        passed, filtered, message = g.check_confidence(nodes)

        assert passed is False
        assert message is not None
        assert "don't have enough" in message.lower() or "confident" in message.lower()

    def test_source_filtering_removes_low_scores(self, guardrails):
        """Chunks below SOURCE_MIN_SCORE should be filtered out."""
        g, _ = guardrails

        nodes = [
            make_node_with_score("Good chunk", "paper.pdf", 0.75),
            make_node_with_score("Borderline chunk", "paper.pdf", 0.45),
            make_node_with_score("Bad chunk", "other.pdf", 0.1),
        ]

        passed, filtered, message = g.check_confidence(nodes)

        assert passed is True
        # The bad chunk (0.1) should be filtered out
        assert len(filtered) < len(nodes)
        # All remaining chunks should meet the minimum score
        from src.rag.guardrails import SOURCE_MIN_SCORE

        for node in filtered:
            assert node.score >= SOURCE_MIN_SCORE

    def test_empty_nodes_rejected(self, guardrails):
        """Empty node list should fail confidence check."""
        g, _ = guardrails

        passed, filtered, message = g.check_confidence([])

        assert passed is False
        assert len(filtered) == 0
        assert message is not None

    def test_nodes_without_scores_pass_through(self, guardrails):
        """Nodes without scores should pass (let the LLM decide)."""
        g, _ = guardrails

        node = TextNode(text="Some text", metadata={"file_name": "doc.pdf"})
        nodes = [NodeWithScore(node=node, score=None)]

        passed, filtered, message = g.check_confidence(nodes)

        assert passed is True
        assert len(filtered) == 1


# ── Cosine Similarity Tests ──────────────────────────


class TestCosineSimilarity:
    """Tests for the cosine similarity helper method."""

    def test_identical_vectors_score_one(self, guardrails):
        """Cosine similarity of a vector with itself should be ~1.0."""
        g, _ = guardrails

        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[1.0, 0.0, 0.0]])

        sims = g._cosine_similarity(vec, matrix)
        assert abs(sims[0] - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self, guardrails):
        """Perpendicular vectors should have ~0.0 similarity."""
        g, _ = guardrails

        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[0.0, 1.0, 0.0]])

        sims = g._cosine_similarity(vec, matrix)
        assert abs(sims[0]) < 1e-6

    def test_opposite_vectors_score_negative(self, guardrails):
        """Opposite vectors should have similarity of -1.0."""
        g, _ = guardrails

        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[-1.0, 0.0, 0.0]])

        sims = g._cosine_similarity(vec, matrix)
        assert abs(sims[0] - (-1.0)) < 1e-6

    def test_multiple_vectors(self, guardrails):
        """Should return one score per row in the matrix."""
        g, _ = guardrails

        vec = np.array([1.0, 0.0, 0.0])
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )

        sims = g._cosine_similarity(vec, matrix)
        assert len(sims) == 3
