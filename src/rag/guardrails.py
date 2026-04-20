"""
Guardrails — Phase 5: Scope, Confidence & Source Quality
=========================================================
Three layers of protection to make the chatbot reliable:

Layer 1 — SCOPE CHECK (before retrieval):
  "Is this question even about AI/ML?"
  Uses the embedding model to compare the question against reference
  AI/ML phrases. If the question is too far from any reference,
  reject it immediately — no retrieval, no LLM call, zero cost.

  How it works:
    We pre-compute embeddings for ~20 reference phrases like
    "machine learning algorithms", "neural network training", etc.
    When a question comes in, we embed it and compute cosine similarity
    against all reference embeddings. If the best match is below
    SCOPE_THRESHOLD, the question is out of scope.

  Why embeddings and not keywords?
    Keywords are brittle — "How does backprop work?" doesn't contain
    the word "machine learning" but is clearly in scope. Embeddings
    capture semantic meaning, so "backprop" is close to "neural network
    training" in vector space even though they share no words.

Layer 2 — CONFIDENCE CHECK (after retrieval, before LLM):
  "Are the retrieved chunks actually relevant to this question?"
  ChromaDB always returns top_k results even if nothing matches well.
  If the highest similarity score is below CONFIDENCE_THRESHOLD,
  return a fallback response without calling the LLM.

  This saves API credits and prevents the LLM from generating
  answers based on irrelevant context (which leads to hallucination).

Layer 3 — SOURCE FILTERING (after retrieval, before LLM):
  "Which of the retrieved chunks are good enough to use?"
  Even if the top chunk passes the confidence check, some of the
  lower-ranked chunks might be noise. We filter out any chunk
  below SOURCE_MIN_SCORE so the LLM only sees high-quality context.

  This is why your sources sometimes showed garbled PDF figure text —
  those chunks had low similarity but still got passed to the LLM.

Tuning guide:
  If too many valid questions get rejected → lower SCOPE_THRESHOLD
  If off-topic questions slip through     → raise SCOPE_THRESHOLD
  If answers say "I don't know" too often → lower CONFIDENCE_THRESHOLD
  If answers hallucinate                  → raise CONFIDENCE_THRESHOLD
"""

from typing import List, Optional, Tuple

import numpy as np
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
from loguru import logger

from src.indexing.embeddings import get_embedding_model

# ── Thresholds (tune these based on testing) ─────────
SCOPE_THRESHOLD = 0.3  # minimum similarity to any reference phrase
CONFIDENCE_THRESHOLD = 0.55  # minimum similarity of the BEST retrieved chunk
SOURCE_MIN_SCORE = 0.35  # minimum similarity to include a chunk as context


# ── Reference phrases for scope checking ─────────────
# These represent the "boundary" of what the chatbot should answer.
# The embedding model will place questions close to these phrases
# if they're about similar topics.
#
# Tips for tuning:
#   - Add phrases for topics you WANT covered
#   - Keep phrases short and specific (embeddings work better this way)
#   - Cover different phrasings of similar topics
SCOPE_REFERENCE_PHRASES = [
    # Core ML
    "machine learning algorithms",
    "supervised learning classification regression",
    "unsupervised learning clustering",
    "deep learning neural networks",
    "model training and optimization",
    "overfitting regularization",
    "gradient descent backpropagation",
    "loss function objective function",
    # NLP / Transformers (your current papers)
    "natural language processing",
    "transformer architecture attention mechanism",
    "BERT language model pre-training",
    "self-attention multi-head attention",
    "word embeddings word2vec",
    "tokenization text processing",
    "encoder decoder sequence to sequence",
    # Data Analytics
    "data analytics statistical analysis",
    "feature engineering data preprocessing",
    "exploratory data analysis visualization",
    "hypothesis testing p-value",
    "A/B testing experimental design",
    # ML Operations
    "model evaluation metrics accuracy precision recall",
    "cross validation train test split",
    "hyperparameter tuning grid search",
    "bias variance tradeoff",
    "confusion matrix ROC curve",
    # General AI
    "artificial intelligence",
    "reinforcement learning reward policy",
    "computer vision image classification",
    "generative AI large language models",
]


class Guardrails:
    """
    Manages all guardrail checks for the RAG pipeline.

    Usage:
        guardrails = Guardrails()

        # Check scope before retrieval
        in_scope, message = guardrails.check_scope(question)
        if not in_scope:
            return message  # "I'm designed to help with AI/ML..."

        # After retrieval, check confidence and filter sources
        passed, filtered_nodes, message = guardrails.check_confidence(source_nodes)
        if not passed:
            return message  # "I don't have enough information..."
        # Use filtered_nodes (only high-quality chunks) for the LLM call
    """

    def __init__(self):
        """
        Initialise guardrails by pre-computing reference phrase embeddings.

        This runs once at startup. The embedding model is already cached
        from ingestion, so this is fast (~1 second for 30 phrases).
        """
        logger.info("Initialising guardrails...")

        self._embed_model = get_embedding_model()

        # Pre-compute reference embeddings (done once, reused for every query)
        logger.info(
            f"Computing scope reference embeddings "
            f"({len(SCOPE_REFERENCE_PHRASES)} phrases)..."
        )
        self._reference_embeddings = self._embed_model.get_text_embedding_batch(
            SCOPE_REFERENCE_PHRASES
        )
        # Convert to numpy array for fast cosine similarity computation
        self._reference_matrix = np.array(self._reference_embeddings)

        logger.info("Guardrails ready ✅")

    def check_scope(self, question: str) -> Tuple[bool, Optional[str]]:
        """
        Layer 1: Is this question about AI/ML/Data Analytics?

        Embeds the question and compares it against reference phrases.
        Returns (True, None) if in scope, (False, message) if out of scope.

        Args:
            question: the user's question

        Returns:
            (in_scope, message):
                in_scope=True,  message=None         → proceed with retrieval
                in_scope=False, message="Sorry..."   → return message to user
        """
        # Embed the question
        question_embedding = np.array(self._embed_model.get_query_embedding(question))

        # Compute cosine similarity against all reference phrases
        # Cosine similarity = dot product of normalised vectors
        #   → 1.0 = identical meaning
        #   → 0.0 = completely unrelated
        similarities = self._cosine_similarity(
            question_embedding, self._reference_matrix
        )
        max_similarity = float(np.max(similarities))
        best_match_idx = int(np.argmax(similarities))

        logger.debug(
            f"Scope check: max similarity = {max_similarity:.4f} "
            f"(best match: '{SCOPE_REFERENCE_PHRASES[best_match_idx]}')"
        )

        if max_similarity < SCOPE_THRESHOLD:
            logger.info(
                f"Question out of scope (similarity={max_similarity:.4f} "
                f"< threshold={SCOPE_THRESHOLD})"
            )
            return False, (
                "I'm designed to help with AI/ML and Data Analytics interview "
                "preparation. Your question seems to be outside that area. "
                "Could you ask something about machine learning, deep learning, "
                "NLP, or data analytics?"
            )

        logger.debug(f"Question in scope (similarity={max_similarity:.4f})")
        return True, None

    def check_confidence(
        self, source_nodes: List[NodeWithScore]
    ) -> Tuple[bool, List[NodeWithScore], Optional[str]]:
        """
        Layer 2 + 3: Are the retrieved chunks relevant enough?

        Checks the best chunk's score against CONFIDENCE_THRESHOLD,
        then filters out any chunks below SOURCE_MIN_SCORE.

        Args:
            source_nodes: list of NodeWithScore from the retriever

        Returns:
            (passed, filtered_nodes, message):
                passed=True  → filtered_nodes has the good chunks, message=None
                passed=False → filtered_nodes is empty, message="I don't have..."
        """
        if not source_nodes:
            return (
                False,
                [],
                (
                    "I couldn't find any relevant information in my sources. "
                    "Try rephrasing your question or asking about a topic "
                    "covered in the source documents."
                ),
            )

        # Get the best similarity score
        scores = [node.score for node in source_nodes if node.score is not None]
        if not scores:
            # No scores available — let the LLM decide
            return True, source_nodes, None

        best_score = max(scores)
        logger.debug(f"Confidence check: best score = {best_score:.4f}")

        # Layer 2: Is the best chunk relevant enough?
        if best_score < CONFIDENCE_THRESHOLD:
            logger.info(
                f"Low confidence (best={best_score:.4f} "
                f"< threshold={CONFIDENCE_THRESHOLD})"
            )
            return (
                False,
                [],
                (
                    "I don't have enough relevant information in my sources to "
                    "answer this confidently. The documents I have don't seem to "
                    "cover this specific topic in enough detail. Try asking about "
                    "concepts from the Attention/Transformer or BERT papers."
                ),
            )

        # Layer 3: Filter out low-quality chunks
        filtered = []
        removed = 0
        for node in source_nodes:
            if node.score is not None and node.score >= SOURCE_MIN_SCORE:
                filtered.append(node)
            else:
                removed += 1

        if removed > 0:
            logger.info(
                f"Source filtering: kept {len(filtered)}/{len(source_nodes)} "
                f"chunks (removed {removed} below {SOURCE_MIN_SCORE})"
            )

        # Edge case: all chunks were filtered out
        if not filtered:
            return (
                False,
                [],
                (
                    "I found some potentially related information, but none of it "
                    "was relevant enough to give you a reliable answer. "
                    "Could you rephrase your question?"
                ),
            )

        return True, filtered, None

    @staticmethod
    def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a single vector and a matrix of vectors.

        Cosine similarity measures the angle between two vectors:
          - 1.0 = vectors point in the same direction (identical meaning)
          - 0.0 = vectors are perpendicular (unrelated)
          - -1.0 = vectors point in opposite directions (opposite meaning)

        For normalised embeddings (which sentence-transformers produces),
        cosine similarity is just the dot product.

        Args:
            vector: shape (384,) — the question embedding
            matrix: shape (N, 384) — the reference phrase embeddings

        Returns:
            Array of shape (N,) with similarity scores
        """
        # Normalise the vector
        vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
        # Normalise each row of the matrix
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        # Dot product of normalised vectors = cosine similarity
        return matrix_norms @ vector_norm


# ── System Prompt ────────────────────────────────────
# Updated from Phase 4 with tighter instructions.
#
# Key changes:
#   - Stronger instruction to ONLY use provided context
#   - Explicit instruction to say which document info came from
#   - Instruction to structure answers in an interview-friendly way

SYSTEM_PROMPT = PromptTemplate(
    """\
You are an AI/ML Interview Preparation Assistant. Your job is to help users \
prepare for interviews on AI, Machine Learning, and Data Analytics topics.

STRICT RULES:
1. ONLY answer based on the context provided below. Do NOT use any outside \
knowledge, even if you know the answer.
2. If the context does not contain enough information to fully answer the \
question, say: "Based on my sources, I can share the following..." and \
provide only what the context supports. Do NOT fill gaps with your own knowledge.
3. Reference which source document the information comes from \
(e.g., "According to the Attention paper..." or "As described in BERT...").
4. Structure answers clearly for interview preparation:
   - Start with a concise definition or summary
   - Follow with key details from the context
   - End with why it matters (if relevant to interviews)
5. Keep answers focused and avoid repeating information.

CONTEXT FROM SOURCE DOCUMENTS:
------------------------------
{context_str}
------------------------------

USER QUESTION: {query_str}

ANSWER:
"""
)
