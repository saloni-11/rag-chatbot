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
  If the highest similarity score is below CONFIDENCE_MIN_COSINE (true cosine),
  return a fallback response without calling the LLM.

  This saves API credits and prevents the LLM from generating
  answers based on irrelevant context (which leads to hallucination).

Layer 3 — SOURCE FILTERING (after retrieval, before LLM):
  "Which of the retrieved chunks are good enough to use?"
  Even if the top chunk passes the confidence check, some of the
  lower-ranked chunks might be noise. We filter out any chunk
  below SOURCE_MIN_COSINE (true cosine) so the LLM only sees high-quality context.

  This is why your sources sometimes showed garbled PDF figure text —
  those chunks had low similarity but still got passed to the LLM.

Tuning guide:
  If too many valid questions get rejected → lower SCOPE_THRESHOLD
  If off-topic questions slip through     → raise SCOPE_THRESHOLD
  If answers say "I don't know" too often → lower CONFIDENCE_MIN_COSINE
  If answers hallucinate                  → raise CONFIDENCE_MIN_COSINE
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
from loguru import logger

from src.indexing.embeddings import get_embedding_model

# ── Thresholds (tuned with the evaluation sweep) ─────
# `python src/evaluation/ragas_eval.py --stage retrieval --sweep` grid-searches
# these against tests/eval_dataset.json. Scope is kept loose because on-topic
# and off-topic questions overlap around 0.2–0.25 similarity; the confidence
# check separates them far more cleanly (off-topic / unanswerable questions
# retrieve at <= 0.50, answerable ones at >= 0.53).
#
# All three thresholds are TRUE COSINE SIMILARITY (-1..1), so they can be
# compared with each other directly.
#
# The scope check computes cosine itself. The confidence check and source
# filter read the score ChromaDB attaches to each retrieved chunk, which is
# NOT cosine: llama-index's Chroma adapter reports exp(-cosine_distance),
# i.e. exp(cosine - 1). See chroma_score_to_cosine() below.
#
# The confidence/source variables were renamed from CONFIDENCE_THRESHOLD and
# SOURCE_MIN_SCORE (which held exp-scale values) so a value written for the
# old scale can never be silently read as a cosine. Old -> new equivalents:
#   CONFIDENCE_THRESHOLD=0.5  ->  CONFIDENCE_MIN_COSINE=0.307  (1 + ln 0.5)
#   SOURCE_MIN_SCORE=0.35     ->  SOURCE_MIN_COSINE=-0.05      (1 + ln 0.35)
SCOPE_THRESHOLD = float(os.getenv("SCOPE_THRESHOLD", "0.2"))
CONFIDENCE_MIN_COSINE = float(os.getenv("CONFIDENCE_MIN_COSINE", "0.307"))
SOURCE_MIN_COSINE = float(os.getenv("SOURCE_MIN_COSINE", "-0.05"))

for _old_name in ("CONFIDENCE_THRESHOLD", "SOURCE_MIN_SCORE"):
    if os.getenv(_old_name) is not None:
        logger.warning(
            f"{_old_name} is set but no longer used: it was on ChromaDB's "
            "exp(cosine - 1) scale. Use CONFIDENCE_MIN_COSINE / "
            "SOURCE_MIN_COSINE (true cosine) instead."
        )


def chroma_score_to_cosine(score: float) -> float:
    """
    Convert a ChromaDB retrieval score back to true cosine similarity.

    The collection uses cosine distance (d = 1 - cosine), and
    llama_index.vector_stores.chroma reports similarity as exp(-d):

        score  = exp(-(1 - cosine)) = exp(cosine - 1)
        cosine = ln(score) + 1

    The transform is monotonic, so it never changes the ranking of chunks;
    it only puts the score on the same scale as the scope check. Scores lie
    in [e^-2, 1] ~= [0.135, 1], which maps back to cosine [-1, 1].
    """
    if score <= 0:
        return -1.0
    return math.log(score) + 1.0


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
    # RAG & Information Retrieval
    "retrieval augmented generation RAG",
    "vector database embeddings retrieval",
    "document chunking text splitting",
    # Topics from your papers
    "generative adversarial networks GANs",
    "residual connections skip connections deep networks",
    "few-shot learning in-context learning",
    "language model pre-training fine-tuning",
    "data-centric AI benchmarks",
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

    def scope_score(self, question: str) -> Tuple[float, str]:
        """
        Highest cosine similarity between the question and any reference
        phrase, plus the phrase that matched. check_scope() compares this
        against SCOPE_THRESHOLD; the evaluation uses it to tune the threshold.
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
        best_match_idx = int(np.argmax(similarities))
        return (
            float(similarities[best_match_idx]),
            SCOPE_REFERENCE_PHRASES[best_match_idx],
        )

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
        max_similarity, best_match = self.scope_score(question)

        logger.debug(
            f"Scope check: max similarity = {max_similarity:.4f} "
            f"(best match: '{best_match}')"
        )

        if max_similarity < SCOPE_THRESHOLD:
            logger.info(
                f"Question out of scope (similarity={max_similarity:.4f} "
                f"< threshold={SCOPE_THRESHOLD})"
            )
            return False, (
                "I'm designed to help with AI/ML and Data Analytics learning."
                "Your question seems to be outside that area. "
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

        Converts each chunk's ChromaDB score to true cosine, checks the best
        one against CONFIDENCE_MIN_COSINE, then filters out any chunks below
        SOURCE_MIN_COSINE.

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

        # Get the best similarity, as true cosine (see chroma_score_to_cosine)
        cosines = [
            chroma_score_to_cosine(node.score)
            for node in source_nodes
            if node.score is not None
        ]
        if not cosines:
            # No scores available — let the LLM decide
            return True, source_nodes, None

        best_cosine = max(cosines)
        logger.debug(f"Confidence check: best cosine = {best_cosine:.4f}")

        # Layer 2: Is the best chunk relevant enough?
        if best_cosine < CONFIDENCE_MIN_COSINE:
            logger.info(
                f"Low confidence (best cosine={best_cosine:.4f} "
                f"< threshold={CONFIDENCE_MIN_COSINE})"
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
            if (
                node.score is not None
                and chroma_score_to_cosine(node.score) >= SOURCE_MIN_COSINE
            ):
                filtered.append(node)
            else:
                removed += 1

        if removed > 0:
            logger.info(
                f"Source filtering: kept {len(filtered)}/{len(source_nodes)} "
                f"chunks (removed {removed} below cosine {SOURCE_MIN_COSINE})"
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
#
# A closing "why it matters" section was removed after evaluation showed it
# invited claims the sources didn't support (lowering RAGAS faithfulness).

SYSTEM_PROMPT = PromptTemplate(
    """\
You are an AI/ML Study Companion. Your job is to help users \
learn about AI, Machine Learning, and Data Analytics topics.

STRICT RULES:
1. ONLY answer based on the context provided below. Do NOT use any outside \
knowledge, even if you know the answer.
2. If the context does not contain enough information to fully answer the \
question, say: "Based on my sources, I can share the following..." and \
provide only what the context supports. Do NOT fill gaps with your own knowledge.
3. Reference which source document the information comes from \
(e.g., "According to the Attention paper..." or "As described in BERT...").
4. Structure answers clearly for learning:
   - Start with a concise definition or summary
   - Follow with key details from the context
5. Keep answers focused and avoid repeating information.

CONTEXT FROM SOURCE DOCUMENTS:
------------------------------
{context_str}
------------------------------

USER QUESTION: {query_str}

ANSWER:
"""
)
