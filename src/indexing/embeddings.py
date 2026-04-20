"""
Embedding Model Setup — Phase 3: Vector Store + Embeddings
===========================================================
Configures the embedding model that converts text chunks into vectors.

What are embeddings?
  An embedding is a list of numbers (a vector) that captures the *meaning*
  of a piece of text. Similar texts get similar vectors.

  Example (simplified to 3 dimensions):
    "What is backpropagation?"  → [0.82, 0.15, 0.91]
    "How does backprop work?"   → [0.80, 0.14, 0.89]  ← very similar!
    "Best pizza in Sydney"      → [0.12, 0.95, 0.03]  ← very different

  In reality, all-MiniLM-L6-v2 produces 384-dimensional vectors.

Why all-MiniLM-L6-v2?
  - Free, runs locally (no API key needed)
  - Small model (~80 MB) — won't strain your laptop
  - Good quality for its size — widely used in RAG systems
  - Fast inference on CPU
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger

# ── Model name (used everywhere so define once) ──────
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # this model outputs 384-dim vectors


def get_embedding_model(
    model_name: str = DEFAULT_EMBED_MODEL,
) -> HuggingFaceEmbedding:
    """
    Create and return the embedding model.

    The first call downloads the model (~80 MB) and caches it locally.
    Subsequent calls load from cache instantly.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFaceEmbedding instance ready for LlamaIndex
    """
    logger.info(f"Loading embedding model: {model_name}")

    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        # Use CPU — we don't need GPU for a small model like MiniLM
        device="cpu",
    )

    logger.info(f"Embedding model loaded (dimension: {EMBEDDING_DIMENSION})")
    return embed_model
