"""
Vector Store (ChromaDB) — Phase 3: Vector Store + Embeddings
=============================================================
Manages the ChromaDB vector database where embeddings are stored and queried.

What is a vector store?
  A database optimised for storing and searching vectors (embeddings).
  When a user asks a question:
    1. The question is embedded into a vector
    2. ChromaDB finds the stored chunks whose vectors are most similar
    3. Those chunks become the "context" the LLM uses to answer

Why ChromaDB?
  - Free, open-source
  - Runs locally (no cloud account needed)
  - Persistent storage (survives restarts)
  - Simple API — great for learning

How persistence works:
  ChromaDB stores data on disk at the path you give it (./data/chroma_db).
  First run:  creates the database and inserts all embeddings (~10–30 seconds)
  Later runs: loads from disk instantly (no re-embedding needed)
  If you change your documents, delete the chroma_db folder and re-run ingestion.
"""

from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from src.indexing.embeddings import EMBEDDING_DIMENSION, get_embedding_model

# ── Defaults ─────────────────────────────────────────
DEFAULT_PERSIST_DIR = "./data/chroma_db"
DEFAULT_COLLECTION = "rag_chatbot"


def get_chroma_vector_store(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> ChromaVectorStore:
    """
    Create or load a ChromaDB vector store.

    Args:
        persist_dir:     where ChromaDB stores its files on disk
        collection_name: name of the collection (like a table in SQL)

    Returns:
        ChromaVectorStore instance wired to a persistent ChromaDB collection
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initialising ChromaDB at: {persist_path.resolve()}")

    # PersistentClient stores data on disk so it survives restarts
    chroma_client = chromadb.PersistentClient(path=str(persist_path))

    # get_or_create_collection:
    #   - first run: creates a new empty collection
    #   - subsequent runs: opens the existing one
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    logger.info(
        f"Collection '{collection_name}' ready "
        f"(existing docs: {chroma_collection.count()})"
    )

    # Wrap in LlamaIndex's ChromaVectorStore adapter
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def build_index_from_nodes(
    nodes: List[TextNode],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStoreIndex:
    """
    Embed chunks and store them in ChromaDB. Returns a queryable index.

    This is the main function called by the ingestion script.
    It wires together: embedding model + ChromaDB + LlamaIndex index.

    Args:
        nodes:           list of TextNode chunks (from DocumentChunker)
        persist_dir:     ChromaDB storage path
        collection_name: collection name

    Returns:
        VectorStoreIndex — use index.as_query_engine() in Phase 4

    What happens internally:
        1. Each node's text is passed through the embedding model
        2. The resulting vector + text + metadata is stored in ChromaDB
        3. A VectorStoreIndex is created that knows how to query ChromaDB
    """
    if not nodes:
        raise ValueError("No nodes provided to index.")

    logger.info(f"Building index from {len(nodes)} nodes...")

    # Get our components
    embed_model = get_embedding_model()
    vector_store = get_chroma_vector_store(persist_dir, collection_name)

    # StorageContext tells LlamaIndex where to store everything
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This is where the magic happens:
    #   - each node is embedded (text → 384-dim vector)
    #   - vector + text + metadata is inserted into ChromaDB
    #   - progress bar shows embedding progress
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info("Index built and persisted to ChromaDB ✅")
    return index


def load_existing_index(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Optional[VectorStoreIndex]:
    """
    Load a previously built index from ChromaDB (no re-embedding).

    Use this when you've already run ingestion and just want to query.
    Returns None if the collection is empty (need to run ingestion first).

    This is what the API server will call at startup in Phase 6.
    """
    vector_store = get_chroma_vector_store(persist_dir, collection_name)

    # Check if there's actually data in the collection
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_collection(collection_name)

    if collection.count() == 0:
        logger.warning(
            "ChromaDB collection is empty. Run ingestion first: "
            "python scripts/ingest_data.py"
        )
        return None

    logger.info(f"Loading existing index ({collection.count()} vectors)")

    embed_model = get_embedding_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    logger.info("Index loaded from ChromaDB ✅")
    return index
