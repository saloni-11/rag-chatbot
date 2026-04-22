"""
scripts/ingest_data.py
======================
Run this script once to load, chunk, embed, and index all your source documents.
Re-run it whenever you add new documents to data/raw/.

Usage (from the project root):
    python scripts/ingest_data.py

What this script does:
    1. Loads documents from data/raw/ (PDFs, markdown, text files)
    2. Chunks them into smaller pieces using SentenceSplitter
    3. Embeds chunks using sentence-transformers (all-MiniLM-L6-v2)
    4. Stores embeddings in ChromaDB for fast retrieval

Note:
    If you want to re-index from scratch (e.g., after changing chunk_size),
    delete the data/chroma_db/ folder first, then re-run this script.
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

# Load .env file (GROQ_API_KEY etc.)
load_dotenv()

# ── Configure loguru for readable output ──────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)


def main():
    logger.info("=" * 50)
    logger.info("AI/ML Study Companion — Data Ingestion Pipeline")
    logger.info("=" * 50)

    # ── Step 1: Load documents ─────────────────
    logger.info("STEP 1: Loading documents from data/raw/")

    from src.ingestion.loader import DocumentLoader

    loader = DocumentLoader(data_dir="./data/raw")
    documents = loader.load_from_directory(extensions=[".pdf", ".md", ".txt"])

    stats = loader.get_stats(documents)
    logger.info(f"Loaded {stats['total_documents']} document sections")
    logger.info(f"Total content: {stats['total_characters']:,} characters")
    logger.info(f"Source files: {stats['source_files']}")

    # ── Step 2: Chunk documents ────────────────
    logger.info("")
    logger.info("STEP 2: Chunking documents...")

    from src.ingestion.chunker import DocumentChunker

    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    nodes = chunker.chunk_documents(documents)
    chunk_stats = chunker.get_stats(nodes)

    logger.info(f"Created {chunk_stats['total_nodes']} chunks")
    logger.info(f"Average chunk size: {chunk_stats['avg_chunk_length']} characters")

    # ── Step 3: Embed + store in ChromaDB ──────
    logger.info("")
    logger.info("STEP 3: Embedding chunks + storing in ChromaDB...")
    logger.info("  (first run downloads the embedding model ~80 MB)")

    from src.indexing.vector_store import build_index_from_nodes

    index = build_index_from_nodes(nodes)

    # ── Step 4: Verify ─────────────────────────
    logger.info("")
    logger.info("STEP 4: Verifying index...")

    from src.indexing.vector_store import load_existing_index

    verify_index = load_existing_index()
    if verify_index is not None:
        logger.info("Index verification passed ✅")
    else:
        logger.error("Index verification FAILED — ChromaDB appears empty")
        sys.exit(1)

    # ── Summary ────────────────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("Ingestion Complete! ✅")
    logger.info(f"  Documents loaded:  {stats['total_documents']}")
    logger.info(f"  Chunks created:    {chunk_stats['total_nodes']}")
    logger.info(f"  Vectors stored:    {chunk_stats['total_nodes']}")
    logger.info(f"  ChromaDB location: ./data/chroma_db/")
    logger.info("")
    logger.info("Next step: Phase 4 — build the RAG query pipeline")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()