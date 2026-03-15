"""
scripts/ingest_data.py
======================
Run this script once to load, chunk, and index all your source documents.
Re-run it whenever you add new documents to data/raw/.

Usage (from the project root):
    python scripts/ingest_data.py

What this script does:
    1. Loads documents from data/raw/ (PDFs, markdown, text files)
    2. Chunks them into smaller pieces using SentenceSplitter
    3. [Phase 3] Embeds chunks using sentence-transformers
    4. [Phase 3] Stores embeddings in ChromaDB

For now (Phase 2), Steps 1 and 2 are implemented.
Steps 3 and 4 are stubbed out until Phase 3.
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
    logger.info("RAG Chatbot — Data Ingestion Pipeline")
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

    # ── Step 3: Embed + index (Phase 3) ───────
    logger.info("")
    logger.info("STEP 3: Embedding + indexing (Phase 3 — not yet implemented)")
    logger.info("  → Come back here after completing Phase 3")

    # ── Step 4: Summary ───────────────────────
    logger.info("")
    logger.info("=" * 50)
    logger.info("Phase 2 Complete! ✅")
    logger.info(f"  Documents loaded: {stats['total_documents']}")
    logger.info(f"  Chunks created:   {chunk_stats['total_nodes']}")
    logger.info("Next step: implement Phase 3 (vector store + embeddings)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()