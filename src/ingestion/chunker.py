"""
Document Chunker — Phase 2: Data Ingestion
===========================================
Splits loaded documents into smaller chunks (called "nodes" in LlamaIndex).

Why chunking matters:
  LLMs have a context window limit — you can't feed in an entire book.
  Chunking splits documents into smaller pieces so:
    1. Each chunk fits in the LLM context window
    2. Retrieval is more precise (fetch the relevant paragraph, not the whole doc)
    3. Embeddings capture focused meaning (not diluted across a long doc)

Key hyperparameters:
  chunk_size:    how many tokens per chunk (512 is a good starting point)
  chunk_overlap: tokens shared between adjacent chunks (prevents losing context
                 at chunk boundaries — like a sliding window)

Example with chunk_size=10, chunk_overlap=3:
  Original: [A B C D E F G H I J K L M]
  Chunk 1:  [A B C D E F G H I J]
  Chunk 2:  [H I J K L M ...]       ← H,I,J are repeated (overlap)
"""

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from loguru import logger


class DocumentChunker:
    """
    Splits documents into nodes using LlamaIndex's SentenceSplitter.

    SentenceSplitter is smarter than a fixed-size splitter:
    it tries to split on sentence boundaries, so chunks don't
    cut off mid-sentence.

    Usage:
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        nodes = chunker.chunk_documents(documents)
        print(f"Created {len(nodes)} chunks")
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # SentenceSplitter is the node parser we'll use.
        # It measures size in tokens (not characters).
        # 512 tokens ≈ ~380 words ≈ ~1.5 paragraphs.
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Include the metadata (file name etc.) in each node
            include_metadata=True,
        )

        logger.info(
            f"DocumentChunker initialised: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Split a list of documents into smaller nodes.

        Args:
            documents: list of LlamaIndex Document objects (from DocumentLoader)

        Returns:
            List of TextNode objects — each node is one chunk, ready to be embedded.
            Each node inherits metadata from its parent document (file_name, etc.)

        How it works internally:
            SentenceSplitter tokenises the text, then slides a window of
            chunk_size tokens across it, creating a new node at each step.
            It respects sentence boundaries so chunks end cleanly.
        """
        if not documents:
            raise ValueError("No documents provided to chunk.")

        logger.info(f"Chunking {len(documents)} documents...")

        nodes = self.parser.get_nodes_from_documents(
            documents,
            show_progress=True,  # shows a progress bar in terminal
        )

        logger.info(f"Created {len(nodes)} chunks from {len(documents)} documents")

        # Log a sample so you can see what a chunk looks like
        if nodes:
            sample = nodes[0]
            logger.debug(f"Sample chunk (first 200 chars): {sample.text[:200]!r}")
            logger.debug(f"Sample metadata: {sample.metadata}")

        return nodes

    def get_stats(self, nodes: List[TextNode]) -> dict:
        """
        Return stats about your chunks — useful for tuning chunk_size.

        If avg_chunk_length is much lower than chunk_size, your docs
        may be short and you could reduce chunk_size.
        If chunks feel too broad when querying, reduce chunk_size.
        If answers lose context, increase chunk_overlap.
        """
        if not nodes:
            return {"total_nodes": 0}

        lengths = [len(node.text) for node in nodes]
        total_chars = sum(lengths)
        avg_chars = total_chars // len(nodes)
        min_chars = min(lengths)
        max_chars = max(lengths)

        # Count unique source files
        source_files = list(
            {node.metadata.get("file_name", "unknown") for node in nodes}
        )

        stats = {
            "total_nodes": len(nodes),
            "total_characters": total_chars,
            "avg_chunk_length": avg_chars,
            "min_chunk_length": min_chars,
            "max_chunk_length": max_chars,
            "source_files": source_files,
            "chunk_size_tokens": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        logger.info("Chunking stats:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats
