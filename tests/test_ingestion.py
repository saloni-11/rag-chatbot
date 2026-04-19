"""
tests/test_ingestion.py
========================
Unit tests for the data ingestion pipeline (loader + chunker).

TESTING CONCEPTS:
=================

What is a unit test?
  A test that checks ONE small piece of code in isolation. Each test
  function (prefixed with test_) tests one specific behaviour.
  If loader.load_from_directory() should raise FileNotFoundError when
  the directory doesn't exist, we write a test that verifies exactly that.

What is pytest?
  Python's most popular testing framework. It discovers functions that
  start with 'test_', runs them, and reports pass/fail. No boilerplate
  classes needed — just write functions.

What are fixtures (@pytest.fixture)?
  Reusable setup code. Instead of repeating "create a temp directory
  with sample files" in every test, you define it once as a fixture
  and pytest injects it into any test that needs it.

What is tmp_path?
  A built-in pytest fixture that gives you a temporary directory for
  each test. It's automatically cleaned up after the test finishes.
  This prevents tests from polluting each other's files.

Why test edge cases?
  Happy-path tests ("it works when given good input") aren't enough.
  You need to test: what if the directory is empty? What if documents
  list is empty? What if chunk_size is tiny? These edge cases are
  where real bugs hide, and interviewers love asking about them.
"""

import pytest
from pathlib import Path

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_data_dir(tmp_path):
    """
    Creates a temporary directory with sample text and markdown files.

    tmp_path is a built-in pytest fixture — it gives us a fresh temp
    directory for each test. We create sample files inside it.
    """
    # Create a sample text file
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "Machine learning is a subset of artificial intelligence. "
        "It allows systems to learn from data without being explicitly programmed. "
        "Supervised learning uses labeled data. Unsupervised learning finds patterns "
        "in unlabeled data. Reinforcement learning learns through trial and error."
    )

    # Create a sample markdown file
    md_file = tmp_path / "notes.md"
    md_file.write_text(
        "# Transformers\n\n"
        "The transformer architecture was introduced in the paper "
        "'Attention Is All You Need'. It relies entirely on self-attention "
        "mechanisms, dispensing with recurrence and convolutions. "
        "The key innovation is the multi-head attention mechanism.\n\n"
        "## Self-Attention\n\n"
        "Self-attention allows each position in the sequence to attend to "
        "all positions in the previous layer. This captures long-range "
        "dependencies more effectively than RNNs."
    )

    return tmp_path


@pytest.fixture
def empty_data_dir(tmp_path):
    """A directory with no matching files."""
    # Create a file with an unsupported extension
    (tmp_path / "data.csv").write_text("col1,col2\n1,2")
    return tmp_path


# ── DocumentLoader Tests ─────────────────────────────

class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def test_load_from_directory_success(self, sample_data_dir):
        """Loader should successfully load .txt and .md files."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        assert len(documents) > 0
        # Check that metadata is attached
        for doc in documents:
            assert "file_name" in doc.metadata
            assert len(doc.text) > 0

    def test_load_returns_correct_file_count(self, sample_data_dir):
        """Should load exactly the number of files we created."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        file_names = {doc.metadata.get("file_name") for doc in documents}
        assert "sample.txt" in file_names
        assert "notes.md" in file_names

    def test_load_nonexistent_directory_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        loader = DocumentLoader(data_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory()

    def test_load_empty_directory_raises(self, empty_data_dir):
        """Should raise ValueError when no matching files found."""
        loader = DocumentLoader(data_dir=str(empty_data_dir))
        with pytest.raises(ValueError, match="No files with extensions"):
            loader.load_from_directory(extensions=[".txt", ".md"])

    def test_load_single_file(self, sample_data_dir):
        """Should load a specific single file."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_single_file(
            str(sample_data_dir / "sample.txt")
        )

        assert len(documents) == 1
        assert "machine learning" in documents[0].text.lower()

    def test_load_single_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_single_file("nonexistent.pdf")

    def test_get_stats(self, sample_data_dir):
        """Stats should return correct counts."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])
        stats = loader.get_stats(documents)

        assert stats["total_documents"] == len(documents)
        assert stats["total_characters"] > 0
        assert stats["avg_characters_per_doc"] > 0
        assert len(stats["source_files"]) > 0

    def test_get_stats_empty(self):
        """Stats on empty list should return zero count."""
        loader = DocumentLoader()
        stats = loader.get_stats([])
        assert stats["total_documents"] == 0


# ── DocumentChunker Tests ────────────────────────────

class TestDocumentChunker:
    """Tests for the DocumentChunker class."""

    def test_chunk_documents_success(self, sample_data_dir):
        """Should produce chunks from loaded documents."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk_documents(documents)

        assert len(nodes) > 0
        # Should produce MORE chunks than documents when chunk_size is small
        assert len(nodes) >= len(documents)

    def test_chunks_have_text(self, sample_data_dir):
        """Every chunk should contain non-empty text."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk_documents(documents)

        for node in nodes:
            assert len(node.text.strip()) > 0

    def test_chunks_have_metadata(self, sample_data_dir):
        """Each chunk should inherit metadata from its parent document."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk_documents(documents)

        for node in nodes:
            assert "file_name" in node.metadata

    def test_chunk_empty_list_raises(self):
        """Should raise ValueError when given empty document list."""
        chunker = DocumentChunker()
        with pytest.raises(ValueError, match="No documents provided"):
            chunker.chunk_documents([])

    def test_chunk_size_affects_count(self, sample_data_dir):
        """Smaller chunk_size should produce more chunks."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        chunker_small = DocumentChunker(chunk_size=64, chunk_overlap=10)
        chunker_large = DocumentChunker(chunk_size=512, chunk_overlap=50)

        nodes_small = chunker_small.chunk_documents(documents)
        nodes_large = chunker_large.chunk_documents(documents)

        assert len(nodes_small) >= len(nodes_large)

    def test_get_stats(self, sample_data_dir):
        """Stats should return correct chunk metrics."""
        loader = DocumentLoader(data_dir=str(sample_data_dir))
        documents = loader.load_from_directory(extensions=[".txt", ".md"])

        chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
        nodes = chunker.chunk_documents(documents)
        stats = chunker.get_stats(nodes)

        assert stats["total_nodes"] == len(nodes)
        assert stats["total_characters"] > 0
        assert stats["min_chunk_length"] <= stats["avg_chunk_length"]
        assert stats["avg_chunk_length"] <= stats["max_chunk_length"]
        assert stats["chunk_size_tokens"] == 128
        assert stats["chunk_overlap"] == 20

    def test_get_stats_empty(self):
        """Stats on empty node list should return zero count."""
        chunker = DocumentChunker()
        stats = chunker.get_stats([])
        assert stats["total_nodes"] == 0
