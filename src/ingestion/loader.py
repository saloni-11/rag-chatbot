"""
Document Loader — Phase 2: Data Ingestion
==========================================
Loads documents from local files (PDF, Markdown, text) into
LlamaIndex Document objects, ready for chunking.

What is a LlamaIndex Document?
  A Document is just a wrapper around a piece of text + metadata.
  Metadata includes things like the file name, file path, creation date.
  LlamaIndex uses this metadata later to tell you WHICH document
  an answer came from — that's how the sources panel in the UI works.
"""

from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader
from loguru import logger


class DocumentLoader:
    """
    Loads documents from multiple sources into LlamaIndex Document objects.

    Usage:
        loader = DocumentLoader(data_dir="./data/raw")
        documents = loader.load_from_directory(extensions=[".pdf", ".md", ".txt"])
        print(f"Loaded {len(documents)} documents")
    """

    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)

    def load_from_directory(
        self,
        extensions: List[str] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load all documents from the raw data directory.

        Args:
            extensions: file extensions to include.
                        Defaults to [".pdf", ".md", ".txt"]
            recursive:  if True, also loads from subdirectories

        Returns:
            List of LlamaIndex Document objects.

        How SimpleDirectoryReader works:
            It scans the directory, reads each file, and wraps the content
            in a Document object with metadata (filename, path, etc).
            For PDFs it extracts text page by page.
            For markdown/text it reads as-is.
        """
        if extensions is None:
            extensions = [".pdf", ".md", ".txt"]

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Create it and add some documents first."
            )

        # Check if there are any matching files
        matching_files = []
        for ext in extensions:
            matching_files.extend(
                self.data_dir.rglob(f"*{ext}")
                if recursive
                else self.data_dir.glob(f"*{ext}")
            )

        if not matching_files:
            raise ValueError(
                f"No files with extensions {extensions} found in {self.data_dir}.\n"
                f"Add some .pdf or .md files to data/raw/ first."
            )

        logger.info(f"Found {len(matching_files)} files to load from {self.data_dir}")

        # SimpleDirectoryReader does the heavy lifting:
        # - reads each file
        # - extracts text (handles PDFs via pypdf)
        # - attaches metadata (file_name, file_path, etc.)
        reader = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            required_exts=extensions,
            recursive=recursive,
            # This adds useful metadata to each document automatically
            filename_as_id=True,
        )

        documents = reader.load_data()

        logger.info(f"Successfully loaded {len(documents)} document pages/sections")
        for doc in documents[:3]:  # preview first 3
            fname = doc.metadata.get("file_name", "unknown")
            logger.debug(f"  → {fname} ({len(doc.text)} chars)")

        return documents

    def load_single_file(self, file_path: str) -> List[Document]:
        """
        Load a single specific file.

        Args:
            file_path: path to a .pdf, .md, or .txt file

        Returns:
            List of Document objects (PDFs may return multiple — one per page)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading single file: {path.name}")

        reader = SimpleDirectoryReader(input_files=[str(path)])
        documents = reader.load_data()

        logger.info(f"Loaded {len(documents)} document(s) from {path.name}")
        return documents

    def get_stats(self, documents: List[Document]) -> dict:
        """
        Return a summary of loaded documents — useful for debugging.

        Args:
            documents: list of loaded Document objects

        Returns:
            dict with counts and character statistics
        """
        if not documents:
            return {"total_documents": 0}

        total_chars = sum(len(doc.text) for doc in documents)
        avg_chars = total_chars // len(documents)
        file_names = list(
            {doc.metadata.get("file_name", "unknown") for doc in documents}
        )

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_doc": avg_chars,
            "source_files": file_names,
        }
