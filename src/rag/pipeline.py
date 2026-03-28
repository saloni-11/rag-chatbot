"""
RAG Pipeline — Phase 4: RAG Core with LlamaIndex
==================================================
This is the brain of the chatbot. It connects three things:
  1. Your ChromaDB vector store (where the embedded chunks live)
  2. The Groq LLM (which generates natural language answers)
  3. LlamaIndex's query engine (which orchestrates retrieval + generation)

How a query flows through the pipeline:
  ┌─────────────────────────────────────────────────────┐
  │ User asks: "What is self-attention?"                │
  │                                                     │
  │ Step 1: Embed the question                          │
  │   → "What is self-attention?" becomes a 384-dim     │
  │     vector using the same MiniLM model              │
  │                                                     │
  │ Step 2: Retrieve similar chunks from ChromaDB       │
  │   → ChromaDB compares the question vector against   │
  │     all 77 stored vectors using cosine similarity   │
  │   → Returns the top-k most similar chunks           │
  │     (default k=3, meaning 3 chunks)                 │
  │                                                     │
  │ Step 3: Build a prompt                              │
  │   → LlamaIndex assembles: system prompt + retrieved │
  │     chunks + user question into one LLM prompt      │
  │                                                     │
  │ Step 4: Send to Groq LLM                            │
  │   → Groq runs llama3-8b-8192 and returns an answer  │
  │   → The answer is grounded in the retrieved chunks  │
  └─────────────────────────────────────────────────────┘

Why Groq?
  Groq provides free API access to open-source LLMs with extremely fast
  inference. We're using llama3-8b-8192 which is good enough for a
  chatbot and keeps costs at zero.

What is top_k?
  When retrieving chunks, top_k controls how many chunks to fetch.
  - top_k=3: focused answers, less context, faster
  - top_k=5: broader answers, more context, slightly slower
  - top_k=10: very broad, risks including irrelevant chunks
  3 is a good default. You can tune this based on answer quality.

What is similarity_top_k vs top_k?
  Same thing — LlamaIndex calls it similarity_top_k in the retriever
  config. It means "return the top k most similar chunks."
"""

import os
from typing import Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq
from loguru import logger

from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import load_existing_index
from src.rag.guardrails import SYSTEM_PROMPT


# ── Default config ───────────────────────────────────
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.1  # low = more deterministic/factual answers


class RAGPipeline:
    """
    The main RAG query pipeline.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("What is the transformer architecture?")
        print(result["answer"])
        print(result["sources"])

    What happens when you call query():
        1. Your question is embedded (text → vector)
        2. ChromaDB retrieves the top_k most similar chunks
        3. Those chunks + your question are sent to Groq's LLM
        4. The LLM generates an answer grounded in those chunks
        5. You get back the answer + the source chunks used
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialise the RAG pipeline.

        Args:
            model:       Groq model name (llama3-8b-8192 is the free tier default)
            top_k:       number of chunks to retrieve per query
            temperature: LLM creativity (0.0 = deterministic, 1.0 = creative)
                         We keep it low (0.1) because we want factual answers,
                         not creative writing.
        """
        self.model = model
        self.top_k = top_k
        self.temperature = temperature

        logger.info("Initialising RAG Pipeline...")
        self._setup()
        logger.info("RAG Pipeline ready ✅")

    def _setup(self):
        """
        Wire together: embedding model + LLM + vector store + query engine.

        This is where LlamaIndex's abstractions shine — it handles the
        orchestration of retrieval → prompt building → LLM call for you.
        """
        # ── 1. Load the embedding model ──────────
        # Same model used during ingestion — MUST be the same, otherwise
        # the question vector would be in a different "space" than the
        # stored chunk vectors, and similarity search would return garbage.
        embed_model = get_embedding_model()
        Settings.embed_model = embed_model

        # ── 2. Set up the Groq LLM ──────────────
        # The API key is read from the GROQ_API_KEY environment variable.
        # If it's not set, this will raise an error.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables.\n"
                "Add it to your .env file: GROQ_API_KEY=your-key-here"
            )

        self.llm = Groq(
            model=self.model,
            api_key=api_key,
            temperature=self.temperature,
        )
        Settings.llm = self.llm
        logger.info(f"LLM: {self.model} (temperature={self.temperature})")

        # ── 3. Load the vector store index ───────
        # This connects to ChromaDB and loads the existing embeddings.
        # No re-embedding happens here — it's just reading from disk.
        self.index = load_existing_index()
        if self.index is None:
            raise RuntimeError(
                "No index found in ChromaDB. Run ingestion first:\n"
                "  python scripts/ingest_data.py"
            )

        # ── 4. Build the query engine ────────────
        # The query engine combines:
        #   - Retriever: fetches top_k similar chunks from ChromaDB
        #   - Response synthesizer: builds the prompt and calls the LLM
        #
        # "compact" mode means: stuff all retrieved chunks into a single
        # prompt (as opposed to "refine" which calls the LLM once per chunk
        # and iteratively refines the answer — slower but sometimes better
        # for very long contexts).
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
            # This is the system prompt that tells the LLM how to behave.
            # It instructs the model to only answer from the provided context.
            text_qa_template=SYSTEM_PROMPT,
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        logger.info(f"Query engine ready (top_k={self.top_k})")

    def query(self, question: str) -> dict:
        """
        Ask a question and get a grounded answer.

        Args:
            question: the user's question (natural language string)

        Returns:
            dict with:
                - "answer":  the LLM's response (grounded in retrieved chunks)
                - "sources": list of source chunks used, each with:
                    - "text": the chunk content
                    - "file_name": which document it came from
                    - "score": similarity score (higher = more relevant)

        What happens internally:
            1. question → embedding model → 384-dim vector
            2. vector → ChromaDB → top_k most similar chunks
            3. chunks + question → Groq LLM → answer
        """
        if not question.strip():
            return {"answer": "Please ask a question.", "sources": []}

        logger.info(f"Query: {question}")

        # LlamaIndex handles the full flow: embed → retrieve → synthesize
        response = self.query_engine.query(question)

        # Extract source information from the response
        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.node.text[:500],  # first 500 chars of the chunk
                "file_name": node.node.metadata.get("file_name", "unknown"),
                "score": round(node.score, 4) if node.score else None,
            })

        logger.info(f"Answer generated from {len(sources)} sources")
        for s in sources:
            logger.debug(f"  → {s['file_name']} (score: {s['score']})")

        return {
            "answer": str(response),
            "sources": sources,
        }