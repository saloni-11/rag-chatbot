"""
RAG Pipeline — Phase 5 Update
==============================
Updated to integrate guardrails between the retrieval and LLM steps.

The query flow is now:

  User Question
      │
      ▼
  [Scope Check] ──── out of scope ──→ "Ask about AI/ML..."
      │ in scope
      ▼
  [Retrieve chunks from ChromaDB]
      │
      ▼
  [Confidence Check] ── low confidence ──→ "I don't have enough info..."
      │ passed
      ▼
  [Filter low-quality chunks]
      │
      ▼
  [Send filtered chunks + question to Groq LLM]
      │
      ▼
  [Return answer + sources]

Why do we split retrieval and synthesis?
  In Phase 4, we used RetrieverQueryEngine which bundles retrieval + LLM
  into one call. That's convenient but means we can't inspect or filter
  the retrieved chunks before they reach the LLM.

  Now we call the retriever and synthesizer separately so we can insert
  guardrail checks in between. This is a common pattern in production
  RAG systems — you almost always want to inspect/filter/rerank retrieved
  chunks before passing them to the LLM.
"""

import os

from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq
from loguru import logger

from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import load_existing_index
from src.rag.guardrails import SYSTEM_PROMPT, Guardrails

# ── Default config ───────────────────────────────────
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.1


class RAGPipeline:
    """
    The main RAG query pipeline with guardrails.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("What is the transformer architecture?")
        print(result["answer"])
        print(result["sources"])
        print(result["guardrail_action"])  # "passed", "scope_rejected", etc.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.top_k = top_k
        self.temperature = temperature

        logger.info("Initialising RAG Pipeline...")
        self._setup()
        logger.info("RAG Pipeline ready ✅")

    def _setup(self):
        """
        Wire together all components.

        Key change from Phase 4:
          We no longer use RetrieverQueryEngine (which bundles everything).
          Instead we keep the retriever and synthesizer as separate objects
          so we can run guardrails between them.
        """
        # ── 1. Embedding model ───────────────────
        embed_model = get_embedding_model()
        Settings.embed_model = embed_model

        # ── 2. Groq LLM ─────────────────────────
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

        # ── 3. Vector store index ────────────────
        self.index = load_existing_index()
        if self.index is None:
            raise RuntimeError(
                "No index found in ChromaDB. Run ingestion first:\n"
                "  python scripts/ingest_data.py"
            )

        # ── 4. Retriever (separate from synthesizer) ─
        # The retriever ONLY fetches chunks — it doesn't call the LLM.
        # This lets us inspect and filter chunks before the LLM sees them.
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

        # ── 5. Response synthesizer (calls the LLM) ─
        # "compact" stuffs all chunks into one prompt.
        # The system prompt template tells the LLM how to behave.
        self.synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode="compact",
            text_qa_template=SYSTEM_PROMPT,
        )

        # ── 6. Guardrails ────────────────────────
        # Initialises scope checking (pre-computes reference embeddings).
        self.guardrails = Guardrails()

        logger.info(f"Query engine ready (top_k={self.top_k}, with guardrails)")

    def query(self, question: str) -> dict:
        """
        Ask a question with full guardrail protection.

        Returns:
            dict with:
                - "answer":           the response text
                - "sources":          list of source chunks used
                - "guardrail_action": what happened
                    "passed"             → normal RAG answer
                    "scope_rejected"     → question was out of scope
                    "low_confidence"     → retrieved chunks weren't relevant
                    "empty_query"        → user sent blank input
        """
        if not question.strip():
            return {
                "answer": "Please ask a question.",
                "sources": [],
                "guardrail_action": "empty_query",
            }

        logger.info(f"Query: {question}")

        # ── Layer 1: Scope check ─────────────────
        # Is this question about AI/ML/Data Analytics?
        # If not, reject immediately without hitting ChromaDB or Groq.
        in_scope, scope_message = self.guardrails.check_scope(question)
        if not in_scope:
            logger.info("Guardrail: scope rejected")
            return {
                "answer": scope_message,
                "sources": [],
                "guardrail_action": "scope_rejected",
            }

        # ── Retrieve chunks from ChromaDB ────────
        # This embeds the question and fetches the top_k most similar chunks.
        # No LLM call happens here — just vector similarity search.
        source_nodes = self.retriever.retrieve(question)

        # ── Layer 2 + 3: Confidence check + filtering ─
        # Are the chunks relevant enough? Filter out low-quality ones.
        passed, filtered_nodes, confidence_message = self.guardrails.check_confidence(
            source_nodes
        )
        if not passed:
            logger.info("Guardrail: low confidence")
            return {
                "answer": confidence_message,
                "sources": self._format_sources(source_nodes),
                "guardrail_action": "low_confidence",
            }

        # ── Send to LLM ─────────────────────────
        # Only the filtered (high-quality) chunks go to the LLM.
        # The synthesizer builds the prompt from the system template +
        # filtered chunks + question, then calls Groq.
        response = self.synthesizer.synthesize(
            question,
            nodes=filtered_nodes,
        )

        logger.info(
            f"Answer generated from {len(filtered_nodes)} sources "
            f"(filtered from {len(source_nodes)})"
        )

        return {
            "answer": str(response),
            "sources": self._format_sources(filtered_nodes),
            "guardrail_action": "passed",
        }

    def _format_sources(self, nodes) -> list:
        """
        Format source nodes into a clean list of dicts.

        Extracts just the useful information — text preview, filename,
        and similarity score — from LlamaIndex's NodeWithScore objects.
        """
        sources = []
        for node in nodes:
            sources.append(
                {
                    "text": node.node.text[:500],
                    "file_name": node.node.metadata.get("file_name", "unknown"),
                    "score": round(node.score, 4) if node.score else None,
                }
            )
        return sources
