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
import time

from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq
from loguru import logger

from src.indexing.embeddings import get_embedding_model
from src.indexing.vector_store import DEFAULT_PERSIST_DIR, load_existing_index
from src.rag.guardrails import SYSTEM_PROMPT, Guardrails

# ── Default config ───────────────────────────────────
DEFAULT_MODEL = "openai/gpt-oss-20b"
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
        with_llm: bool = True,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ):
        """
        with_llm=False skips the Groq setup so retrieval + guardrails can be
        run (and evaluated) offline with no API key. query() needs the LLM;
        retrieve() does not.
        """
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.with_llm = with_llm
        self.persist_dir = persist_dir

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
        self.llm = None
        if self.with_llm:
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
        else:
            logger.info("LLM disabled — retrieval + guardrails only")

        # ── 3. Vector store index ────────────────
        self.index = load_existing_index(persist_dir=self.persist_dir)
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
        self.synthesizer = None
        if self.with_llm:
            self.synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode="compact",
                text_qa_template=SYSTEM_PROMPT,
            )

        # ── 6. Guardrails ────────────────────────
        # Initialises scope checking (pre-computes reference embeddings).
        self.guardrails = Guardrails()

        logger.info(f"Query engine ready (top_k={self.top_k}, with guardrails)")

    def retrieve(self, question: str) -> dict:
        """
        Run everything up to (but not including) the LLM call:
        scope check → retrieval → confidence check → source filtering.

        Needs no API key, so it is also what the offline evaluation uses.

        Returns:
            dict with:
                - "guardrail_action": "passed", "scope_rejected",
                                      "low_confidence" or "empty_query"
                - "message":          fallback text when a guardrail fired
                - "nodes":            chunks to send to the LLM (filtered)
                - "retrieved_nodes":  all top_k chunks, before filtering
                - "timings":          per-stage latency in milliseconds
        """
        timings = {}
        result = {
            "guardrail_action": "passed",
            "message": None,
            "nodes": [],
            "retrieved_nodes": [],
            "timings": timings,
        }

        if not question.strip():
            result["guardrail_action"] = "empty_query"
            result["message"] = "Please ask a question."
            return result

        # ── Layer 1: Scope check ─────────────────
        # Is this question about AI/ML/Data Analytics?
        # If not, reject immediately without hitting ChromaDB or Groq.
        t0 = time.perf_counter()
        in_scope, scope_message = self.guardrails.check_scope(question)
        timings["scope_ms"] = (time.perf_counter() - t0) * 1000
        if not in_scope:
            logger.info("Guardrail: scope rejected")
            result["guardrail_action"] = "scope_rejected"
            result["message"] = scope_message
            return result

        # ── Retrieve chunks from ChromaDB ────────
        # This embeds the question and fetches the top_k most similar chunks.
        # No LLM call happens here — just vector similarity search.
        t0 = time.perf_counter()
        source_nodes = self.retriever.retrieve(question)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        result["retrieved_nodes"] = source_nodes

        # ── Layer 2 + 3: Confidence check + filtering ─
        # Are the chunks relevant enough? Filter out low-quality ones.
        passed, filtered_nodes, confidence_message = self.guardrails.check_confidence(
            source_nodes
        )
        if not passed:
            logger.info("Guardrail: low confidence")
            result["guardrail_action"] = "low_confidence"
            result["message"] = confidence_message
            return result

        result["nodes"] = filtered_nodes
        return result

    def query(self, question: str) -> dict:
        """
        Ask a question with full guardrail protection.

        Returns:
            dict with:
                - "answer":           the response text
                - "sources":          list of source chunks used (UI previews)
                - "contexts":         full text of the chunks sent to the LLM
                - "timings":          per-stage latency in milliseconds
                - "guardrail_action": what happened
                    "passed"             → normal RAG answer
                    "scope_rejected"     → question was out of scope
                    "low_confidence"     → retrieved chunks weren't relevant
                    "empty_query"        → user sent blank input
        """
        if self.synthesizer is None:
            raise RuntimeError("query() needs the LLM — create with with_llm=True")

        t_start = time.perf_counter()
        logger.info(f"Query: {question}")

        stage = self.retrieve(question)
        timings = stage["timings"]

        if stage["guardrail_action"] != "passed":
            timings["total_ms"] = (time.perf_counter() - t_start) * 1000
            return {
                "answer": stage["message"],
                "sources": self._format_sources(stage["retrieved_nodes"]),
                "contexts": [],
                "timings": timings,
                "guardrail_action": stage["guardrail_action"],
            }

        filtered_nodes = stage["nodes"]

        # ── Send to LLM ─────────────────────────
        # Only the filtered (high-quality) chunks go to the LLM.
        # The synthesizer builds the prompt from the system template +
        # filtered chunks + question, then calls Groq.
        t0 = time.perf_counter()
        response = self.synthesizer.synthesize(
            question,
            nodes=filtered_nodes,
        )
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"Answer generated from {len(filtered_nodes)} sources "
            f"(filtered from {len(stage['retrieved_nodes'])}) "
            f"in {timings['total_ms']:.0f} ms"
        )

        return {
            "answer": str(response),
            "sources": self._format_sources(filtered_nodes),
            "contexts": [node.node.get_content() for node in filtered_nodes],
            "timings": timings,
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
