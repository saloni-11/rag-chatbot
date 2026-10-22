"""
Hybrid Retrieval + Reranking
============================
Replaces "take the dense top 3" with a wider search that is narrowed down
by a model that reads the question and each chunk together.

  Question
     │
     ├──► Dense (ChromaDB, all-MiniLM-L6-v2)  top N by cosine similarity
     ├──► BM25 (rank-bm25, same chunk text)    top N by keyword overlap
     │
     ▼
  [Reciprocal Rank Fusion]   merge the two rankings by rank, not score
     │
     ▼
  [Cross-encoder rerank]     cross-encoder/ms-marco-MiniLM-L-6-v2 scores
     │                       each (question, chunk) pair
     ▼
  Ranked candidate pool  →  guardrails (source filter, top-k cut, confidence)

Why hybrid?
  The evaluation's main failure was "right paper, wrong passage". Dense
  retrieval matches on meaning, so it finds the right paper but can rank
  a generic passage above the one holding the specific fact. BM25 rewards
  exact terms (model names, numbers, jargon), which is what those
  specific passages share with the question.

Why Reciprocal Rank Fusion?
  Cosine similarity (-1..1) and BM25 (unbounded) are on different scales,
  so adding them needs a normalisation that is itself a tuning problem.
  RRF only uses each chunk's rank in each list:
      score(chunk) = sum over lists of 1 / (k + rank)
  k damps the difference between neighbouring ranks; 60 is the usual
  default. With the whole fused pool reranked, k cannot change the final
  order — it only matters when RERANK_POOL caps the pool before reranking.

Why a cross-encoder?
  The embedding model encodes question and chunk separately and compares
  two vectors. A cross-encoder reads them together, so it can see whether
  the chunk actually answers the question. It is too slow to run over the
  whole corpus, which is why it only reranks the fused candidates.

Every stage can be switched off (use_bm25 / use_reranker) to measure it
on its own.

Scores:
  Each returned node's .score is on ChromaDB's exp(cosine - 1) scale, the
  same scale the dense retriever has always produced, so the guardrails
  and the API's source scores work unchanged. The cosine is computed
  exactly from the stored embeddings for every candidate, including
  chunks that only BM25 found.
"""

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from loguru import logger
from rank_bm25 import BM25Okapi

from src.rag.guardrails import cosine_to_chroma_score

# ── Config (tuned on tests/eval_dataset.json with scripts/tune_hybrid.py) ─
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# How many chunks each retriever contributes before fusion
CANDIDATES_PER_RETRIEVER = int(os.getenv("HYBRID_CANDIDATES", "15"))
RRF_K = int(os.getenv("RRF_K", "60"))
# Rerank only the first RERANK_POOL fused chunks (0 = rerank all of them).
# This is a latency lever: the cross-encoder is the slowest stage.
RERANK_POOL = int(os.getenv("RERANK_POOL", "10"))

# Words that carry no retrieval signal; BM25's IDF already discounts common
# words, this just stops "what is the ..." matching every chunk.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how in is it its
    of on or that the their this to was were what when where which who why
    with""".split()
)


def tokenize(text: str) -> List[str]:
    """
    Lower-case word tokens for BM25.

    NFKC folds the PDF ligatures ("ﬁ" -> "fi") so that "fine-tuning" in a
    question matches "ﬁne-tuning" in a chunk.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]


def reciprocal_rank_fusion(
    rankings: List[List[str]], k: int = RRF_K
) -> List[Tuple[str, float]]:
    """
    Merge ranked lists of ids by Reciprocal Rank Fusion.

    Returns (id, fused score) pairs, best first. Ties keep the order in
    which ids were first seen (dense list first), so the result is
    deterministic.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


@dataclass
class Candidate:
    """One chunk in the candidate pool, with where it came from."""

    node_id: str
    cosine: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


@dataclass
class HybridResult:
    nodes: List[NodeWithScore]  # final order, best first
    candidates: List[Candidate]  # same order, with per-stage details
    timings: Dict[str, float] = field(default_factory=dict)


class CrossEncoderReranker:
    """Thin wrapper so the model loads once and can be swapped in tests."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker: {model_name}")
        # 512 tokens is the model's limit; longer chunks are truncated
        self.model = CrossEncoder(model_name, max_length=512, device="cpu")
        self.model_name = model_name

    def score(self, question: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(question, t) for t in texts], show_progress_bar=False
        )
        return [float(s) for s in scores]


class HybridRetriever:
    """
    Dense + BM25 retrieval, fused with RRF, reranked by a cross-encoder.

    Reads every chunk (text, metadata, embedding) from the ChromaDB
    collection once at startup, so BM25 indexes exactly the chunks that
    dense retrieval searches.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embed_model,
        candidates_per_retriever: int = CANDIDATES_PER_RETRIEVER,
        rrf_k: int = RRF_K,
        rerank_pool: int = RERANK_POOL,
        use_bm25: bool = True,
        use_reranker: bool = True,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.embed_model = embed_model
        self.candidates_per_retriever = candidates_per_retriever
        self.rrf_k = rrf_k
        self.rerank_pool = rerank_pool
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker

        self.collection = chromadb.PersistentClient(
            path=str(persist_dir)
        ).get_collection(collection_name)
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        self.ids: List[str] = list(data["ids"])
        self.texts: Dict[str, str] = dict(zip(self.ids, data["documents"]))
        self._row = {doc_id: i for i, doc_id in enumerate(self.ids)}

        # Rebuild LlamaIndex nodes the same way the Chroma adapter does
        self.nodes = {}
        for doc_id, text, metadata in zip(
            self.ids, data["documents"], data["metadatas"]
        ):
            node = metadata_dict_to_node(metadata)
            node.set_content(text)
            self.nodes[doc_id] = node

        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        self.embeddings = matrix / (
            np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        )

        self.bm25 = None
        if use_bm25:
            self.bm25 = BM25Okapi([tokenize(self.texts[i]) for i in self.ids])

        self.reranker = None
        if use_reranker:
            self.reranker = reranker or CrossEncoderReranker()

        logger.info(
            f"Hybrid retriever ready: {len(self.ids)} chunks | "
            f"candidates/retriever={candidates_per_retriever} | rrf_k={rrf_k} | "
            f"rerank_pool={rerank_pool or 'all'} | bm25={use_bm25} | "
            f"reranker={self.reranker.model_name if self.reranker else None}"
        )

    # ── Individual stages (also used by the tuning script) ─

    def embed_query(self, question: str) -> np.ndarray:
        vector = np.asarray(
            self.embed_model.get_query_embedding(question), dtype=np.float32
        )
        return vector / (np.linalg.norm(vector) + 1e-10)

    def dense_search(self, query_vector: np.ndarray, n: int) -> List[str]:
        """Top n chunk ids from ChromaDB's HNSW index (the existing retriever)."""
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()], n_results=n, include=[]
        )
        return list(result["ids"][0])

    def bm25_search(self, question: str, n: int) -> List[str]:
        """Top n chunk ids by BM25; chunks sharing no query term are left out."""
        scores = self.bm25.get_scores(tokenize(question))
        order = np.argsort(-scores, kind="stable")[:n]
        return [self.ids[i] for i in order if scores[i] > 0]

    def cosines(self, query_vector: np.ndarray, ids: List[str]) -> List[float]:
        rows = self.embeddings[[self._row[i] for i in ids]]
        return [float(c) for c in rows @ query_vector]

    def rerank_scores(self, question: str, ids: List[str]) -> List[float]:
        return self.reranker.score(question, [self.texts[i] for i in ids])

    # ── Full pipeline ─────────────────────────────

    def retrieve(self, question: str) -> HybridResult:
        timings = {}
        n = self.candidates_per_retriever

        t0 = time.perf_counter()
        query_vector = self.embed_query(question)
        dense_ids = self.dense_search(query_vector, n)
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000

        bm25_ids: List[str] = []
        if self.bm25 is not None:
            t0 = time.perf_counter()
            bm25_ids = self.bm25_search(question, n)
            timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], self.rrf_k)
        if self.rerank_pool:
            fused = fused[: self.rerank_pool]
        fused_ids = [doc_id for doc_id, _ in fused]

        dense_rank = {d: r for r, d in enumerate(dense_ids, start=1)}
        bm25_rank = {d: r for r, d in enumerate(bm25_ids, start=1)}
        candidates = [
            Candidate(
                node_id=doc_id,
                cosine=cos,
                dense_rank=dense_rank.get(doc_id),
                bm25_rank=bm25_rank.get(doc_id),
                rrf_score=rrf_score,
            )
            for (doc_id, rrf_score), cos in zip(
                fused, self.cosines(query_vector, fused_ids)
            )
        ]

        if self.reranker is not None:
            t0 = time.perf_counter()
            for cand, score in zip(candidates, self.rerank_scores(question, fused_ids)):
                cand.rerank_score = score
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            # Stable sort: equal reranker scores keep their fused order
            candidates.sort(key=lambda c: -c.rerank_score)

        nodes = [
            NodeWithScore(
                node=self.nodes[c.node_id], score=cosine_to_chroma_score(c.cosine)
            )
            for c in candidates
        ]
        return HybridResult(nodes=nodes, candidates=candidates, timings=timings)
