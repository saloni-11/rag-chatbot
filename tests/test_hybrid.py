"""
tests/test_hybrid.py
=====================
Unit tests for hybrid retrieval (BM25 + dense, RRF fusion, reranking).

The retriever is built without ChromaDB or the real models: a fake
collection returns a fixed dense ranking and a fake reranker scores
chunks from a lookup table, so each stage's logic is tested on its own.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from rank_bm25 import BM25Okapi

from src.rag.guardrails import chroma_score_to_cosine
from src.rag.hybrid import HybridRetriever, reciprocal_rank_fusion, tokenize

# ── Tokenizer ────────────────────────────────────────


def test_tokenize_folds_pdf_ligatures():
    """'ﬁ' in PDF text must match 'fi' typed in a question."""
    assert tokenize("ﬁne-tuning") == tokenize("fine-tuning") == ["fine", "tuning"]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the BLEU score?") == ["bleu", "score"]


# ── Reciprocal Rank Fusion ───────────────────────────


def test_rrf_scores_by_rank():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_lists():
    order = [d for d, _ in reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)]
    assert order[0] == "b"


def test_rrf_ties_keep_first_seen_order():
    """Equal scores fall back to dense-first order, so fusion is deterministic."""
    order = [d for d, _ in reciprocal_rank_fusion([["a"], ["b"]], k=60)]
    assert order == ["a", "b"]


# ── HybridRetriever ──────────────────────────────────

TEXTS = {
    "d1": "attention mechanism transformer encoder",
    "d2": "BLEU score results on WMT 2014 English German",
    "d3": "residual learning deep networks",
    "d4": "generative adversarial nets minimax game",
}


def make_retriever(dense_ids, rerank_table=None, use_bm25=True, candidates=2, pool=0):
    """HybridRetriever wired to fakes instead of ChromaDB and real models."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.ids = list(TEXTS)
    r.texts = dict(TEXTS)
    r._row = {d: i for i, d in enumerate(r.ids)}
    r.nodes = {
        d: TextNode(id_=d, text=t, metadata={"file_name": f"{d}.pdf"})
        for d, t in TEXTS.items()
    }
    # Embeddings chosen so cosine with the query vector [1, 0] is known
    cosines = {"d1": 0.6, "d2": 0.3, "d3": 0.1, "d4": -0.2}
    r.embeddings = np.array(
        [[cosines[d], np.sqrt(1 - cosines[d] ** 2)] for d in r.ids], dtype=np.float32
    )
    r.embed_model = MagicMock()
    r.embed_model.get_query_embedding.return_value = [1.0, 0.0]
    r.collection = MagicMock()
    r.collection.query.side_effect = lambda query_embeddings, n_results, include: {
        "ids": [dense_ids[:n_results]]
    }
    r.candidates_per_retriever = candidates
    r.rrf_k = 60
    r.rerank_pool = pool
    r.bm25 = BM25Okapi([tokenize(TEXTS[d]) for d in r.ids]) if use_bm25 else None
    r.reranker = None
    if rerank_table is not None:
        r.reranker = MagicMock(model_name="fake")
        r.reranker.score.side_effect = lambda q, texts: [
            rerank_table[next(d for d, t in TEXTS.items() if t == text)]
            for text in texts
        ]
    return r


def test_bm25_adds_keyword_match_dense_missed():
    r = make_retriever(dense_ids=["d1", "d3"], rerank_table=None)
    result = r.retrieve("BLEU score on WMT 2014")
    ids = [c.node_id for c in result.candidates]
    assert "d2" in ids
    d2 = next(c for c in result.candidates if c.node_id == "d2")
    assert d2.dense_rank is None and d2.bm25_rank == 1


def test_reranker_decides_final_order():
    table = {"d1": 0.1, "d2": 5.0, "d3": 2.0, "d4": -3.0}
    r = make_retriever(dense_ids=["d1", "d3"], rerank_table=table)
    result = r.retrieve("BLEU score on WMT 2014")
    assert [c.node_id for c in result.candidates] == ["d2", "d3", "d1"]
    assert "rerank_ms" in result.timings


def test_scores_are_exact_cosine_on_chroma_scale():
    r = make_retriever(dense_ids=["d1", "d3"], rerank_table=None)
    result = r.retrieve("BLEU score on WMT 2014")
    by_id = {n.node.node_id: chroma_score_to_cosine(n.score) for n in result.nodes}
    assert by_id["d1"] == pytest.approx(0.6, abs=1e-5)
    assert by_id["d2"] == pytest.approx(0.3, abs=1e-5)  # BM25-only chunk


def test_rerank_pool_caps_pairs_sent_to_reranker():
    table = {d: 0.0 for d in TEXTS}
    r = make_retriever(dense_ids=["d1", "d3"], rerank_table=table, pool=2)
    result = r.retrieve("BLEU score on WMT 2014")
    assert len(result.candidates) == 2
    assert len(r.reranker.score.call_args[0][1]) == 2


def test_stages_can_be_switched_off():
    """No BM25 and no reranker leaves the dense ranking untouched."""
    r = make_retriever(dense_ids=["d3", "d1"], rerank_table=None, use_bm25=False)
    result = r.retrieve("BLEU score on WMT 2014")
    assert [c.node_id for c in result.candidates] == ["d3", "d1"]
