"""
scripts/tune_hybrid.py — tune hybrid retrieval + reranking on the TUNING set
============================================================================
Tunes the candidates per retriever, the RRF k constant, the rerank pool cap
and the final top-k against tests/eval_dataset.json, scoring each config
with the same RAGAS context precision / context recall judge as the main
evaluation (src/evaluation/ragas_eval.py).

It never reads the frozen held-out sets: any dataset path containing
"heldout" is refused.

How it stays affordable on Groq's free tier:
  1. COLLECT (offline): for every question, retrieve once with the widest
     settings — dense top 20, BM25 top 20 — and cross-encoder-score the
     union. Every config is then simulated from this data, so all configs
     see the same retrieval (a paired comparison; HNSW cannot differ
     between them).
  2. CONTEXT PRECISION: RAGAS judges each retrieved chunk on its own
     ("was this chunk useful for the reference answer?") and combines the
     verdicts into average precision. So a verdict is cached per
     (question, chunk) and shared by every config that retrieves that chunk.
  3. CONTEXT RECALL judges all contexts together, so it is cached per
     (question, ordered contexts) and only run for the configs named with
     --recall.

USAGE:
  python scripts/tune_hybrid.py                       # collect + simulate, no API
  python scripts/tune_hybrid.py --precision           # + judge precision
  python scripts/tune_hybrid.py --recall baseline,C1  # + judge recall for configs
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Importing ragas_eval stubs the vertexai import RAGAS needs and sets up logging
from src.evaluation import ragas_eval  # noqa: E402
from src.evaluation.ragas_eval import DEFAULT_JUDGE_MODEL, logger  # noqa: E402

CACHE_DIR = Path("data/eval_cache/tuning")
MAX_CANDIDATES = 20

GRID = {
    "candidates": [10, 15, 20],
    "rrf_k": [10, 30, 60],
    "rerank_pool": [0, 10, 15],  # 0 = rerank the whole fused pool
    "final_k": [3, 5],
}


# ─────────────────────────────────────────────────────
# Collect
# ─────────────────────────────────────────────────────


def collect(dataset: list, path: Path) -> dict:
    """Retrieve once per question with the widest settings and score the union."""
    from src.rag import guardrails as g
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        with_llm=False,
        retrieval_mode="hybrid",
        hybrid_options={"candidates_per_retriever": MAX_CANDIDATES},
    )
    h = pipeline.hybrid
    data = {"chunks": {}, "questions": {}}

    for i, entry in enumerate(dataset):
        q = entry["question"]
        logger.info(f"  [{i + 1}/{len(dataset)}] collecting {entry['id']}")
        scope_sim, _ = pipeline.guardrails.scope_score(q)
        qvec = h.embed_query(q)

        # The current pipeline: the dense VectorIndexRetriever at top_k=3
        baseline = [n.node.node_id for n in pipeline.retriever.retrieve(q)]
        dense = h.dense_search(qvec, MAX_CANDIDATES)
        bm25 = h.bm25_search(q, MAX_CANDIDATES)

        union = list(dict.fromkeys(baseline + dense + bm25))
        t0 = time.perf_counter()
        rerank = h.rerank_scores(q, union)
        rerank_ms = (time.perf_counter() - t0) * 1000

        for doc_id in union:
            node = h.nodes[doc_id]
            data["chunks"][doc_id] = {
                "file_name": node.metadata.get("file_name"),
                "text": h.texts[doc_id],
            }
        data["questions"][entry["id"]] = {
            "category": entry["category"],
            "scope_similarity": scope_sim,
            "baseline": baseline,
            "dense": dense,
            "bm25": bm25,
            "cosine": dict(zip(union, h.cosines(qvec, union))),
            "rerank": dict(zip(union, rerank)),
            "rerank_ms_union": rerank_ms,
        }

    data["thresholds"] = {
        "scope": g.SCOPE_THRESHOLD,
        "confidence_min_cosine": g.CONFIDENCE_MIN_COSINE,
        "source_min_cosine": g.SOURCE_MIN_COSINE,
    }
    data["reranker"] = h.reranker.model_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ─────────────────────────────────────────────────────
# Simulate configs (mirrors HybridRetriever + Guardrails.select_sources)
# ─────────────────────────────────────────────────────


def apply_config(qd: dict, thresholds: dict, cfg: dict) -> dict:
    from src.rag.hybrid import reciprocal_rank_fusion

    cos = qd["cosine"]
    if cfg["mode"] == "baseline":
        # check_confidence: best of the 3 must pass, then filter
        ids = qd["baseline"]
        best = max(cos[i] for i in ids)
        top = [i for i in ids if cos[i] >= thresholds["source_min_cosine"]]
        displaced = len(ids) - len(top)
        pool_size = 0
    else:
        n = cfg["candidates"]
        dense = qd["dense"][:n]
        bm25 = qd["bm25"][:n] if cfg.get("bm25", True) else []
        fused = [d for d, _ in reciprocal_rank_fusion([dense, bm25], cfg["rrf_k"])]
        if cfg["rerank_pool"]:
            fused = fused[: cfg["rerank_pool"]]
        pool_size = len(fused)
        ranked = (
            sorted(fused, key=lambda d: -qd["rerank"][d])
            if cfg.get("rerank", True)
            else fused
        )
        kept = [d for d in ranked if cos[d] >= thresholds["source_min_cosine"]]
        displaced = sum(1 for d in ranked[: cfg["final_k"]] if d not in set(kept))
        top = kept[: cfg["final_k"]]
        best = max((cos[d] for d in top), default=-1.0)

    if qd["scope_similarity"] < thresholds["scope"]:
        action = "scope_rejected"
    elif not top or best < thresholds["confidence_min_cosine"]:
        action = "low_confidence"
    else:
        action = "passed"
    return {
        "action": action,
        "contexts": top if action == "passed" else [],
        "displaced": displaced,
        "pool_size": pool_size,
    }


def build_configs() -> dict:
    configs = {"baseline": {"mode": "baseline", "final_k": 3}}
    for n, k, pool, final in itertools.product(*GRID.values()):
        if pool and pool >= 2 * n:
            continue  # cap larger than any possible pool: same as no cap
        if not pool and k != 60:
            continue  # k cannot change the order when the whole pool is reranked
        name = f"n{n}_k{k}_pool{pool or 'all'}_top{final}"
        configs[name] = {
            "mode": "hybrid",
            "candidates": n,
            "rrf_k": k,
            "rerank_pool": pool,
            "final_k": final,
        }
    # Ablations: each stage on its own
    for final in (3, 5):
        configs[f"dense_rerank_n15_top{final}"] = {
            "mode": "hybrid", "candidates": 15, "rrf_k": 60, "rerank_pool": 0,
            "final_k": final, "bm25": False,
        }
        configs[f"rrf_only_n15_k60_top{final}"] = {
            "mode": "hybrid", "candidates": 15, "rrf_k": 60, "rerank_pool": 0,
            "final_k": final, "rerank": False,
        }
    return configs


# ─────────────────────────────────────────────────────
# Judge (RAGAS metrics, cached)
# ─────────────────────────────────────────────────────


class Budget(Exception):
    pass


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:20]


def _load(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            out[rec["key"]] = rec
    return out


class Judge:
    def __init__(self, judge_model: str):
        from langchain_core.callbacks import BaseCallbackHandler
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

        llm, _ = ragas_eval.get_judge(judge_model)
        self.model = judge_model
        self.precision = LLMContextPrecisionWithReference(llm=llm)
        self.recall = LLMContextRecall(llm=llm)
        self.tokens = 0
        self.calls = 0

        judge = self

        class Usage(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                judge.calls += 1
                usage = (response.llm_output or {}).get("token_usage") or {}
                judge.tokens += usage.get("total_tokens", 0)

        self.callbacks = [Usage()]

    def _score(self, metric, sample) -> float:
        for attempt in range(6):
            try:
                score = asyncio.run(
                    metric.single_turn_ascore(sample, callbacks=self.callbacks)
                )
                if score is not None and not math.isnan(score):
                    return float(score)
                logger.warning("    judge returned NaN, retrying")
            except Exception as e:  # noqa: BLE001 — Groq errors arrive wrapped
                msg = str(e)
                if "per day" in msg or "TPD" in msg or "RPD" in msg:
                    raise Budget(msg[:300])
                wait = min(60, 5 * 2**attempt)
                logger.warning(f"    judge error ({msg[:120]}), retry in {wait}s")
                time.sleep(wait)
        return float("nan")

    def precision_verdict(self, entry, chunk_text) -> float:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=entry["question"],
            retrieved_contexts=[chunk_text],
            reference=entry["ground_truth"],
        )
        return self._score(self.precision, sample)

    def recall_score(self, entry, contexts) -> float:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=entry["question"],
            retrieved_contexts=contexts,
            reference=entry["ground_truth"],
        )
        return self._score(self.recall, sample)


def average_precision(verdicts: list) -> float:
    """RAGAS's LLMContextPrecisionWithReference formula."""
    denominator = sum(verdicts) + 1e-10
    numerator = sum(
        (sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts))
    )
    return numerator / denominator


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--recollect", action="store_true")
    parser.add_argument("--precision", action="store_true", help="judge precision")
    parser.add_argument("--recall", default="", help="comma-separated config names")
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--output", default="data/hybrid_tuning.json")
    args = parser.parse_args()

    if "heldout" in Path(args.dataset).name:
        sys.exit("Refusing to tune on a held-out set. Use the tuning set.")

    dataset = ragas_eval.load_eval_dataset(args.dataset)
    by_id = {e["id"]: e for e in dataset}

    cand_path = CACHE_DIR / "candidates.json"
    if cand_path.exists() and not args.recollect:
        data = json.loads(cand_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded collected candidates from {cand_path}")
    else:
        data = collect(dataset, cand_path)
    thresholds = data["thresholds"]

    configs = build_configs()
    results = {
        name: {
            qid: apply_config(qd, thresholds, cfg)
            for qid, qd in data["questions"].items()
        }
        for name, cfg in configs.items()
    }

    answerable = [q for q, e in by_id.items() if e["category"] == "answerable"]
    # Judge in priority order (baseline, top-3 configs, top-5, ablations) so
    # a run cut short by the daily quota has the most useful verdicts first
    def priority(name):
        cfg = configs[name]
        if cfg["mode"] == "baseline":
            return 0
        ablation = cfg.get("bm25") is False or cfg.get("rerank") is False
        return 3 if ablation else (1 if cfg["final_k"] == 3 else 2)

    pairs = list(
        dict.fromkeys(
            (q, c)
            for name in sorted(results, key=priority)
            for q in answerable
            for c in results[name][q]["contexts"]
        )
    )
    logger.info(f"{len(configs)} configs | {len(pairs)} unique (question, chunk) pairs")

    judge_model = args.judge_model
    p_path = CACHE_DIR / f"precision_{judge_model.replace('/', '_')}.jsonl"
    r_path = CACHE_DIR / f"recall_{judge_model.replace('/', '_')}.jsonl"
    p_cache, r_cache = _load(p_path), _load(r_path)

    def p_key(q, c):
        e = by_id[q]
        return _key(judge_model, e["question"], e["ground_truth"], data["chunks"][c]["text"])

    def r_key(q, ctx):
        e = by_id[q]
        return _key(
            judge_model, e["question"], e["ground_truth"],
            [data["chunks"][c]["text"] for c in ctx],
        )

    recall_configs = [c for c in args.recall.split(",") if c]
    for c in recall_configs:
        if c not in configs:
            sys.exit(f"Unknown config {c!r}")

    judge = None
    if args.precision or recall_configs:
        judge = Judge(judge_model)
        try:
            if args.precision:
                todo = [p for p in pairs if p_key(*p) not in p_cache]
                logger.info(f"Precision: {len(todo)} pairs to judge")
                for q, c in todo:
                    if judge.calls >= args.max_calls:
                        raise Budget("--max-calls reached")
                    v = judge.precision_verdict(by_id[q], data["chunks"][c]["text"])
                    if math.isnan(v):
                        continue
                    rec = {"key": p_key(q, c), "qid": q, "chunk": c, "verdict": round(v)}
                    p_cache[rec["key"]] = rec
                    with open(p_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    logger.info(f"    {q} {c[:8]} -> {rec['verdict']} (tokens so far {judge.tokens})")
            for name in recall_configs:
                for q in answerable:
                    ctx = results[name][q]["contexts"]
                    if not ctx or r_key(q, ctx) in r_cache:
                        continue
                    if judge.calls >= args.max_calls:
                        raise Budget("--max-calls reached")
                    s = judge.recall_score(by_id[q], [data["chunks"][c]["text"] for c in ctx])
                    if math.isnan(s):
                        continue
                    rec = {"key": r_key(q, ctx), "qid": q, "contexts": ctx, "score": s}
                    r_cache[rec["key"]] = rec
                    with open(r_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    logger.info(f"    recall {name} {q} -> {s:.3f} (tokens so far {judge.tokens})")
        except Budget as e:
            logger.warning(f"Stopped judging: {e}")
        logger.info(f"Judge usage this run: {judge.calls} calls, {judge.tokens} tokens")

    # ── Score every config ────────────────────────
    negatives = [q for q, e in by_id.items() if e["category"] != "answerable"]
    table = {}
    for name, res in results.items():
        cps, crs, missing_p = [], [], 0
        for q in answerable:
            ctx = res[q]["contexts"]
            if not ctx:
                continue
            verdicts = [p_cache.get(p_key(q, c), {}).get("verdict") for c in ctx]
            if None in verdicts:
                missing_p += 1
            else:
                cps.append(average_precision(verdicts))
            r = r_cache.get(r_key(q, ctx))
            if r:
                crs.append(r["score"])
        passed = [q for q in answerable if res[q]["action"] == "passed"]
        table[name] = {
            "config": configs[name],
            "context_precision": round(statistics.mean(cps), 4) if cps else None,
            "precision_n": len(cps),
            "precision_missing": missing_p,
            "context_recall": round(statistics.mean(crs), 4) if crs else None,
            "recall_n": len(crs),
            "answerable_pass": f"{len(passed)}/{len(answerable)}",
            "negatives_declined": (
                f"{sum(res[q]['action'] != 'passed' for q in negatives)}/{len(negatives)}"
            ),
            "source_filter_displaced": sum(res[q]["displaced"] for q in passed),
            "questions_with_displacement": sum(1 for q in passed if res[q]["displaced"]),
            "mean_rerank_pairs": round(
                statistics.mean(res[q]["pool_size"] for q in by_id), 1
            ),
        }

    ranked = sorted(
        table.items(),
        key=lambda kv: -(kv[1]["context_precision"] or 0),
    )
    print(f"\n{'config':34s} {'CP':>6s} {'n':>3s} {'CR':>6s} {'n':>3s}  pass   decl  disp  pairs")
    for name, row in ranked:
        cp = f"{row['context_precision']:.3f}" if row["context_precision"] is not None else "  -  "
        cr = f"{row['context_recall']:.3f}" if row["context_recall"] is not None else "  -  "
        print(
            f"{name:34s} {cp:>6s} {row['precision_n']:3d} {cr:>6s} {row['recall_n']:3d}  "
            f"{row['answerable_pass']:5s} {row['negatives_declined']:5s} "
            f"{row['source_filter_displaced']:4d}  {row['mean_rerank_pairs']:5.1f}"
        )

    out = {
        "dataset": args.dataset,
        "judge_model": judge_model,
        "thresholds": thresholds,
        "reranker": data["reranker"],
        "grid": GRID,
        "configs": table,
        "per_question": {
            name: {q: res[q] for q in by_id} for name, res in results.items()
        },
    }
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"Saved {args.output}")


if __name__ == "__main__":
    main()
