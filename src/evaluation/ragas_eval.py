"""
src/evaluation/ragas_eval.py — Phase 10: RAG Evaluation
========================================================
Evaluates the pipeline in two stages, so each part is measured with the
right tool:

STAGE 1 — Retrieval + guardrails (offline, deterministic, no API key)
  Every question in the dataset carries a category and, for answerable
  questions, the source document(s) that contain the answer.
    Retrieval   Hit@k and MRR against the labelled source documents
    Guardrails  answerable questions let through, unanswerable and
                out-of-scope questions declined (and by which layer)
    Latency     scope check and retrieval, p50 / p95
  --sweep adds a grid search over SCOPE_THRESHOLD and CONFIDENCE_THRESHOLD.

STAGE 2 — Generation quality with RAGAS (needs GROQ_API_KEY)
  Runs the full pipeline on the answerable questions and scores the
  answers with an LLM judge:
    faithfulness                          answer supported by the contexts?
    answer_relevancy                      answer addresses the question?
    llm_context_precision_with_reference  retrieved chunks relevant?
    context_recall                        contexts cover the ground truth?
  Questions a guardrail declined are counted separately instead of being
  scored — a refusal has no claims to be faithful to. End-to-end latency
  is recorded for every answer.

  The judge is a separate, larger model than the generator (by default)
  so the 8B model is not grading its own answers. Scores are cached per
  question, so a run that hits Groq's rate limits can simply be re-run
  and it picks up where it stopped.

HOW TO RUN:
  python src/evaluation/ragas_eval.py                   # both stages
  python src/evaluation/ragas_eval.py --stage retrieval # offline only
  python src/evaluation/ragas_eval.py --stage retrieval --sweep
  python src/evaluation/ragas_eval.py --judge-model qwen/qwen3.8-27b
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# langchain_community >= 0.3 removed chat_models.vertexai; stub it so ragas can import
def _stub_vertexai():
    class ChatVertexAI:
        pass

    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod


_stub_vertexai()

from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

load_dotenv()

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
DEFAULT_CACHE_DIR = "data/eval_cache"

GENERATION_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
}


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def load_eval_dataset(dataset_path: str) -> list:
    """Load the evaluation dataset from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    counts = {}
    for entry in dataset:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1
    logger.info(f"Loaded {len(dataset)} questions from {path.name}: {counts}")
    return dataset


def _percentile(values: list, pct: float) -> float:
    """Nearest-rank percentile; fine for the small samples we have."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency_summary(values: list) -> dict:
    if not values:
        return {}
    return {
        "p50_ms": round(statistics.median(values), 1),
        "p95_ms": round(_percentile(values, 95), 1),
        "n": len(values),
    }


def _clean(value):
    """NaN is not valid JSON; store it as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _mean(values: list):
    valid = [v for v in values if v is not None]
    return round(statistics.mean(valid), 4) if valid else None


# ─────────────────────────────────────────────────────
# Stage 1 — Retrieval + guardrails
# ─────────────────────────────────────────────────────


def evaluate_retrieval(pipeline, dataset: list) -> dict:
    """
    Score the retriever and guardrails against the labelled dataset.

    Retrieval metrics call the retriever directly, so they measure retriever
    quality on its own; the guardrail decision is measured separately via
    pipeline.retrieve(), which runs the three guardrail layers.
    """
    from src.rag import guardrails as g

    rows = []
    for entry in dataset:
        question = entry["question"]

        scope_sim, scope_match = pipeline.guardrails.scope_score(question)
        nodes = pipeline.retriever.retrieve(question)
        stage = pipeline.retrieve(question)

        ranked_files = [n.node.metadata.get("file_name", "unknown") for n in nodes]
        scores = [n.score for n in nodes if n.score is not None]
        expected = set(entry.get("source_files", []))

        first_hit = next(
            (i + 1 for i, f in enumerate(ranked_files) if f in expected), None
        )

        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "question": question,
                "guardrail_action": stage["guardrail_action"],
                "scope_similarity": round(scope_sim, 4),
                "scope_best_match": scope_match,
                "best_retrieval_score": round(max(scores), 4) if scores else None,
                "retrieved_files": ranked_files,
                "expected_files": sorted(expected),
                "first_relevant_rank": first_hit,
                "timings": {k: round(v, 1) for k, v in stage["timings"].items()},
            }
        )

    answerable = [r for r in rows if r["category"] == "answerable"]
    negatives = [r for r in rows if r["category"] != "answerable"]

    def rate(items, predicate):
        return (
            round(sum(1 for r in items if predicate(r)) / len(items), 4)
            if items
            else None
        )

    declined = ("scope_rejected", "low_confidence")

    summary = {
        "retrieval": {
            f"hit_rate@{pipeline.top_k}": rate(
                answerable, lambda r: r["first_relevant_rank"] is not None
            ),
            "hit_rate@1": rate(answerable, lambda r: r["first_relevant_rank"] == 1),
            "mrr": (
                round(
                    statistics.mean(
                        1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                        for r in answerable
                    ),
                    4,
                )
                if answerable
                else None
            ),
            "n": len(answerable),
        },
        "guardrails": {
            "answerable_pass_rate": rate(
                answerable, lambda r: r["guardrail_action"] == "passed"
            ),
            "unanswerable_decline_rate": rate(
                [r for r in negatives if r["category"] == "unanswerable"],
                lambda r: r["guardrail_action"] in declined,
            ),
            "out_of_scope_decline_rate": rate(
                [r for r in negatives if r["category"] == "out_of_scope"],
                lambda r: r["guardrail_action"] in declined,
            ),
            "out_of_scope_rejected_by_scope_layer": rate(
                [r for r in negatives if r["category"] == "out_of_scope"],
                lambda r: r["guardrail_action"] == "scope_rejected",
            ),
            "thresholds": {
                "scope": g.SCOPE_THRESHOLD,
                "confidence": g.CONFIDENCE_THRESHOLD,
                "source_min_score": g.SOURCE_MIN_SCORE,
            },
        },
        "latency": {
            "scope_check": _latency_summary(
                [r["timings"]["scope_ms"] for r in rows if "scope_ms" in r["timings"]]
            ),
            "retrieval": _latency_summary(
                [
                    r["timings"]["retrieval_ms"]
                    for r in rows
                    if "retrieval_ms" in r["timings"]
                ]
            ),
        },
    }
    return {"summary": summary, "per_question": rows}


def sweep_thresholds(rows: list, source_min_score: float) -> list:
    """
    Grid-search SCOPE_THRESHOLD x CONFIDENCE_THRESHOLD using the scores
    recorded in stage 1 — no re-embedding needed.

    A question gets through when scope_similarity >= scope threshold and
    best_retrieval_score >= confidence threshold (and >= source_min_score,
    so at least one chunk survives filtering). Ranked by balanced accuracy:
    the mean of the answerable pass rate and the negative decline rate.
    """
    answerable = [r for r in rows if r["category"] == "answerable"]
    negatives = [r for r in rows if r["category"] != "answerable"]

    def passes(r, scope_t, conf_t):
        best = r["best_retrieval_score"] or 0.0
        return (
            r["scope_similarity"] >= scope_t
            and best >= conf_t
            and best >= source_min_score
        )

    results = []
    for scope_t in [round(0.10 + 0.05 * i, 2) for i in range(8)]:
        for conf_t in [round(0.25 + 0.05 * i, 2) for i in range(8)]:
            pass_rate = sum(passes(r, scope_t, conf_t) for r in answerable) / len(
                answerable
            )
            decline_rate = sum(not passes(r, scope_t, conf_t) for r in negatives) / len(
                negatives
            )
            results.append(
                {
                    "scope_threshold": scope_t,
                    "confidence_threshold": conf_t,
                    "answerable_pass_rate": round(pass_rate, 4),
                    "negative_decline_rate": round(decline_rate, 4),
                    "balanced_accuracy": round((pass_rate + decline_rate) / 2, 4),
                }
            )

    # Ties → prefer the loosest thresholds (fewest false rejections)
    results.sort(
        key=lambda r: (
            -r["balanced_accuracy"],
            r["scope_threshold"],
            r["confidence_threshold"],
        )
    )
    return results


# ─────────────────────────────────────────────────────
# Stage 2 — Generation quality with RAGAS
# ─────────────────────────────────────────────────────


def get_judge(judge_model: str):
    """
    Groq judge LLM + local HuggingFace embeddings wrapped for RAGAS.
    Keeps evaluation on free-tier tools — no OpenAI key needed.
    """
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env — needed for stage 2")

    # gpt-oss models reason before answering, and those hidden tokens count
    # against max_tokens. Faithfulness lists every statement in the answer,
    # so with the default budget the output gets cut off mid-JSON
    # (LLMDidNotFinishException). Give it room and keep reasoning short.
    extra = {"reasoning_effort": "low"} if "gpt-oss" in judge_model else {}
    judge = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=8192,
            **extra,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    )
    logger.info(f"Judge LLM: Groq ({judge_model}) | embeddings: all-MiniLM-L6-v2")
    return judge, embeddings


def _cache_key(entry: dict, pipeline) -> str:
    """Same question + ground truth + pipeline config → reuse the answer."""
    from src.rag import guardrails as g

    payload = json.dumps(
        [
            entry["question"],
            entry["ground_truth"],
            g.SYSTEM_PROMPT.get_template(),
            # The thresholds decide which chunks reach the LLM, so an answer
            # generated under different thresholds must not be reused.
            [g.SCOPE_THRESHOLD, g.CONFIDENCE_THRESHOLD, g.SOURCE_MIN_SCORE],
            pipeline.model,
            pipeline.top_k,
            pipeline.persist_dir,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def evaluate_generation(
    pipeline, dataset: list, judge_model: str, cache_dir: str
) -> dict:
    """Run the full pipeline on answerable questions and score with RAGAS."""
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    answerable = [e for e in dataset if e["category"] == "answerable"]
    judge, embeddings = get_judge(judge_model)

    metrics = {
        "faithfulness": Faithfulness(),
        # strictness=1: Groq does not support n>1 completions, so the default
        # of 3 would triple the judge calls against the rate limit
        "answer_relevancy": ResponseRelevancy(strictness=1),
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }
    # Groq's free tier rate-limits aggressively. RAGAS defaults to 16 workers
    # and turns every failed call into a silent NaN — keep concurrency low and
    # retry with backoff instead.
    run_config = RunConfig(max_workers=2, max_retries=8, max_wait=60, timeout=180)

    # One record per question: the generated answer plus whatever scores the
    # judge produced. Later lines win, so a re-run reuses the same answer and
    # only asks the judge for the metrics that failed last time.
    cache_path = Path(cache_dir) / f"{judge_model.replace('/', '_')}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cache[record["key"]] = record

    rows = []
    for i, entry in enumerate(answerable):
        logger.info(f"  [{i + 1}/{len(answerable)}] {entry['question'][:60]}")
        key = _cache_key(entry, pipeline)
        record = cache.get(key)

        if record is None:
            result = pipeline.query(entry["question"])
            record = {
                "key": key,
                "id": entry["id"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "guardrail_action": result["guardrail_action"],
                "sources": [s["file_name"] for s in result["sources"]],
                "timings": {k: round(v, 1) for k, v in result["timings"].items()},
                "scores": {},
            }
        else:
            logger.info("    answer loaded from cache")

        row = {
            "id": entry["id"],
            "question": entry["question"],
            "answer": record["answer"],
            "ground_truth": entry["ground_truth"],
            "guardrail_action": record["guardrail_action"],
            "sources": record["sources"],
            "timings": record["timings"],
            "scores": dict(record["scores"]),
        }
        rows.append(row)

        if record["guardrail_action"] != "passed":
            logger.warning(f"    declined by guardrail: {record['guardrail_action']}")
            continue

        missing = [m for m in GENERATION_METRICS if row["scores"].get(m) is None]
        if missing:
            sample = SingleTurnSample(
                user_input=entry["question"],
                response=record["answer"],
                retrieved_contexts=record["contexts"],
                reference=entry["ground_truth"],
            )
            scored = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[metrics[m] for m in missing],
                llm=judge,
                embeddings=embeddings,
                run_config=run_config,
                show_progress=False,
            ).to_pandas()
            for m in missing:
                row["scores"][m] = _clean(float(scored[m].iloc[0]))

            failed = [m for m in missing if row["scores"][m] is None]
            if failed:
                logger.warning(f"    judge failed on {failed} — re-run to retry")
        else:
            logger.info("    scores loaded from cache")

        record["scores"] = row["scores"]
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    answered = [r for r in rows if r["guardrail_action"] == "passed"]
    summary = {
        metric: {
            "mean": _mean([r["scores"].get(metric) for r in answered]),
            "n_scored": sum(1 for r in answered if r["scores"].get(metric) is not None),
        }
        for metric in GENERATION_METRICS
    }
    summary["answered"] = len(answered)
    summary["declined_by_guardrail"] = len(rows) - len(answered)
    summary["latency"] = {
        "llm": _latency_summary(
            [r["timings"]["llm_ms"] for r in answered if "llm_ms" in r["timings"]]
        ),
        "end_to_end": _latency_summary([r["timings"]["total_ms"] for r in answered]),
    }
    return {"summary": summary, "per_question": rows}


# ─────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────


def _bar(value) -> str:
    if value is None:
        return " " * 20 + "   n/a"
    filled = int(round(value * 20))
    return "█" * filled + "░" * (20 - filled) + f" {value:.3f}"


def print_report(report: dict):
    print("\n" + "=" * 64)
    print("RAG EVALUATION")
    print("=" * 64)
    cfg = report["config"]
    print(
        f"Corpus: {cfg['index_chunks']} chunks | top_k={cfg['top_k']} | "
        f"generator={cfg['generator_model']}"
    )

    stage1 = report.get("retrieval_and_guardrails", {}).get("summary")
    if stage1:
        print("\nRetrieval")
        for key, value in stage1["retrieval"].items():
            if key != "n":
                print(f"  {key:38s} {_bar(value)}")
        print("\nGuardrails")
        for key, value in stage1["guardrails"].items():
            if key != "thresholds":
                print(f"  {key:38s} {_bar(value)}")
        print(f"  thresholds: {stage1['guardrails']['thresholds']}")
        print("\nLatency (local)")
        for stage, lat in stage1["latency"].items():
            if lat:
                print(
                    f"  {stage:14s} p50 {lat['p50_ms']:8.1f} ms   p95 {lat['p95_ms']:8.1f} ms"
                )

    sweep = report.get("threshold_sweep")
    if sweep:
        print("\nThreshold sweep (top 5 by balanced accuracy)")
        for r in sweep[:5]:
            print(
                f"  scope>={r['scope_threshold']:.2f} conf>={r['confidence_threshold']:.2f}"
                f"  pass {r['answerable_pass_rate']:.2f}  decline "
                f"{r['negative_decline_rate']:.2f}  bal.acc {r['balanced_accuracy']:.3f}"
            )

    stage2 = report.get("generation", {}).get("summary")
    if stage2:
        print(f"\nGeneration (RAGAS, judge={cfg.get('judge_model')})")
        for key, label in GENERATION_METRICS.items():
            m = stage2[key]
            print(f"  {label:38s} {_bar(m['mean'])}  (n={m['n_scored']})")
        print(
            f"  answered {stage2['answered']}, "
            f"declined by guardrail {stage2['declined_by_guardrail']}"
        )
        for stage, lat in stage2["latency"].items():
            if lat:
                print(
                    f"  latency {stage:10s} p50 {lat['p50_ms']:8.1f} ms   p95 {lat['p95_ms']:8.1f} ms"
                )

    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--output", default="data/eval_results.json")
    parser.add_argument(
        "--stage",
        choices=["retrieval", "generation", "all"],
        default="all",
        help="retrieval = offline stage 1 only; generation = RAGAS stage 2 only",
    )
    parser.add_argument("--sweep", action="store_true", help="grid-search thresholds")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--persist-dir",
        default="./data/chroma_db",
        help="ChromaDB directory to evaluate (e.g. an older index to compare)",
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    from src.indexing import vector_store
    from src.rag import guardrails as g
    from src.rag.pipeline import DEFAULT_TOP_K, RAGPipeline

    dataset = load_eval_dataset(args.dataset)
    needs_llm = args.stage in ("generation", "all")
    pipeline = RAGPipeline(
        top_k=args.top_k or DEFAULT_TOP_K,
        with_llm=needs_llm,
        persist_dir=args.persist_dir,
    )

    from importlib.metadata import PackageNotFoundError, version

    import chromadb

    try:
        ragas_version = version("ragas")
    except PackageNotFoundError:  # stage 1 runs without ragas installed
        ragas_version = None

    index_chunks = (
        chromadb.PersistentClient(path=args.persist_dir)
        .get_collection(vector_store.DEFAULT_COLLECTION)
        .count()
    )

    report = {
        "config": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "dataset_size": len(dataset),
            "index_dir": args.persist_dir,
            "index_chunks": index_chunks,
            "top_k": pipeline.top_k,
            "generator_model": pipeline.model,
            "judge_model": args.judge_model if needs_llm else None,
            "ragas_version": ragas_version,
        }
    }

    if args.stage in ("retrieval", "all"):
        logger.info("STAGE 1: retrieval + guardrails (offline)")
        report["retrieval_and_guardrails"] = evaluate_retrieval(pipeline, dataset)
        if args.sweep:
            report["threshold_sweep"] = sweep_thresholds(
                report["retrieval_and_guardrails"]["per_question"],
                g.SOURCE_MIN_SCORE,
            )

    if needs_llm:
        logger.info("STAGE 2: generation quality (RAGAS)")
        t0 = time.perf_counter()
        report["generation"] = evaluate_generation(
            pipeline, dataset, args.judge_model, args.cache_dir
        )
        logger.info(f"Stage 2 finished in {time.perf_counter() - t0:.0f}s")

    print_report(report)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
