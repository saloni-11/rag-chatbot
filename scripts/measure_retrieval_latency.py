"""
scripts/measure_retrieval_latency.py — what hybrid retrieval + reranking costs
==============================================================================
Times pipeline.retrieve() (scope check → retrieval → guardrails, no LLM) on
the tuning-set questions, for the dense baseline and hybrid configs, and
reports the added per-query latency.

The cross-encoder's cost grows with the number of chunks it reranks, and
with fewer CPU cores. HuggingFace's free CPU tier has 2 vCPUs, so the
default run limits torch to 2 threads to approximate it (it is still a
different CPU, so treat the result as an estimate, not a measurement of
the Space).

USAGE:
  python scripts/measure_retrieval_latency.py               # 2 threads
  python scripts/measure_retrieval_latency.py --threads 0   # torch default
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ragas_eval import _percentile, load_eval_dataset, logger  # noqa: E402

# (label, retrieval_mode, hybrid_options)
CONFIGS = [
    ("dense (current)", "dense", {}),
    ("hybrid n15, rerank pool 10", "hybrid", {"candidates_per_retriever": 15, "rerank_pool": 10}),
    ("hybrid n15, rerank pool 15", "hybrid", {"candidates_per_retriever": 15, "rerank_pool": 15}),
    ("hybrid n15, rerank all", "hybrid", {"candidates_per_retriever": 15, "rerank_pool": 0}),
    ("hybrid n20, rerank pool 15", "hybrid", {"candidates_per_retriever": 20, "rerank_pool": 15}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--threads", type=int, default=2, help="0 = torch default")
    parser.add_argument("--output", default="data/latency_retrieval.json")
    args = parser.parse_args()
    if "heldout" in Path(args.dataset).name:
        sys.exit("Latency is measured on the tuning set only.")

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    from src.rag.pipeline import RAGPipeline

    questions = [e["question"] for e in load_eval_dataset(args.dataset)]
    results = {}
    for label, mode, options in CONFIGS:
        pipeline = RAGPipeline(with_llm=False, retrieval_mode=mode, hybrid_options=options)
        pipeline.retrieve(questions[0])  # warm-up: first inference is slower
        rows = []
        for q in questions:
            stage = pipeline.retrieve(q)
            rows.append(
                {
                    "action": stage["guardrail_action"],
                    "pairs": len(stage["candidates"]),
                    **{k: round(v, 1) for k, v in stage["timings"].items()},
                }
            )
        # Scope-rejected questions never reach retrieval; time the rest
        timed = [r for r in rows if "retrieval_ms" in r]
        summary = {
            "n": len(timed),
            "retrieval_p50_ms": round(statistics.median(r["retrieval_ms"] for r in timed), 1),
            "retrieval_p95_ms": round(_percentile([r["retrieval_ms"] for r in timed], 95), 1),
        }
        if mode == "hybrid":
            summary["rerank_p50_ms"] = round(statistics.median(r["rerank_ms"] for r in timed), 1)
            summary["mean_pairs"] = round(statistics.mean(r["pairs"] for r in timed), 1)
        results[label] = {"config": {"mode": mode, **options}, "summary": summary, "per_question": rows}
        logger.info(f"{label}: {summary}")

    base = results[CONFIGS[0][0]]["summary"]
    print(f"\nRetrieval latency, {len(questions)} tuning questions, torch threads={threads}")
    print(f"{'config':30s} {'pairs':>5s} {'p50 ms':>8s} {'p95 ms':>8s} {'+p50 vs dense':>14s}")
    for label, r in results.items():
        s = r["summary"]
        print(
            f"{label:30s} {s.get('mean_pairs', 0):5.1f} {s['retrieval_p50_ms']:8.1f} "
            f"{s['retrieval_p95_ms']:8.1f} {s['retrieval_p50_ms'] - base['retrieval_p50_ms']:+14.1f}"
        )

    out = Path(args.output)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing[f"threads_{threads}"] = results
    out.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
