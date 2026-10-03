"""
Measure end-to-end query latency of the RAG pipeline.

Runs each question sequentially through RAGPipeline.query() and records:
  - total:              scope check + retrieval + guardrails + LLM generation
  - retrieval+guardrails: everything except the LLM call (total - llm)
  - llm:                the Groq generation call on its own

One warm-up query runs first and is excluded, so model loading and the first
network connection don't skew the numbers. Questions run one at a time.

Groq's free tier allows 8,000 tokens per minute for gpt-oss-20b, and one RAG
query uses a large share of that, so back-to-back queries get HTTP 429 and the
client silently waits and retries. That wait is rate limiting, not pipeline
latency. The script therefore counts 429 retries per query and computes the
latency summaries over retry-free queries only; --pause spaces queries out so
they stay under the limit.

Percentiles use the nearest-rank method (p95 of 24 values = 23rd smallest).

Usage:
  python scripts/measure_latency.py --pause 25
  python scripts/measure_latency.py --pause 0 --output data/latency_back_to_back.json
"""

import argparse
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def percentile(values, pct):
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarise(values):
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
    }


def hardware_info():
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_gb"] = None
    return info


class RetryCounter(logging.Handler):
    """Counts the openai client's 'Retrying request' log lines (429s etc.)."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.count = 0

    def emit(self, record):
        if "Retrying request" in record.getMessage():
            self.count += 1


def main():
    parser = argparse.ArgumentParser(description="Measure RAG pipeline latency")
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--output", default="data/latency_results.json")
    parser.add_argument(
        "--pause",
        type=float,
        default=25.0,
        help="seconds to wait between queries (keeps under Groq's per-minute limit)",
    )
    args = parser.parse_args()

    from src.rag.pipeline import RAGPipeline

    with open(args.dataset, encoding="utf-8") as f:
        questions = [
            e["question"] for e in json.load(f) if e["category"] == "answerable"
        ]

    retries = RetryCounter()
    client_logger = logging.getLogger("openai._base_client")
    client_logger.setLevel(logging.INFO)
    client_logger.addHandler(retries)

    pipeline = RAGPipeline()

    print("Warm-up query (excluded)...")
    pipeline.query(questions[0])
    time.sleep(args.pause)

    rows = []
    for i, question in enumerate(questions, 1):
        retries.count = 0
        result = pipeline.query(question)
        t = result["timings"]
        row = {
            "question": question,
            "guardrail_action": result["guardrail_action"],
            "retries": retries.count,
            "total_ms": round(t["total_ms"], 1),
            "llm_ms": round(t.get("llm_ms", 0.0), 1),
            "retrieval_and_guardrails_ms": round(
                t["total_ms"] - t.get("llm_ms", 0.0), 1
            ),
        }
        rows.append(row)
        print(
            f"[{i:2d}/{len(questions)}] {row['guardrail_action']:14s} "
            f"retries {row['retries']}  total {row['total_ms']:7.1f} ms  "
            f"llm {row['llm_ms']:7.1f} ms  "
            f"retrieval+guardrails {row['retrieval_and_guardrails_ms']:6.1f} ms"
        )
        if i < len(questions):
            time.sleep(args.pause)

    answered = [r for r in rows if r["guardrail_action"] == "passed"]
    clean = [r for r in answered if r["retries"] == 0]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "generator_model": pipeline.model,
        "top_k": pipeline.top_k,
        "pause_between_queries_s": args.pause,
        "hardware": hardware_info(),
        "queries": len(rows),
        "answered_by_llm": len(answered),
        "queries_with_retries": len(answered) - len(clean),
        "end_to_end_retry_free": (
            summarise([r["total_ms"] for r in clean]) if clean else None
        ),
        "llm_only_retry_free": (
            summarise([r["llm_ms"] for r in clean]) if clean else None
        ),
        "end_to_end_all": summarise([r["total_ms"] for r in answered]),
        "retrieval_and_guardrails": summarise(
            [r["retrieval_and_guardrails_ms"] for r in rows]
        ),
        "per_query": rows,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"\nAnswered by LLM: {len(answered)}/{len(rows)}, "
        f"with 429 retries: {report['queries_with_retries']}"
    )
    for key in (
        "end_to_end_retry_free",
        "llm_only_retry_free",
        "end_to_end_all",
        "retrieval_and_guardrails",
    ):
        s = report[key]
        if s:
            print(
                f"{key:26s} median {s['median_ms']:8.1f} ms   "
                f"p95 {s['p95_ms']:8.1f} ms   (n={s['n']})"
            )
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
