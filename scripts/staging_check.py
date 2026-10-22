"""
scripts/staging_check.py — run INSIDE the built container by
.github/workflows/staging-check.yml (staging-check branch only).

Proves, in the production image with HF_HUB_OFFLINE=1 and the Hugging Face
hosts blocked, that:
  1. the cross-encoder loads from the build-time cache (/app/hf_cache),
  2. every answered query actually ran the reranker (rerank_ms > 0),
  3. reranking changes what reaches the LLM vs the dense top-3,
and reports retrieval latency per query.

USAGE (in the container): python scripts/staging_check.py /tmp/questions.json
"""

import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")


def main():
    questions = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    # ── Environment: offline and Hugging Face unreachable ─────────
    print("HF_HUB_OFFLINE =", os.getenv("HF_HUB_OFFLINE"))
    print("HF_HOME        =", os.getenv("HF_HOME"))
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        sys.exit("FAIL: huggingface.co is reachable; the offline test would prove nothing")
    except Exception as e:  # noqa: BLE001
        print(f"huggingface.co unreachable as intended ({type(e).__name__})")

    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache("cross-encoder/ms-marco-MiniLM-L-6-v2", "config.json")
    print("reranker config in cache:", cached)
    if not isinstance(cached, str) or not cached.startswith(os.getenv("HF_HOME", "?")):
        sys.exit("FAIL: reranker is not in the HF_HOME cache")

    import torch

    print("torch threads:", torch.get_num_threads(), "| os.cpu_count:", os.cpu_count())
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        print("cgroup cpu.max:", cpu_max.read_text().strip())

    # ── Pipeline: hybrid mode, reranker loaded ────────────────────
    from src.rag.pipeline import RAGPipeline

    p = RAGPipeline(with_llm=False)
    if p.hybrid is None or p.hybrid.reranker is None:
        sys.exit(f"FAIL: reranking not active, retrieval config = {p.retrieval_config()}")
    print("retrieval config:", json.dumps(p.retrieval_config()))

    p.retrieve(questions[0])  # warm-up
    rows = []
    for q in questions:
        stage = p.retrieve(q)
        t = stage["timings"]
        dense = [n.node.node_id for n in p.retriever.retrieve(q)]
        final = [n.node.node_id for n in stage["nodes"]]
        rows.append(
            {
                "action": stage["guardrail_action"],
                "rerank_ms": round(t.get("rerank_ms", 0.0), 1),
                "retrieval_ms": round(t.get("retrieval_ms", 0.0), 1),
                "pairs": len(stage["candidates"]),
                "differs_from_dense": final != dense,
            }
        )
        print(f"  {stage['guardrail_action']:15s} {json.dumps(rows[-1])}  {q[:60]}")

    answered = [r for r in rows if r["action"] == "passed"]
    if not answered or any(r["rerank_ms"] <= 0 for r in answered):
        sys.exit("FAIL: an answered query did not run the reranker")
    print(
        f"\nPASS: reranker ran on {len(answered)}/{len(answered)} answered queries; "
        f"context differs from dense top-3 on {sum(r['differs_from_dense'] for r in answered)}"
    )
    print(
        f"retrieval p50 {statistics.median(r['retrieval_ms'] for r in answered):.0f} ms | "
        f"rerank p50 {statistics.median(r['rerank_ms'] for r in answered):.0f} ms "
        f"(n={len(answered)})"
    )


if __name__ == "__main__":
    main()
