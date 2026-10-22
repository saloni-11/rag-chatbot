"""
scripts/eval_fresh.py — one-shot held-out measurement: baseline vs hybrid
=========================================================================
Measures the locked hybrid config against the dense baseline on the frozen
fresh held-out set (tests/eval_heldout_fresh.json), with the same RAGAS
context precision / recall judge as the tuning (scripts/tune_hybrid.py).

Retrieval is non-deterministic (HNSW), so --collect is run once per
process (each run is a fresh process) and --report scores every run and
shows per-metric ranges. Judge verdicts are cached per (question, chunk)
for precision and per (question, contexts) for recall, so a chunk or
context list seen in several runs, or by both pipelines, is judged once.

USAGE:
  python scripts/eval_fresh.py --collect --run 1     # repeat for runs 2, 3
  python scripts/eval_fresh.py --report
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tune_hybrid import Budget, Judge, _key, _load, average_precision  # noqa: E402

from src.evaluation import ragas_eval  # noqa: E402
from src.evaluation.ragas_eval import DEFAULT_JUDGE_MODEL, logger  # noqa: E402

DATASET = "tests/eval_heldout_fresh.json"
FROZEN_SHA256 = "9c23efed97020baf1975252ac5ef99fc02bd162897cecb62eb6dab2a07fa3158"
OUT_DIR = Path("data/eval_cache/fresh")


def check_freeze() -> str:
    digest = hashlib.sha256(Path(DATASET).read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        sys.exit(f"{DATASET} sha256 {digest} does not match the frozen {FROZEN_SHA256}")
    return digest


def collect(run: int):
    from src.rag.pipeline import RAGPipeline

    dataset = ragas_eval.load_eval_dataset(DATASET)
    # Locked config = the code defaults (15 per retriever, k=60, pool 10, top-3)
    pipeline = RAGPipeline(with_llm=False, retrieval_mode="hybrid")
    config = pipeline.retrieval_config()
    rows, chunks = {}, {}

    def ids(nodes):
        for n in nodes:
            chunks[n.node.node_id] = {
                "file_name": n.node.metadata.get("file_name"),
                "text": n.node.get_content(),
            }
        return [n.node.node_id for n in nodes]

    for entry in dataset:
        q = entry["question"]
        in_scope, _ = pipeline.guardrails.check_scope(q)

        # Baseline: the pre-hybrid pipeline (dense top-3, check_confidence)
        if in_scope:
            dense = pipeline.retriever.retrieve(q)
            passed, kept, _ = pipeline.guardrails.check_confidence(dense)
            baseline = {
                "action": "passed" if passed else "low_confidence",
                "retrieved": ids(dense),
                "contexts": ids(kept) if passed else [],
            }
        else:
            baseline = {"action": "scope_rejected", "retrieved": [], "contexts": []}

        stage = pipeline.retrieve(q)
        hybrid = {
            "action": stage["guardrail_action"],
            "retrieved": ids(stage["retrieved_nodes"]),
            "contexts": ids(stage["nodes"]) if stage["guardrail_action"] == "passed" else [],
            "displaced": stage["displaced"],
        }
        rows[entry["id"]] = {"baseline": baseline, "hybrid": hybrid}
        logger.info(f"  run {run} {entry['id']}: baseline {baseline['action']}, hybrid {hybrid['action']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"run{run}.json"
    path.write_text(
        json.dumps({"run": run, "config": config, "rows": rows, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved {path}")


def report(judge_model: str, max_calls: int, output: str):
    dataset = ragas_eval.load_eval_dataset(DATASET)
    by_id = {e["id"]: e for e in dataset}
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_DIR.glob("run*.json"))]
    chunks = {}
    for r in runs:
        chunks.update(r["chunks"])
    answerable = [e["id"] for e in dataset if e["category"] == "answerable"]

    tag = judge_model.replace("/", "_")
    p_path, r_path = OUT_DIR / f"precision_{tag}.jsonl", OUT_DIR / f"recall_{tag}.jsonl"
    p_cache, r_cache = _load(p_path), _load(r_path)

    def p_key(q, c):
        e = by_id[q]
        return _key(judge_model, e["question"], e["ground_truth"], chunks[c]["text"])

    def r_key(q, ctx):
        e = by_id[q]
        return _key(judge_model, e["question"], e["ground_truth"], [chunks[c]["text"] for c in ctx])

    pairs = list(dict.fromkeys(
        (q, c) for r in runs for q in answerable for pipe in ("baseline", "hybrid")
        for c in r["rows"][q][pipe]["contexts"]
    ))
    lists = list(dict.fromkeys(
        (q, tuple(r["rows"][q][pipe]["contexts"])) for r in runs for q in answerable
        for pipe in ("baseline", "hybrid") if r["rows"][q][pipe]["contexts"]
    ))
    todo_p = [p for p in pairs if p_key(*p) not in p_cache]
    todo_r = [l for l in lists if r_key(l[0], list(l[1])) not in r_cache]
    logger.info(f"{len(runs)} runs | precision {len(todo_p)}/{len(pairs)} to judge | recall {len(todo_r)}/{len(lists)} to judge")

    if todo_p or todo_r:
        judge = Judge(judge_model)
        try:
            for q, c in todo_p:
                if judge.calls >= max_calls:
                    raise Budget("--max-calls reached")
                v = judge.precision_verdict(by_id[q], chunks[c]["text"])
                if math.isnan(v):
                    continue
                rec = {"key": p_key(q, c), "qid": q, "chunk": c, "verdict": round(v)}
                p_cache[rec["key"]] = rec
                with open(p_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            for q, ctx in todo_r:
                if judge.calls >= max_calls:
                    raise Budget("--max-calls reached")
                s = judge.recall_score(by_id[q], [chunks[c]["text"] for c in ctx])
                if math.isnan(s):
                    continue
                rec = {"key": r_key(q, list(ctx)), "qid": q, "contexts": list(ctx), "score": s}
                r_cache[rec["key"]] = rec
                with open(r_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        except Budget as e:
            logger.warning(f"Stopped judging: {e}")
        logger.info(f"Judge usage this run: {judge.calls} calls, {judge.tokens} tokens")

    def score(q, ctx):
        if not ctx:
            return None, None
        v = [p_cache.get(p_key(q, c), {}).get("verdict") for c in ctx]
        cp = average_precision(v) if None not in v else None
        r = r_cache.get(r_key(q, ctx))
        return cp, (r["score"] if r else None)

    per_q, per_run = {}, []
    for r in runs:
        run_means = {}
        for pipe in ("baseline", "hybrid"):
            cps, crs = [], []
            for q in answerable:
                row = r["rows"][q][pipe]
                cp, cr = score(q, row["contexts"])
                per_q.setdefault(q, {}).setdefault(pipe, []).append(
                    {"run": r["run"], "action": row["action"], "cp": cp, "cr": cr,
                     "contexts": row["contexts"], "displaced": row.get("displaced", 0)}
                )
                if cp is not None:
                    cps.append(cp)
                if cr is not None:
                    crs.append(cr)
            run_means[pipe] = {
                "cp": round(statistics.mean(cps), 4) if cps else None, "cp_n": len(cps),
                "cr": round(statistics.mean(crs), 4) if crs else None, "cr_n": len(crs),
            }
        per_run.append({"run": r["run"], **run_means})

    negatives = {
        e["id"]: {pipe: [r["rows"][e["id"]][pipe]["action"] for r in runs] for pipe in ("baseline", "hybrid")}
        for e in dataset if e["category"] != "answerable"
    }

    def rng(pipe, m):
        vals = [pr[pipe][m] for pr in per_run if pr[pipe][m] is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    print(f"\nFresh held-out set, {len(runs)} runs, answerable n={len(answerable)}")
    for pipe in ("baseline", "hybrid"):
        cp, cr = rng(pipe, "cp"), rng(pipe, "cr")
        print(f"  {pipe:8s} precision {cp[0]:.3f}-{cp[1]:.3f}   recall {cr[0]:.3f}-{cr[1]:.3f}")
    for pr in per_run:
        print(f"  run {pr['run']}: " + "  ".join(
            f"{p} cp {pr[p]['cp']} (n={pr[p]['cp_n']}) cr {pr[p]['cr']} (n={pr[p]['cr_n']})" for p in ("baseline", "hybrid")))
    print("\nPer question (values per run)")
    for q in answerable:
        for pipe in ("baseline", "hybrid"):
            rs = per_q[q][pipe]
            fmt = lambda x: "-" if x is None else f"{x:.2f}"
            print(f"  {q:15s} {pipe:8s} cp [{', '.join(fmt(x['cp']) for x in rs)}]  cr [{', '.join(fmt(x['cr']) for x in rs)}]"
                  f"  same contexts all runs: {len({tuple(x['contexts']) for x in rs}) == 1}"
                  + (f"  displaced {[x['displaced'] for x in rs]}" if pipe == "hybrid" else ""))
    print("\nNegatives (action per run)")
    for q, acts in negatives.items():
        print(f"  {q:15s} {by_id[q]['category']:13s} baseline {acts['baseline']}  hybrid {acts['hybrid']}")

    Path(output).write_text(json.dumps({
        "dataset": DATASET, "sha256": FROZEN_SHA256, "judge_model": judge_model,
        "config": runs[0]["config"], "per_run": per_run, "per_question": per_q, "negatives": negatives,
    }, indent=1), encoding="utf-8")
    logger.info(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--run", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-calls", type=int, default=150)
    parser.add_argument("--output", default="data/eval_fresh_hybrid.json")
    args = parser.parse_args()

    logger.info(f"Freeze check: sha256 {check_freeze()} matches")
    if args.collect:
        collect(args.run)
    if args.report:
        report(args.judge_model, args.max_calls, args.output)
    logger.info(f"Freeze check: sha256 {check_freeze()} matches")


if __name__ == "__main__":
    main()
