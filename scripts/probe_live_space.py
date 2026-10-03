"""Probe the live Space: compare its guardrail decisions and chunk scores
with local offline retrieval for the tuning-set questions."""

import json
import sys
import time

import requests

sys.path.insert(0, ".")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")

from src.rag.pipeline import RAGPipeline  # noqa: E402

URL = "https://salonisamant01-study-companion.hf.space/api/query"
dataset = json.load(open("tests/eval_dataset.json", encoding="utf-8"))
pipe = RAGPipeline(with_llm=False)

rows = []
for e in dataset:
    q = e["question"]
    scope, _ = pipe.guardrails.scope_score(q)
    if scope < 0.2:  # scope layer rejects locally and remotely; no confidence info
        continue
    local_nodes = pipe.retriever.retrieve(q)
    local_scores = sorted((round(n.score, 4) for n in local_nodes), reverse=True)

    for attempt in range(4):
        r = requests.post(URL, json={"question": q}, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(30)
    data = r.json() if r.status_code == 200 else {}
    remote_scores = sorted(
        (s["score"] for s in data.get("sources", []) if s.get("score") is not None),
        reverse=True,
    )
    row = {
        "id": e["id"],
        "http": r.status_code,
        "remote_action": data.get("guardrail_action"),
        "local_best": local_scores[0] if local_scores else None,
        "local_scores": local_scores,
        "remote_scores": remote_scores,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(25 if row["remote_action"] == "passed" else 3)

json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
