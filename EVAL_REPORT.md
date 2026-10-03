# Evaluation Report: AI/ML Study Companion

**Date:** 2026-10-01 (Tasks 1, 2 and 4: measurements taken 10:07–10:38 UTC); 2026-10-03 (Task 3 run started 01:35 UTC; the leaked-question check ran shortly after)
**Branch:** `eval-verification`, based on `main` at `d657a4d` (the commit deployed to the HuggingFace Space)
**Held-out set:** frozen in commit `31c4819` before any evaluation was run against it.
SHA-256 of the committed file: `dc8b3f0db31a8903bb84a2994a42eef054b25929f10ad4917ca867066613987b`
(verify with `git show 31c4819:tests/eval_heldout.json | sha256sum`; a Windows checkout may change line endings and therefore the on-disk hash)

Every number below was measured during this session and can be reproduced with the commands given. Anything that could not be measured is marked as such.

## Summary

| Item | Result |
|---|---|
| Generator / judge models | `openai/gpt-oss-20b` / `openai/gpt-oss-120b` (Groq) |
| Corpus | 9 PDFs → 562 chunks (local index), 512-token chunks, 50-token overlap |
| Tests | 43 passed / 0 failed (local and CI on `d657a4d`) |
| Held-out: answerable questions wrongly rejected | **0/12** |
| Held-out: unanswerable questions correctly refused | **2/4** |
| Held-out: out-of-scope questions rejected | **4/4** |
| Held-out: retrieval Hit@3 (document level) | **12/12** |
| Held-out RAGAS faithfulness | **0.791** (n=12/12) |
| Held-out RAGAS answer relevancy | **0.885** (n=12/12) |
| Held-out RAGAS context precision | **0.361** (n=12/12) |
| Held-out RAGAS context recall | **0.458** (n=12/12) |
| End-to-end latency, paced (headline) | **median 1,092.7 ms, p95 1,602.2 ms** (n=24, 0 rate-limit retries) |
| End-to-end latency, back-to-back | median 14,046.0 ms, p95 18,724.9 ms (n=24, 20/24 queries retried after HTTP 429) |

## Environment

| | |
|---|---|
| Machine | Intel Core Ultra 5 125H, 18 logical CPUs, 15.5 GB RAM, Windows 11 (10.0.26200) |
| Python | 3.12.12 (conda env `ragbot-env`) |
| Libraries | llama-index-core 0.10.68.post1, llama-index-llms-groq 0.1.4, chromadb 0.5.5, sentence-transformers 3.0.1, pypdf 4.3.1, ragas 0.2.15, langchain-groq 1.1.3 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| LLM provider | Groq free tier (`on_demand` service tier) |

## Task 1: Resume facts audit

| Claim | Verdict | Evidence |
|---|---|---|
| Generator is Groq GPT-OSS 20B | Confirmed | `DEFAULT_MODEL = "openai/gpt-oss-20b"` in `src/rag/pipeline.py:53`. The API builds the pipeline with defaults (`src/api/main.py:49`), and the live Space's `/api/health` returned `"model":"openai/gpt-oss-20b"`. |
| Judge is a 120B model | Confirmed | `DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")` in `src/evaluation/ragas_eval.py:78`. `EVAL_JUDGE_MODEL` is not set. |
| 9 PDFs | Confirmed | 9 `.pdf` files in `data/raw/`, all tracked in git. |
| 562 chunks | Confirmed for the **local** index | The local ChromaDB collection `rag_chatbot` holds 562 chunks from 9 distinct files. The deployed index is rebuilt inside Docker and is **not identical** (see "Production configuration findings"). Its chunk count could not be read. |
| Chunk size 512 | Confirmed | `DocumentChunker(chunk_size=512, chunk_overlap=50)` in `scripts/ingest_data.py:66` → LlamaIndex `SentenceSplitter` (sentence-aware, 512 tokens, 50 overlap). |
| 43 tests | Confirmed, with a caveat | Local `python -m pytest`: 43 passed, 0 failed. CI run 36835994192 on `d657a4d`: 43 passed, 67% coverage. **Caveat:** the 2 root-endpoint tests in `tests/test_api.py` fail if `frontend/dist/` exists locally, because the app then serves the React build at `/` instead of JSON. They pass in CI, where no build exists. |
| CI pipeline | Confirmed | `.github/workflows/ci.yml` runs on push or PR to `main` with three sequential jobs: **Lint** (black 24.8.0 `--check`, isort, flake8 `--max-line-length 100 --ignore E501,W503,E203`) → **Test** (pytest with coverage) → **Docker Build**. |
| Deployment trigger | Confirmed | `.github/workflows/deploy.yml` runs on `workflow_run` of "CI Pipeline", `types: [completed]`, `branches: [main]`, and only `if` CI succeeded. It force-pushes to the Space and sets the `GROQ_API_KEY` Space secret. Branches other than `main` never deploy. |

### Guardrail thresholds

Defined in `src/rag/guardrails.py:66-68`. Each reads an environment variable and falls back to the code default.

| Threshold | Code default | Local (`.env`) | Live Space |
|---|---|---|---|
| `SCOPE_THRESHOLD` | 0.2 | 0.2 | not set as a variable or secret → **0.2** |
| `CONFIDENCE_THRESHOLD` | 0.5 | 0.5 | Space secret, value unreadable. On 2026-10-01 it behaved as below 0.5; the owner set it to **0.5** on 2026-10-03 (see below) |
| `SOURCE_MIN_SCORE` | 0.35 | **0.3** | not set as a variable or secret → **0.35** |

**Documented mismatch:** the local `.env` sets `SOURCE_MIN_SCORE=0.3`, while the code default (and therefore the live Space) uses 0.35. Earlier evaluations in this repo ran at 0.3. The held-out evaluation in this report ran at **0.35**, set on the command line only.

**Dead configuration:** `.env` also sets `GROQ_MODEL`, `TOP_K_RETRIEVAL`, `CHUNK_SIZE` and `CHUNK_OVERLAP`, but no code reads them. The effective values are hardcoded: model `openai/gpt-oss-20b`, `top_k = 3` (`src/rag/pipeline.py:54`), chunk size 512 and overlap 50.

## Production configuration findings

These came out of checking which thresholds the live Space uses (`scripts/probe_live_space.py`, raw output in `data/live_space_probe.json`).

1. **Space configuration.** The HF API (authenticated with the Space owner's token) reported **no Space variables** and two secrets, `GROQ_API_KEY` and `CONFIDENCE_THRESHOLD`. Secret values cannot be read.
2. **The live confidence threshold is below 0.5.** The 27 tuning-set questions that pass the scope check locally (scope similarity ≥ 0.2) were sent to the live `/api/query`:
   - 23 returned `passed` (HTTP 200). None returned `low_confidence`.
   - 4 returned HTTP 500 with Groq error `429 ... tokens per day (TPD): Limit 200000` for `openai/gpt-oss-20b`. The LLM is only called after both the scope and confidence checks pass, so **these 4 questions passed both live guardrails**. They include one out-of-scope question (`oos-04`, "What is a good exercise routine for building muscle?") and two unanswerable ones (`unans-02`, `unans-03`). Locally, at confidence 0.5, all three are refused; their local best-chunk scores are 0.454, 0.463 and 0.472.
   - The exact live value cannot be determined: the secret is unreadable, and the 500 responses carry no chunk scores.
3. **The live index differs from the local index.** For every question that returned scores, the live chunk scores differ from local ones (for example `attn-01` best chunk 0.5522 live vs 0.5273 local, and `rag-01` 0.7989 vs 0.6885). The Docker image rebuilds the index during the build, evidently producing different chunks; the cause was not investigated. **The retrieval and guardrail numbers in this report describe the local index**, and production behaviour can differ.
   - **Follow-up, 2026-10-03:** the Space owner set the `CONFIDENCE_THRESHOLD` secret to 0.5 and restarted the Space. A live smoke test (raw output in `data/live_smoke_test.json`) then gave:
     - an answerable question ("How does BERT's masked language modeling choose and replace tokens?"): `passed`, best live chunk score 0.7064, 2.5 s
     - out-of-scope `oos-04` ("What is a good exercise routine for building muscle?"): `low_confidence`, best live chunk score 0.4409, 0.9 s, refused without an LLM call

     `oos-04` had passed the live guardrails before the change. It is now refused by the confidence layer, as in the local evaluation, where it scored 0.454 locally. This is consistent with a live threshold of 0.5, though two questions cannot pin down the exact value.
4. **Evaluation and production share one Groq quota.** The 429s above came from the 200,000 tokens-per-day limit on `gpt-oss-20b`. It is shared by the whole Groq organisation, and this session's latency runs and live probes used it up. While it is exhausted, **the live app returns errors for every question that reaches the LLM.**

## Task 2: Held-out guardrail evaluation

**Set:** `tests/eval_heldout.json`, 20 new questions: 12 answerable (covering all 9 papers), 4 unanswerable (on-topic but absent from the corpus), and 4 out-of-scope (two of them tech-flavoured). Each answerable ground truth was checked against the indexed text. Each unanswerable topic (SVMs/kernel trick, random forests, DBSCAN, bagging/boosting) has 0 matching chunks out of 562. No question repeats a tuning-set question or its answer. One incidental shared word: the `ho-llmsec-01` answer contains "human-like", which also appears in tuning answer `llmsec-02` in an unrelated sense.

**Frozen thresholds:** scope 0.2, confidence 0.5, source minimum 0.35. No sweep was run and nothing was retuned after seeing these results.

```bash
SOURCE_MIN_SCORE=0.35 python src/evaluation/ragas_eval.py --stage retrieval \
  --dataset tests/eval_heldout.json --output data/eval_heldout_stage1.json
```

| Metric | Held-out (this report) | Tuning set (in-sample, for comparison) |
|---|---|---|
| Answerable questions wrongly rejected | **0/12** (0%) | 0/24 |
| Unanswerable questions correctly refused | **2/4** (50%) | 3/3 |
| Out-of-scope questions rejected | **4/4** (100%), all by the scope layer | 5/5 |
| All negatives declined | **6/8** (75%) | 8/8 |
| Retrieval Hit@3 (right paper in top 3) | **12/12** | 23/24 |
| Retrieval Hit@1 | 12/12 | 23/24 |

**The 100% correct-refusal result from the tuning set does not hold on held-out data.** Two unanswerable questions passed the confidence check and would be sent to the LLM:

| Question | Best chunk score | Threshold | Outcome |
|---|---|---|---|
| ho-unans-01: How does the kernel trick work in support vector machines? | 0.512 | 0.5 | passed (should refuse) |
| ho-unans-04: What is the difference between bagging and boosting in ensemble learning? | 0.510 | 0.5 | passed (should refuse) |
| ho-unans-02: How does a random forest compute feature importance? | 0.470 | 0.5 | refused ✓ |
| ho-unans-03: How does the DBSCAN clustering algorithm decide which points are noise? | 0.432 | 0.5 | refused ✓ |

On this set, answerable questions' best scores ranged from 0.567 to 0.776.

**What the LLM did with the two leaked questions** (measured 2026-10-03, same thresholds, raw output in `data/heldout_leaked_unanswerable.json`): it declined both. It replied that the provided context does not contain information about the kernel trick, or about bagging and boosting. The grounding prompt acted as a second line of defence, so no hallucinated answer was produced, but each leak still cost an LLM call.

Hit@3 is document-level: it counts a hit when any of the top 3 chunks comes from the right paper. It does not check that the chunk contains the answer.

## Task 3: RAGAS on held-out answerable questions

This was first blocked on 2026-10-01, when the generator's daily Groq quota was exhausted (see Production configuration findings, item 4). It was run on **2026-10-03 (01:35 UTC)**, after the quota recovered:

```bash
SOURCE_MIN_SCORE=0.35 python src/evaluation/ragas_eval.py --stage generation \
  --dataset tests/eval_heldout.json --output data/eval_heldout_stage2.json \
  --cache-dir data/eval_cache/heldout
```

Generator `openai/gpt-oss-20b`, judge `openai/gpt-oss-120b`, thresholds 0.2 / 0.5 / 0.35, top_k 3. All 12 answerable questions passed the guardrails and were answered.

| Metric | Mean | Questions scored |
|---|---|---|
| Faithfulness | **0.791** | 12/12 |
| Answer relevancy | **0.885** | 12/12 |
| Context precision (with reference) | **0.361** | 12/12 |
| Context recall | **0.458** | 12/12 |

No judge call failed: the run log contains 0 "judge failed" or exception lines, so no metric was skipped or returned NaN. The harness reports `n_scored` per metric, never averages over failed calls, and leaves a failed metric unscored rather than counting it as 0. The answer cache key includes the guardrail thresholds, so answers generated at `SOURCE_MIN_SCORE=0.3` are never reused at 0.35.

Per question:

| ID | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| ho-attn-01 | 1.000 | 0.999 | 1.000 | 1.000 |
| ho-attn-02 | 0.750 | 0.919 | 0.000 | 0.000 |
| ho-bert-01 | 1.000 | 0.968 | 0.000 | 0.000 |
| ho-resnet-01 | 1.000 | 0.866 | 0.333 | 0.500 |
| ho-resnet-02 | 0.800 | 0.869 | 0.000 | 0.500 |
| ho-gan-01 | 0.750 | 0.795 | 1.000 | 1.000 |
| ho-gan-02 | 1.000 | 0.951 | 0.000 | 0.000 |
| ho-gpt3-01 | 0.857 | 0.720 | 0.000 | 0.000 |
| ho-rag-01 | 1.000 | 0.996 | 1.000 | 1.000 |
| ho-rag-02 | 0.000 | 0.643 | 0.000 | 0.000 |
| ho-dataperf-01 | 0.500 | 0.937 | 1.000 | 1.000 |
| ho-llmsec-01 | 0.833 | 0.961 | 0.000 | 0.500 |

**Why the context scores are low.** For 5 of 12 questions (attn-02, bert-01, gan-02, gpt3-01, rag-02), context recall is 0. Manual inspection of the cached contexts confirmed that none of their top-3 chunks contains the answer: no label-smoothing value, no BooksCorpus, no Helvetica scenario, no WebText classifier, no mention of Facebook. The right *paper* was retrieved every time (document-level Hit@3 12/12), but not the right *passage*. What the generator did in those 5 cases:

- **3 of 5 declined**, saying the sources do not contain the answer (attn-02, bert-01, gan-02).
- **1 of 5 answered from related but off-target retrieved material** (gpt3-01 described the contamination analysis instead of the quality-filtering classifier; faithfulness 0.857).
- **1 of 5 used outside knowledge:** for rag-02 it answered "Lewis et al. (2020)", which the retrieved chunks do not support (faithfulness 0.000).

For comparison, the tuning set (24 questions, `SOURCE_MIN_SCORE=0.3`, the same harness) scored faithfulness 0.875, answer relevancy 0.839, context precision 0.781 and context recall 0.833. Those held-out questions target narrower details (specific hyperparameters, datasets and definitions), and passage-level retrieval is the weak point on them.

## Task 4: Latency

```bash
python scripts/measure_latency.py --pause 25 --output data/latency_paced.json
python scripts/measure_latency.py --pause 0  --output data/latency_back_to_back.json
```

24 answerable tuning-set questions, run sequentially through `RAGPipeline.query()` in-process, after one excluded warm-up query. Timings come from `time.perf_counter()`. "Retrieval + guardrails" is everything except the LLM call (scope check, embedding, ChromaDB search, confidence check and filtering). Percentiles use the nearest-rank method (p95 of 24 values = the 23rd smallest). Retries are counted from the OpenAI client's `Retrying request` log lines.

### Headline: paced run (25 s between queries)

| Measure | n | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| End-to-end | 24 | **1,092.7 ms** | **1,602.2 ms** | 607.1 ms | 2,567.4 ms |
| LLM generation only | 24 | 1,005.5 ms | 1,506.6 ms | 511.7 ms | 2,480.5 ms |
| Retrieval + guardrails | 24 | 90.0 ms | 102.5 ms | 75.0 ms | 114.2 ms |

Rate-limit retries: **0 of 24 queries.**

### Back-to-back run (no pause)

| Measure | n | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| End-to-end, all queries | 24 | 14,046.0 ms | 18,724.9 ms | 501.9 ms | 19,233.5 ms |
| End-to-end, retry-free queries only | 4 | 768.8 ms | 1,733.9 ms | 501.9 ms | 1,733.9 ms |
| Retrieval + guardrails | 24 | 58.5 ms | 102.7 ms | 47.8 ms | 112.7 ms |

Rate-limit retries: **20 of 24 queries** retried at least once (24 retries in total). The first 4 queries ran without retries. From the 5th on, Groq returned HTTP 429 because of the free tier's **8,000 tokens-per-minute** limit on `gpt-oss-20b` (read from the `x-ratelimit-limit-tokens` response header). The OpenAI client then waited before retrying; in a separate diagnostic run, waits per retry ranged from 3 to 15 s (8, 13, 13, 12, 3 and 15 s). Those waits are free-tier throttling, not pipeline compute. A single RAG query uses a large share of the per-minute budget, so sustained back-to-back traffic on the free tier waits about 14 s per query. No retry logic was changed.

### Notes

- These are in-process timings on the machine above. They exclude HTTP/FastAPI overhead and do not describe the deployed Space, which runs on HuggingFace's `cpu-basic` hardware.
- Groq generation time depends on network conditions and Groq's load at the time of the run.
- Retrieval + guardrails varied between runs on the same machine: medians of 90.0 ms (paced) and 58.5 ms (back-to-back). An earlier, discarded run measured about 37 ms. Treat the figure as "tens of milliseconds", not a precise constant.

## Known limitations

1. **Tiny held-out negatives.** Only 4 unanswerable and 4 out-of-scope questions. The 2/4 refusal rate has wide uncertainty; it shows the tuning-set 100% does not generalise, but not what the true rate is.
2. **Thresholds sit close to the boundary.** The two leaked unanswerable questions scored 0.510 and 0.512 against a 0.5 threshold. Small changes to the index or embedding model can flip these decisions.
3. **Production differs from what was measured.** The live index produces different chunk scores than the local one, so all guardrail and retrieval figures here describe the local index at the stated thresholds. The live confidence threshold was below 0.5 until 2026-10-03; it is now set to 0.5, which a two-question smoke test is consistent with but cannot prove exactly.
4. **Passage-level retrieval is weak on held-out questions.** Context recall is 0.458, and 5 of 12 questions retrieved the right paper but not the answer-bearing passage. Document-level Hit@3 (12/12) hides this. The RAGAS numbers in `README.md` come from the 24-question tuning set at `SOURCE_MIN_SCORE=0.3` and are higher than the held-out results.
5. **Shared quota.** Evaluation and the live app draw on the same Groq organisation's limits, so evaluation runs can take the live demo offline.
6. **Hit@3 is document-level.** It does not confirm the retrieved chunk contains the answer.
7. **Test fragility.** Two API tests depend on whether `frontend/dist/` exists.
8. **Latency is local.** In-process timings on a laptop, on Groq's free tier; network-dependent.

## Files

| File | Contents |
|---|---|
| `tests/eval_heldout.json` | Frozen held-out set (20 questions) |
| `data/eval_heldout_stage1.json` | Held-out retrieval and guardrail results, per question |
| `data/eval_heldout_stage2.json` | Held-out RAGAS results: answers, sources, per-question scores |
| `data/heldout_leaked_unanswerable.json` | LLM responses to the two unanswerable questions that passed the guardrails |
| `data/latency_paced.json`, `data/latency_back_to_back.json` | Per-query latency, retry counts, hardware |
| `data/live_space_probe.json` | Live-Space guardrail decisions and scores vs local (before the threshold fix) |
| `data/live_smoke_test.json` | Live smoke test after `CONFIDENCE_THRESHOLD` was set to 0.5 |
| `scripts/measure_latency.py` | Latency harness |
| `scripts/probe_live_space.py` | Live-Space probe |
