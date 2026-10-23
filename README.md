---
title: AI/ML Study Companion
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🤖 AI/ML Study Companion

[![CI](https://github.com/saloni-11/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/saloni-11/rag-chatbot/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace_Spaces-blue)](https://huggingface.co/spaces/salonisamant01/study-companion)

A production-grade Retrieval-Augmented Generation (RAG) study companion for AI and Data Analytics learning. Built with **LlamaIndex**, **ChromaDB**, **Groq LLM**, **FastAPI**, and a **React** frontend — deployed on **HuggingFace Spaces** with **GitHub Actions** CI/CD.

**[Try the live demo →](https://huggingface.co/spaces/salonisamant01/study-companion)**

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[React Frontend] ──────► [FastAPI Backend]
  (Vite + Tailwind)              │
                       ┌─────────┴──────────┐
                       │                    │
                 [Guardrails]        [LlamaIndex RAG]
                 - scope check            │
                 - confidence      ┌──────┴──────┐
                   threshold       │             │
                 - source filter  [Hybrid      [Groq LLM]
                                  retrieval]  (gpt-oss-20b)
                                   │
              ┌────────────────────┼────────────────────┐
     [ChromaDB dense]          [BM25 keyword]     same 562 chunks
     (all-MiniLM-L6-v2)        (rank-bm25)
              └──────► Reciprocal Rank Fusion ◄─────────┘
                                   │
                  [Cross-encoder rerank, top 3 to the LLM]
                  (cross-encoder/ms-marco-MiniLM-L-6-v2)
```

`RETRIEVAL_MODE=dense` switches back to the original dense-only top-3.

---

## 🛠️ Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| RAG Framework | LlamaIndex | Chosen over LangChain for deeper RAG learning |
| Vector Store | ChromaDB | Free, local, persistent |
| LLM | Groq API (openai/gpt-oss-20b) | Free tier, fast inference (Llama 3.1 8B was retired by Groq) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Runs locally, free |
| Backend API | FastAPI + Uvicorn | With Pydantic schemas |
| Frontend | React (Vite + Tailwind CSS) | Chat UI with source panel |
| Containerisation | Docker + Docker Compose | Multi-stage build, health check |
| CI/CD | GitHub Actions | Lint → test → Docker build → deploy |
| Deployment | HuggingFace Spaces | Docker-based, auto-deploy on push |
| Testing | Pytest | 43 tests (unit + integration) |
| Evaluation | RAGAS 0.2.15 + custom harness | Retrieval Hit@k/MRR, guardrail accuracy, RAGAS generation metrics |
| Code Quality | black + isort + flake8 | Pinned versions, runs in CI |

---

## 📁 Project Structure

```
rag-chatbot/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint → test → Docker build
│       └── deploy.yml          # Auto-deploy to HuggingFace Spaces
├── .gitignore
├── .env.example                # Environment variable template
├── README.md
│
├── requirements.txt            # Full deps (Docker / deployment)
├── requirements-phase2.txt     # Phase 2: data ingestion only
├── requirements-phase3.txt     # Phase 3: embeddings + vector store
├── requirements-phase4.txt     # Phase 4: RAG + Groq LLM
├── requirements-phase6.txt     # Phase 6: FastAPI backend
├── requirements-phase10.txt    # Phase 10: RAGAS evaluation
├── requirements-dev.txt        # Dev/test dependencies
│
├── data/
│   ├── raw/                    # Source documents (PDFs, MD files)
│   ├── chroma_db/              # ChromaDB vector store (gitignored)
│   └── eval_results.json       # Latest evaluation report
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py           # Document loaders (PDF, MD, text)
│   │   └── chunker.py          # Chunking strategies (SentenceSplitter)
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embeddings.py       # Embedding model setup (all-MiniLM-L6-v2)
│   │   └── vector_store.py     # ChromaDB operations
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # RAG query pipeline (orchestrator)
│   │   └── guardrails.py       # Scope check, confidence threshold, source filtering
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app + CORS + lifespan + static serving
│   │   ├── routes.py           # API endpoints (/api/query, /api/health)
│   │   └── schemas.py          # Pydantic request/response models
│   └── evaluation/
│       └── ragas_eval.py       # Retrieval, guardrail and RAGAS evaluation
│
├── frontend/                   # React app (Vite + Tailwind CSS)
│   ├── index.html
│   ├── vite.config.js          # Vite config with Tailwind + API proxy
│   ├── package.json
│   └── src/
│       ├── main.jsx            # React entry point
│       ├── index.css           # Tailwind import + custom styles
│       ├── App.jsx             # Main chat application
│       └── components/
│           ├── ChatMessage.jsx # Chat message bubble component
│           └── SourcePanel.jsx # Retrieved sources side panel
│
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_ingestion.py       # Unit tests for loader + chunker
│   ├── test_guardrails.py      # Unit tests with mocked embeddings
│   ├── test_api.py             # Integration tests for API endpoints
│   └── eval_dataset.json       # 32 labelled evaluation questions
│
├── Dockerfile                  # Multi-stage build (Node → Python)
├── docker-compose.yml          # Local dev orchestration
│
└── scripts/
    ├── ingest_data.py          # Data ingestion pipeline
    └── test_rag.py             # Interactive RAG testing script
```

---

## 🚀 Phases

| Phase | What | Skills Learned | Status |
|---|---|---|---|
| 1 | Project setup, GitHub repo, dev environment | Git flow, project structure | ✅ |
| 2 | Data ingestion pipeline | Document loaders, chunking strategies | ✅ |
| 3 | Vector store + embeddings | ChromaDB, sentence-transformers | ✅ |
| 4 | RAG core with LlamaIndex | LlamaIndex query engine, retrieval | ✅ |
| 5 | Guardrails implementation | Scope checking, confidence thresholds, source filtering | ✅ |
| 6 | FastAPI backend | REST APIs, Pydantic, async Python, CORS | ✅ |
| 7 | React frontend | Vite, Tailwind CSS, component composition, API integration | ✅ |
| 8 | Docker + CI/CD | Multi-stage Dockerfile, GitHub Actions, automated testing | ✅ |
| 9 | Deploy to HuggingFace Spaces | Docker deployment, secrets management, CI/CD pipeline | ✅ |
| 10 | RAG Evaluation with RAGAS | Faithfulness, context precision, evaluation datasets | ✅ |

---

## ⚙️ Local Setup

### Quick start (phased installation)

Dependencies are split into per-phase files to keep installs lightweight.

```bash
# 1. Clone the repo
git clone https://github.com/saloni-11/rag-chatbot.git
cd rag-chatbot

# 2. Create conda environment
conda create -n ragbot python=3.12 -y
conda activate ragbot

# 3. Install dependencies (phase by phase)
pip install -r requirements-phase2.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-phase3.txt
pip install -r requirements-phase4.txt
pip install -r requirements-phase6.txt

# (Optional) Phase 10: RAGAS evaluation
pip install -r requirements-phase10.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=your-key-here
#   HF_HUB_OFFLINE=1

# 5. Add source documents to data/raw/
#    (PDFs, markdown, or text files about AI/ML topics)

# 6. Run data ingestion
python scripts/ingest_data.py

# 7. Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running the app

You need two terminals running simultaneously:

```bash
# Terminal 1: FastAPI backend (from project root)
uvicorn src.api.main:app --reload

# Terminal 2: React frontend (from frontend folder)
cd frontend
npm run dev
```

Then open `http://localhost:5173/` in your browser.

### API documentation

With the backend running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 🐳 Docker

```bash
# Build and run locally
docker-compose up --build

# Or build manually
docker build -t study-companion .
docker run -p 8000:8000 --env-file .env study-companion
```

The Dockerfile runs data ingestion during build, baking the ChromaDB index into the image.

---

## 🚀 Deployment

The app auto-deploys to HuggingFace Spaces on every push to `main`:

1. **CI pipeline** (`.github/workflows/ci.yml`) runs lint → test → Docker build
2. **Deploy pipeline** (`.github/workflows/deploy.yml`) pushes to HuggingFace if CI passes
3. **HuggingFace** builds the Docker image and serves the app

Secrets (`GROQ_API_KEY`, `HF_TOKEN`) are managed through GitHub Secrets and HuggingFace Space Secrets — never committed to code.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest --cov=src --cov-report=term-missing
```

57 tests across four suites: ingestion (unit), guardrails (unit with mocks), hybrid retrieval (unit with fake retrievers and reranker), and API (integration with test client).

---

## 📊 RAG Evaluation (Phase 10)

📄 **Full methodology, commands, raw counts and per-question results: [EVAL_REPORT.md](EVAL_REPORT.md).**

There are three labelled question sets:

| Set | File | Questions | Role |
|---|---|---|---|
| **Tuning set** | `tests/eval_dataset.json` | 24 answerable, 3 unanswerable, 5 out-of-scope | Used to choose the guardrail thresholds and the hybrid-retrieval settings, so its results are **in-sample** |
| **Held-out set** | `tests/eval_heldout.json` | 12 answerable, 4 unanswerable, 4 out-of-scope | New questions, frozen in commit `31c4819` before any evaluation ran on them, then evaluated once at fixed thresholds (dense pipeline) |
| **Fresh held-out set** | `tests/eval_heldout_fresh.json` | 7 answerable, 2 unanswerable, 1 out-of-scope | Frozen in commit `79ea4c4` before hybrid retrieval was built; used once to measure it. Now spent |

### Hybrid retrieval + reranking: fresh held-out results (headline)

> ⚠️ **Small sample.** There are only 7 answerable questions, so one judge verdict moves a mean by about **0.14**. Read the result as "roughly 0.45 → 0.71 precision, about 3 of 7 questions materially better", not as precise decimals.

The baseline is the previous dense-only pipeline (ChromaDB top 3). Hybrid retrieval takes the top 15 from dense search and the top 15 from BM25, fuses them with Reciprocal Rank Fusion (k = 60), reranks the first 10 fused chunks with a cross-encoder, and sends the best 3 to the LLM. Both pipelines ran on the same questions, 3 times each, because ChromaDB's HNSW search is not deterministic. All 3 runs retrieved identical chunks for every question, so each range is a single value. The range covers retrieval variation only: each context was judged once, so judge noise is not in it. The set's sha256 was checked before and after every run.

| Answerable, n = 7 | Dense baseline | Hybrid + rerank |
|---|---|---|
| Context precision | ≈ 0.45 | ≈ 0.71 |
| Context recall | ≈ 0.57 | ≈ 0.71 |

| Question | Precision, base → hybrid | Recall, base → hybrid |
|---|---|---|
| Q1 GPT-3 context window | 0.00 → 0.00 | 0.00 → 0.00 |
| Q2 GPT-3 limitations | 1.00 → 1.00 | 1.00 → 1.00 |
| Q3 DataPerf acquisition | 0.33 → **1.00** | 1.00 → 1.00 |
| Q4 DataPerf speech selection | 0.83 → **1.00** | 1.00 → 1.00 |
| Q5 RAG retrieval granularity | 0.00 → **1.00** | 0.00 → **1.00** |
| Q6 RAG survey training categories | 0.00 → 0.00 | 0.00 → 0.00 |
| Q7 BERT input representation | 1.00 → 1.00 | 1.00 → 1.00 |

**Where the gain comes from.** The reranker surfaces better passages from a wider pool. In Q3 and Q4, dense search had already found relevant passages but ranked irrelevant ones above them; after reranking, the relevant ones come first. In Q5, dense search missed the answer passage entirely, and hybrid retrieval found it. That is the only recall gain.

**What it is, honestly.** Hybrid + rerank is a precision intervention. It reorders and filters what reaches the LLM. It improves recall only when a passage was missed entirely, as in Q5. The tuning set shows the same pattern: precision 0.785 → 0.865, recall unchanged at 0.833 (n = 24).

**Negatives (same in all 3 runs, both pipelines):** Q9 (Gini coefficient, unanswerable) was declined by the confidence check, and Q10 (a leaky tap, out of scope) by the scope check. **Q8 (t-SNE, unanswerable) was not declined.** See the known limitations below.

**Tuning (tuning set only).** `scripts/tune_hybrid.py` simulated 47 configurations from one retrieval per question and scored them with the same RAGAS judge. Judge verdicts were cached per (question, chunk), so a chunk shared by two configurations got the same verdict. Results:
- The RRF constant k (10, 30, 60) made no difference at all.
- Candidates per retriever (10, 15, 20) and rerank pool size (10, 15, all) made no meaningful difference (precision 0.851–0.868), so the cheapest top setting was chosen.
- Top-5 instead of top-3 raised recall by 0.021, two questions up and two down. Top-5 contains top-3, so the downs are judge noise. It cost 0.035 precision and two more chunks per prompt, so top-3 was kept.

**Latency cost (measured on the live Space).** These are 10 paced queries to the deployed Space (free CPU tier) on 2026-10-23, all answered, with no rate-limit retries:

| Live Space, n = 10 | Min | Median | Max |
|---|---|---|---|
| End-to-end, as a user experiences it (includes network) | 1.95 s | **2.3 s** | 2.66 s |
| Server-side compute (the app's own per-query total) | 1.26 s | **1.6 s** | 1.98 s |

Hybrid + rerank adds roughly **0.5 s of compute per query** over the dense pipeline: 1.09 s → 1.6 s server-side. It does not double response time. About 0.7 s of the 2.3 s a user waits is network and Hugging Face's proxy; that gap was steady on every query (0.67–0.77 s).

The 0.5 s is approximate, for three reasons:
- The dense baseline was measured on a different day, by running the pipeline on a local machine. Dense retrieval itself takes about 20 ms, so the hardware difference matters little.
- Both figures include Groq's response time, which varies.
- The end-to-end figures were measured from one location; users nearer Hugging Face's servers will see a smaller network gap.

Reranking time can't be isolated on the Space, because its logs record only each query's total. Reranking all ~24 fused chunks instead of 10 roughly doubled the added time in local 2-thread tests, which is why the rerank pool is capped at 10 (`RERANK_POOL`).

**The source filter now does real work.** It runs on the reranked pool before the top-3 cut. In tuning question rag-03, the reranker ranked the RAG survey's title-and-author page 2nd (cosine 0.183). The filter removed it, and the next passage took its place. It removed nothing on the fresh set.

### Remaining failure: right paper, wrong passage (Q1, Q6)

Q1 and Q6 fail in both pipelines. In both, the answer passage is clean prose and is in the reranked candidate pool, but the final stage ranks it out of the top 3:

- **Q1** ("What is the GPT-3 context window, and how many in-context examples fit?"). The passage stating "K in the range of 10 to 100 … context window (n_ctx = 2048)" is ranked 2nd by BM25 and makes the 10-chunk pool, but the reranker places it 4th. Above it are chunks that begin with figure captions about in-context examples ("Figure 3.8: Performance on SuperGLUE … number of examples in context"). Its dense cosine is only 0.221, so even ranked into the top 3 it would be removed by the source filter (0.25). The Section 2.4 passage that also states 2048 is ranked 6th by BM25 but does not survive the pool cap.
- **Q6** ("What four training approaches does the survey review?"). The passage naming all four ("training-free, independent training, sequential training, and joint training") has cosine 0.456 and is in the pool, but the reranker ranks it 8th. The top 3 are dominated by chunks that begin with the survey's repeated page header ("A Survey on RAG Meeting LLMs … KDD '24").

So the failures are in final-stage ranking, and parsing contamination feeds them: captions and page headers that the PDF parser leaves inline make the wrong chunks look relevant to the reranker. That gives the parser work a concrete target. Strip running headers and figure captions from chunk text, then re-check whether Q1 and Q6's answer passages reach the top 3. Q1 also shows that the cosine-based source filter can block a passage that BM25 found correctly. Because the fresh set is now spent, the parser change has to be measured on new frozen questions.

"Unanswerable" means on-topic for AI/ML but not covered by the corpus, so the bot should decline. "Out-of-scope" means off-topic.

Evaluation runs in two stages:

```bash
# Stage 1: retrieval + guardrails. Offline, deterministic, no API key.
python src/evaluation/ragas_eval.py --stage retrieval --dataset tests/eval_heldout.json

# Stage 2: generation quality with RAGAS. Needs GROQ_API_KEY.
pip install -r requirements-phase10.txt
python src/evaluation/ragas_eval.py --stage generation --dataset tests/eval_heldout.json
```

Stage 2 scores answers with RAGAS (faithfulness, answer relevancy, context precision, context recall). The judge (`gpt-oss-120b`) is a separate, larger model than the generator (`gpt-oss-20b`), so the model never grades its own answers. Embeddings come from HuggingFace, so no OpenAI key is needed.

### Held-out results (dense pipeline, before hybrid retrieval)

Thresholds: scope 0.2, confidence 0.5, source minimum 0.35 (the production values at the time, on ChromaDB's score scale). Local 562-chunk index, dense top_k = 3.

| Guardrail metric | Result |
|---|---|
| Answerable questions wrongly rejected | **0/12** |
| Out-of-scope questions refused | **4/4**, all by the scope layer |
| Unanswerable (on-topic, not in corpus) questions refused | **2/4** |
| Retrieval Hit@3 (right *paper* in top 3) | 12/12 |

The two unanswerable questions that got through (the SVM kernel trick, and bagging vs boosting) scored 0.512 and 0.510 against the 0.5 confidence threshold. Both reached the LLM, and the LLM declined both, saying its sources don't cover the topic. So no answer was made up, but each cost an LLM call.

| RAGAS metric | Mean | Scored |
|---|---|---|
| Faithfulness | 0.791 | 12/12 |
| Answer relevancy | 0.885 | 12/12 |
| Context precision | 0.361 | 12/12 |
| Context recall | 0.458 | 12/12 |

The low context scores are the main finding. For 5 of 12 questions, retrieval found the right paper but not the passage containing the answer, which document-level Hit@3 doesn't show. In 3 of those 5 the bot said its sources didn't contain the answer, in 1 it answered from related but off-target text, and in 1 it used outside knowledge (faithfulness 0.0).

| Latency (local, Groq free tier) | Median | p95 |
|---|---|---|
| End-to-end, paced queries (n=24, 0 rate-limit retries) | 1.09 s | 1.60 s |
| End-to-end, back-to-back queries (n=24, 20 rate-limited) | 14.0 s | 18.7 s |

Back-to-back queries exceed Groq's free-tier limit of 8,000 tokens per minute, and the client then waits and retries. See the report for details.

### Tuning-set results (in-sample)

⚠️ The guardrail thresholds were chosen on these questions, so the guardrail rows below measure fit, not generalisation. Compare the held-out results above. Source minimum score was 0.3 for these runs.

| Metric (tuning set) | Result |
|---|---|
| Answerable questions let through by guardrails | 24/24 (100%) |
| Unanswerable questions declined | 3/3 (100%) |
| Out-of-scope questions declined | 5/5 (100%) |
| Retrieval Hit@3 / MRR | 0.958 / 0.958 (23/24) |
| RAGAS faithfulness / answer relevancy | 0.875 / 0.839 (n=24) |
| RAGAS context precision / context recall | 0.781 / 0.833 (n=24) |

### What the evaluation found and fixed

- **Stale index.** The local ChromaDB index held only 2 of the 9 papers (77 chunks), because it was built before the other 7 papers were added. Re-running ingestion would have appended duplicate vectors to the old collection, so ingestion now rebuilds the collection from scratch (562 chunks). On the same dataset, Hit@3 went from **0.33 to 0.96**.
- **Truncated judge context.** The first version of the evaluation passed the UI's 500-character source previews to RAGAS, while the LLM saw full ~1,400-character chunks. That penalised faithfulness for claims that were actually supported. The pipeline now returns the full contexts separately.
- **Silent NaN scores.** RAGAS defaults to 16 concurrent workers and records every failed call as NaN. On Groq's free tier, that meant most scores were lost to rate limits. The harness now uses 2 workers with retry/backoff, reports how many questions each metric actually scored, and caches scores so an interrupted run can resume.
- **Guardrail thresholds tuned with data.** At the previous settings (scope 0.30, confidence 0.40), 1 answerable question was wrongly rejected and 2 of the 3 unanswerable questions reached the LLM. `--sweep` grid-searches both thresholds. On the tuning set, off-topic and on-topic questions overlap around 0.2–0.25 scope similarity, but retrieval-confidence scores separated them cleanly: off-topic and unanswerable questions scored ≤ 0.50, answerable ones ≥ 0.53. The defaults are now scope 0.20 and confidence 0.50. On the held-out set, that separation held for answerable questions (0/12 wrongly rejected), but 2 of 4 unanswerable questions scored just above 0.5 and got through.
- **A prompt instruction hurt faithfulness.** The system prompt asked every answer to end with "why it matters". The model filled that section with claims the sources didn't support: GPT-3's correct "175 billion parameters" answer scored 0.25. Removing the instruction raised faithfulness from **0.79 to 0.90** on the same 16 questions (better on 8, worse on 6, unchanged on 2). Retrieval was unchanged, and context recall held identical on all 16, so the gain is not judge noise.
- **Retired model.** Groq retired Llama 3.1 8B mid-evaluation (the API returned 404). The generator is now `gpt-oss-20b`.

### Known limitations

- **Position-encoding miss.** "How does the Transformer encode position?" retrieves BERT chunks ahead of the Attention paper. That is a limit of the small embedding model and a candidate for a reranker or hybrid BM25 search.
- **Right paper, wrong passage.** For "How many parameters do BERT-base and BERT-large have?", the right paper is retrieved but not the passage with the numbers. The bot correctly says it doesn't know rather than guessing.
- **LLM-security paper.** Its PDF extracts with the spaces between words missing ("Asurveyonlargelanguagemodel..."), and its two questions score lowest (faithfulness 0.40, context recall 0.0). A layout-aware PDF parser would likely fix it.
- **Passage-level retrieval.** On the first held-out set (dense pipeline), 5 of 12 questions retrieved the right paper but not the answer passage (context recall 0.458). Hybrid retrieval + reranking fixed some of these cases on the fresh set (see above). Q1 and Q6 still fail, because captions and page headers help the wrong chunks win the rerank.
- **Thin-margin confidence check.** The fresh set's t-SNE question (unanswerable) got through in both pipelines. Its best chunk scored cosine 0.334 against the 0.307 threshold. Like the earlier held-out misses, a plausibly phrased on-topic question can sit just above the line. The threshold was deliberately **not** changed to catch it: the question is on the now-spent fresh set, and tuning to it would contaminate that result. Any fix will be a separate change, measured on new questions.
- **Reranker latency.** On the live Space, hybrid retrieval with reranking adds about 0.5 s of compute per query, for a 2.3 s median wait (see above).
- **Small held-out set.** Only 4 unanswerable and 4 out-of-scope held-out questions, so 2/4 and 4/4 are rough estimates. The confidence threshold sits close to the boundary: the two misses scored 0.510 and 0.512 against 0.5.
- **Local vs deployed index.** All evaluation numbers in this README were measured on the local 562-chunk index. The deployed index is not that index: the Docker build re-runs ingestion and builds its own, and it is not identical. A build of the same Dockerfile on 2026-10-22 produced 553 chunks, and the deployed chunk scores differ slightly from the local ones. Production retrieval can therefore differ from what was measured. The cause of the 9-chunk difference is still being investigated.
- **Shared rate limits.** Evaluation runs and the live demo share one Groq organisation's quota, so heavy evaluation can exhaust the daily token budget for the live app.

---

## 🔒 Guardrails

Two active guardrails decide whether a question is answered at all. Both run before the LLM is called, so a refused question costs no API call:

- **Scope check** (`SCOPE_THRESHOLD = 0.2`) — embeds the question and compares it against reference AI/ML phrases using cosine similarity. Off-topic questions are rejected before ChromaDB or the LLM are touched.
- **Confidence check** (`CONFIDENCE_MIN_COSINE = 0.307`) — after retrieval, if the best chunk's cosine similarity is below the threshold, the bot returns a fallback response instead of calling the LLM.

A third stage, the **source-quality filter** (`SOURCE_MIN_COSINE = 0.25`), removes individual chunks below the threshold before they reach the LLM. With hybrid retrieval it runs on the whole reranked candidate pool *before* the top-3 cut, so a weak chunk the reranker liked is replaced by the next one instead of just shrinking the context. In tuning question rag-03, it removed the RAG survey's title page (cosine 0.183), which the reranker had ranked 2nd. The confidence check then runs on the 3 chunks that would actually reach the LLM. Every candidate's cosine, including chunks only BM25 found, is computed exactly from the stored embeddings. In dense-only mode (`RETRIEVAL_MODE=dense`), the filter is a no-op at top_k = 3, as before. It is kept below the confidence threshold so it can only trim weak chunks, never refuse a question by itself.

All thresholds are true cosine similarity and configurable via environment variables (`SCOPE_THRESHOLD`, `CONFIDENCE_MIN_COSINE`, `SOURCE_MIN_COSINE`). ChromaDB reports retrieval scores as e^(cosine − 1); the guardrails convert them back to cosine before comparing (see `chroma_score_to_cosine` in `src/rag/guardrails.py`). The old `CONFIDENCE_THRESHOLD` / `SOURCE_MIN_SCORE` variables held values on ChromaDB's scale and are now ignored, with a warning.

---

## 📝 License

MIT