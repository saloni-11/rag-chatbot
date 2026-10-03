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
                 - source filter  [ChromaDB]  [Groq LLM]
                              Vector Store  (gpt-oss-20b)
                                   │
                            [Embeddings]
                   (sentence-transformers/all-MiniLM-L6-v2)
```

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

43 tests across three suites: ingestion (unit), guardrails (unit with mocks), and API (integration with test client).

---

## 📊 RAG Evaluation (Phase 10)

📄 **Full methodology, commands, raw counts and per-question results: [EVAL_REPORT.md](EVAL_REPORT.md).**

There are two labelled question sets, covering all 9 source papers:

| Set | File | Questions | Role |
|---|---|---|---|
| **Tuning set** | `tests/eval_dataset.json` | 24 answerable, 3 unanswerable, 5 out-of-scope | Used to choose the guardrail thresholds, so its guardrail results are **in-sample** |
| **Held-out set** | `tests/eval_heldout.json` | 12 answerable, 4 unanswerable, 4 out-of-scope | New questions, frozen in commit `31c4819` before any evaluation ran on them, then evaluated once at fixed thresholds |

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

### Held-out results (headline)

Thresholds: scope 0.2, confidence 0.5, source minimum 0.35 (the production values). Local 562-chunk index, top_k = 3.

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
- **Passage-level retrieval.** On held-out questions, 5 of 12 retrieved the right paper but not the answer passage (context recall 0.458). A reranker or hybrid BM25 search is the next thing to try.
- **Small held-out set.** Only 4 unanswerable and 4 out-of-scope held-out questions, so 2/4 and 4/4 are rough estimates. The confidence threshold sits close to the boundary: the two misses scored 0.510 and 0.512 against 0.5.
- **Local vs deployed index.** The Docker build rebuilds the index, and the deployed chunk scores differ slightly from the local ones, so production retrieval can differ from what was measured.
- **Shared rate limits.** Evaluation runs and the live demo share one Groq organisation's quota, so heavy evaluation can exhaust the daily token budget for the live app.

---

## 🔒 Guardrails

Three layers of protection, each saving unnecessary API calls:

- **Scope guardrail** — embeds the question and compares against reference AI/ML phrases using cosine similarity. Off-topic questions are rejected before hitting ChromaDB or the LLM.
- **Confidence threshold** — checks retrieval similarity scores after ChromaDB search. If the best chunk isn't relevant enough, returns a fallback response without calling the LLM.
- **Source filtering** — removes low-scoring chunks before sending to the LLM, ensuring it only sees high-quality context.

All thresholds are configurable via environment variables (`SCOPE_THRESHOLD`, `CONFIDENCE_THRESHOLD`, `SOURCE_MIN_SCORE`).

---

## 📝 License

MIT