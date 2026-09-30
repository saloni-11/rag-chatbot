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

The evaluation dataset (`tests/eval_dataset.json`) has 32 labelled questions across all 9 source papers:

- **24 answerable**: each has a ground-truth answer written from the paper and the source document(s) it should be retrieved from
- **3 unanswerable**: on-topic for AI/ML, but not covered by the corpus, so the bot should decline rather than guess
- **5 out-of-scope**: off-topic questions the scope guardrail should reject

Evaluation runs in two stages:

```bash
# Stage 1: retrieval + guardrails. Offline, deterministic, no API key.
python src/evaluation/ragas_eval.py --stage retrieval --sweep

# Stage 2: generation quality with RAGAS. Needs GROQ_API_KEY.
pip install -r requirements-phase10.txt
python src/evaluation/ragas_eval.py
# Results saved to data/eval_results.json
```

Stage 2 scores answers with RAGAS (faithfulness, answer relevancy, context precision, context recall). The judge is a separate, larger Groq model than the 8B generator, so the model never grades its own answers. Embeddings come from HuggingFace, so no OpenAI key is needed.

### Stage 1 results (562-chunk index, top_k = 3)

| Metric | Score |
|---|---|
| Retrieval Hit@3 / MRR | 0.958 / 0.958 |
| Answerable questions let through by guardrails | 1.00 (24/24) |
| Unanswerable questions declined | 1.00 (3/3) |
| Out-of-scope questions declined | 1.00 (5/5) |
| Scope check + retrieval latency (p50) | ~11 ms + ~16 ms |

### What the evaluation found and fixed

- **Stale index.** The local ChromaDB index held only 2 of the 9 papers (77 chunks), because it was built before the other 7 papers were added. Re-running ingestion would have appended duplicate vectors to the old collection, so ingestion now rebuilds the collection from scratch (562 chunks). On the same dataset, Hit@3 went from **0.33 to 0.96**.
- **Truncated judge context.** The first version of the evaluation passed the UI's 500-character source previews to RAGAS, while the LLM saw full ~1,400-character chunks. That penalised faithfulness for claims that were actually supported. The pipeline now returns the full contexts separately.
- **Silent NaN scores.** RAGAS defaults to 16 concurrent workers and records every failed call as NaN. On Groq's free tier, that meant most scores were lost to rate limits. The harness now uses 2 workers with retry/backoff, reports how many questions each metric actually scored, and caches scores so an interrupted run can resume.
- **Guardrail thresholds tuned with data.** At the previous settings (scope 0.30, confidence 0.40), 1 answerable question was wrongly rejected and 2 of the 3 unanswerable questions reached the LLM. `--sweep` grid-searches both thresholds. Off-topic and on-topic questions overlap around 0.2–0.25 scope similarity, but the retrieval-confidence scores separate them cleanly: off-topic and unanswerable questions score ≤ 0.50, answerable ones ≥ 0.53. The defaults are now scope 0.20 and confidence 0.50. The margins are thin and were tuned on this same dataset, so more held-out questions are the next step.
- **Known miss.** "How does the Transformer encode position?" retrieves BERT chunks ahead of the Attention paper. That is a limit of the small embedding model and a candidate for a reranker or hybrid BM25 search.

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