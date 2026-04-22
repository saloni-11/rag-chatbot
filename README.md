# 🤖 AI/ML Study Companion

A production-grade Retrieval-Augmented Generation (RAG) chatbot for AI and Data Analytics learning. Built with **LlamaIndex**, **ChromaDB**, **Groq LLM**, **FastAPI**, and a **React** frontend — deployed on **HuggingFace Spaces** with **GitHub Actions** CI/CD.

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
                              Vector Store  (llama3.1-8b-instant)
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
| LLM | Groq API (llama-3.1-8b-instant) | Free tier, fast inference |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Runs locally, free |
| Backend API | FastAPI + Uvicorn | With Pydantic schemas |
| Frontend | React (Vite + Tailwind CSS) | Chat UI with source panel |
| Containerisation | Docker + Docker Compose | Multi-stage build, health check |
| CI/CD | GitHub Actions | Lint → test → Docker build pipeline |
| Deployment | HuggingFace Spaces | Free tier |
| Evaluation | RAGAS | Phase 10 |
| Testing | Pytest | With coverage |
| Code Quality | black + isort + flake8 | Runs in CI |

---

## 📁 Project Structure

```
rag-chatbot/
├── .github/
│   └── workflows/              # GitHub Actions CI/CD
├── .gitignore
├── .env.example                # Environment variable template
├── README.md
│
├── requirements.txt            # Full deps (Docker / deployment)
├── requirements-phase2.txt     # Phase 2: data ingestion only
├── requirements-phase3.txt     # Phase 3: embeddings + vector store
├── requirements-phase4.txt     # Phase 4: RAG + Groq LLM
├── requirements-phase6.txt     # Phase 6: FastAPI backend
├── requirements-dev.txt        # Dev/test dependencies
│
├── data/
│   ├── raw/                    # Source documents (PDFs, MD files)
│   └── chroma_db/              # ChromaDB vector store (gitignored)
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
│   │   ├── main.py             # FastAPI app entry point + CORS + lifespan
│   │   ├── routes.py           # API endpoints (/api/query, /api/health)
│   │   └── schemas.py          # Pydantic request/response models
│   └── evaluation/
│       ├── __init__.py
│       └── ragas_eval.py       # RAG evaluation with RAGAS (Phase 10)
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
│   ├── test_ingestion.py
│   ├── test_guardrails.py
│   └── test_api.py
│
├── Dockerfile                  # Multi-stage build (Node → Python)
├── docker-compose.yml          # Local dev orchestration
│
├── docs/
│   └── architecture.md         # Detailed architecture notes
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
| 8 | Docker + CI/CD | Dockerfile, GitHub Actions pipelines | ✅ |
| 9 | Deploy to HuggingFace Spaces | Cloud deployment | ⬜ |
| 10 | RAG Evaluation with RAGAS | Faithfulness, relevancy metrics | ⬜ |

---

## ⚙️ Local Setup

### Quick start (phased installation)

Dependencies are split into per-phase files to keep installs lightweight.

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/rag-chatbot.git
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
docker-compose up --build
```

The full `requirements.txt` is used inside Docker where disk/RAM constraints don't apply.

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src
```

---

## 📊 Evaluation

```bash
python src/evaluation/ragas_eval.py
```

Metrics tracked: **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**

---

## 🔒 Guardrails

This chatbot implements three layers of protection:

- **Scope guardrail** — uses embedding similarity against reference AI/ML phrases to reject off-topic questions before hitting the LLM (saves API calls)
- **Confidence threshold** — checks retrieval similarity scores and returns a fallback response if the best chunk isn't relevant enough
- **Source filtering** — removes low-scoring chunks before sending to the LLM, so it only sees high-quality context

---

## 📝 License

MIT