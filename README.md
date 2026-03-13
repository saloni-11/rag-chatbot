# 🤖 AI/ML Interview Prep RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot for AI and Data Analytics interview preparation. Built with **LlamaIndex**, **ChromaDB**, **Groq LLM**, **FastAPI**, and a **React** frontend — deployed on **HuggingFace Spaces** with **GitHub Actions** CI/CD.

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
                 - hallucination   ┌──────┴──────┐
                   prevention      │             │
                              [ChromaDB]    [Groq LLM]
                              Vector Store  (llama3-8b-8192)
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
| LLM | Groq API (llama3-8b-8192) | Free tier, fast inference |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Runs locally, free |
| Backend API | FastAPI + Uvicorn | With Pydantic schemas |
| Frontend | React (Vite + Tailwind CSS) | Replaced Streamlit for portfolio impact |
| CI/CD | GitHub Actions | Industry standard pipelines |
| Deployment | HuggingFace Spaces | Free tier |
| Evaluation | RAGAS | Phase 10 |
| Testing | Pytest | With coverage |
| Code Quality | black + isort + flake8 | Runs in CI |

---

## 📁 Project Structure

```
rag-chatbot/
├── .github/
│   └── workflows/           # GitHub Actions CI/CD
├── .gitignore
├── README.md
├── .env.example             # Environment variable template
│
├── requirements.txt         # Full deps (Docker / deployment)
├── requirements-phase2.txt  # Phase 2: data ingestion only
├── requirements-phase3.txt  # Phase 3: embeddings + vector store
├── requirements-phase4.txt  # Phase 4: RAG + Groq LLM
├── requirements-phase6.txt  # Phase 6: FastAPI backend
├── requirements-dev.txt     # Dev/test dependencies
│
├── docker/
│   ├── Dockerfile           # App container
│   └── docker-compose.yml   # Local dev orchestration
│
├── data/
│   ├── raw/                 # Source documents (PDFs, MD files)
│   └── processed/           # Chunked/processed data (gitignored)
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # Document loaders (PDF, MD, text)
│   │   └── chunker.py       # Chunking strategies (SentenceSplitter)
│   ├── indexing/
│   │   ├── embeddings.py    # Embedding model setup
│   │   └── vector_store.py  # ChromaDB operations
│   ├── rag/
│   │   ├── pipeline.py      # LlamaIndex RAG pipeline
│   │   └── guardrails.py    # Source grounding & guardrails
│   ├── api/
│   │   ├── main.py          # FastAPI app entry point
│   │   ├── routes.py        # API endpoints
│   │   └── schemas.py       # Pydantic request/response models
│   └── evaluation/
│       └── ragas_eval.py    # RAG evaluation with RAGAS
│
├── frontend/                # React app (Vite + Tailwind CSS)
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   └── test_api.py
│
├── docs/
│   └── architecture.md      # Detailed architecture notes
│
└── scripts/
    └── ingest_data.py       # One-time data ingestion script
```

---

## 🚀 Phases

| Phase | What | Skills Learned | Status |
|---|---|---|---|
| 1 | Project setup, GitHub repo, dev environment | Git flow, project structure | ✅ |
| 2 | Data ingestion pipeline | Document loaders, chunking strategies | ✅ |
| 3 | Vector store + embeddings | ChromaDB, sentence-transformers | ⬜ |
| 4 | RAG core with LlamaIndex | LlamaIndex query engine, retrieval | ⬜ |
| 5 | Guardrails implementation | Source grounding, prompt engineering | ⬜ |
| 6 | FastAPI backend | REST APIs, Pydantic, async Python | ⬜ |
| 7 | React frontend | Vite, Tailwind CSS, API integration | ⬜ |
| 8 | Docker + CI/CD | Dockerfile, GitHub Actions pipelines | ⬜ |
| 9 | Deploy to HuggingFace Spaces | Cloud deployment | ⬜ |
| 10 | RAG Evaluation with RAGAS | Faithfulness, relevancy metrics | ⬜ |

---

## ⚙️ Local Setup

### Quick start (phased installation)

Dependencies are split into per-phase files to keep installs lightweight. Install only what your current phase needs.

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot

# 2. Create conda environment
conda create -n ragbot python=3.12 -y
conda activate ragbot

# 3. Install Phase 2 dependencies (~200 MB)
pip install -r requirements-phase2.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Add source documents to data/raw/
#    (PDFs, markdown, or text files about AI/ML topics)

# 6. Run data ingestion
python scripts/ingest_data.py
```

### Installing further phases

```bash
# Phase 3 — embeddings + vector store (~2.5 GB, includes PyTorch)
# TIP: install CPU-only PyTorch first to save ~1.5 GB:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-phase3.txt

# Phase 4 — RAG pipeline + Groq LLM
pip install -r requirements-phase4.txt

# Phase 6 — FastAPI backend
pip install -r requirements-phase6.txt

# Dev/test tools
pip install -r requirements-dev.txt
```

### Running the app (once all phases are built)

```bash
# Terminal 1: FastAPI backend
uvicorn src.api.main:app --reload

# Terminal 2: React frontend
cd frontend && npm run dev
```

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

This chatbot implements:
- **Scope guardrail** — only answers AI/ML/Data Analytics questions
- **Source grounding** — responses cite retrieved document chunks
- **Hallucination prevention** — answers constrained to retrieved context only
- **Confidence threshold** — low-confidence retrievals return a fallback response

---

## 📝 License

MIT