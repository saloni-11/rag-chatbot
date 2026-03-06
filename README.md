# 🤖 AI/ML Interview Prep RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot for AI and Data Analytics interview preparation. Built with **LlamaIndex**, **ChromaDB**, **Groq LLM**, **FastAPI**, and deployed on **HuggingFace Spaces** with full **GitLab CI/CD**.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Streamlit UI] ──────► [FastAPI Backend]
                              │
                    ┌─────────┴──────────┐
                    │                    │
              [Guardrails]        [LlamaIndex RAG]
              - scope check            │
              - hallucination   ┌──────┴──────┐
                prevention      │             │
                           [ChromaDB]    [Groq LLM]
                           Vector Store  (free tier)
                                │
                         [Embeddings]
                    (HuggingFace sentence-transformers)
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| RAG Framework | LlamaIndex | Clean RAG abstractions, great for learning RAG deeply |
| Vector Store | ChromaDB | Free, local, production-ready |
| LLM | Groq (llama3-8b) | Free tier, very fast inference |
| Embeddings | sentence-transformers | Free, runs locally |
| Backend API | FastAPI | Industry standard, async, auto docs |
| Frontend | Streamlit | Fast to build, free HuggingFace deployment |
| CI/CD | GitLab CI | Industry standard for pipelines |
| Containerisation | Docker | Reproducible environments |
| Evaluation | RAGAS | RAG-specific metrics |
| Testing | Pytest | Standard Python testing |

---

## 📁 Project Structure

```
rag-chatbot/
├── .gitlab-ci.yml           # CI/CD pipeline
├── .gitignore
├── README.md
├── requirements.txt         # All dependencies
├── requirements-dev.txt     # Dev/test dependencies
├── .env.example             # Environment variable template
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
│   │   ├── loader.py        # Document loaders (PDF, web, MD)
│   │   └── chunker.py       # Chunking strategies
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
├── app/
│   └── streamlit_app.py     # Streamlit frontend
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   └── test_api.py
│
├── notebooks/
│   └── 01_exploration.ipynb # Experimentation notebook
│
├── docs/
│   └── architecture.md      # Detailed architecture notes
│
└── scripts/
    └── ingest_data.py       # One-time data ingestion script
```

---

## 🚀 Phases

| Phase | What | Skills Learned |
|---|---|---|
| 1 | Project setup, GitLab repo, dev environment | Git flow, project structure |
| 2 | Data ingestion pipeline | Document loaders, chunking strategies |
| 3 | Vector store + embeddings | ChromaDB, sentence-transformers |
| 4 | RAG core with LlamaIndex | LlamaIndex query engine, retrieval |
| 5 | Guardrails implementation | Source grounding, prompt engineering |
| 6 | FastAPI backend | REST APIs, Pydantic, async Python |
| 7 | Streamlit frontend | UI, session state |
| 8 | Docker + CI/CD | Dockerfile, GitLab CI pipelines |
| 9 | Deploy to HuggingFace Spaces | Cloud deployment |
| 10 | RAG Evaluation with RAGAS | Faithfulness, relevancy metrics |

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone <your-gitlab-repo-url>
cd rag-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Ingest data
python scripts/ingest_data.py

# 6. Run FastAPI backend
uvicorn src.api.main:app --reload

# 7. Run Streamlit frontend (new terminal)
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 🧪 Running Tests

```bash
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
