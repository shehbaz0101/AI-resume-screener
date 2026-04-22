# NEXUS — AI Resume Screener

RAG-powered resume screening system. Upload resumes → vector  index → LLM ranks candidates against a job description.

## Stack
| Layer | Tech |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent) |
| Ranking | LightGBM + SHAP |
| NLP | spaCy `en_core_web_sm` |
| UI | Streamlit |

---

## Setup

**1. Create and activate virtual environment**
```powershell
python -m venv .aienv
.\.aienv\Scripts\Activate.ps1
```

**2. Install dependencies**
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Configure environment**
```powershell
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**4. Train the ranking model**
```powershell
python -m src.ranking.train_model
```

---

## Run

```powershell
streamlit run streamlit_app/dashboard.py
```

Opens at `http://localhost:8501`

---

## Tests

```powershell
pytest tests/ -v
```

Run only unit tests (no PDF needed):
```powershell
pytest tests/unit/ -v
```

Run with coverage:
```powershell
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
airesumescreener/
├── src/
│   ├── config.py              # Central settings (pydantic-settings)
│   ├── exceptions.py          # Custom exception hierarchy
│   ├── logger.py              # Rotating file + console logging
│   ├── models.py              # Pydantic data models
│   ├── parsers/               # PDF + DOCX parsing
│   ├── extractor/             # Contact, entity, skill extraction
│   ├── preprocessing/         # Text cleaning, skill normalisation
│   ├── embeddings/            # SentenceTransformer embedding
│   ├── vector_store/          # ChromaDB client + resume index
│   ├── matching/              # Job description similarity search
│   ├── llm/                   # LLM client (Groq) + analyzers
│   ├── rag/                   # RAG pipeline
│   ├── ranking/               # LightGBM ranker + SHAP
│   └── explainability/        # SHAP explainer
├── streamlit_app/
│   └── dashboard.py           # Streamlit UI
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── unit/                  # Unit tests (no API calls)
│   └── integration/           # End-to-end pipeline tests
├── data/
│   ├── raw/                   # Training CSV
│   ├── sample_resumes/        # Test PDFs
│   └── chroma_db/             # ChromaDB persistence
├── models/                    # Serialised rank_model.pkl
├── logs/                      # Rotating log files
├── .env.example               # Environment template
├── pytest.ini
└── requirements.txt
```

---

## Key Bug Fixes (vs original)

| Bug | Fix |
|---|---|
| `TextCleaner` regex `[a-zA-Z\s]` stripped ALL letters | Changed to `[^a-zA-Z\s]` |
| ChromaDB rejected skills list in metadata | Skills joined to comma-separated string |
| `ChromaClient` was in-memory — data lost on restart | Switched to `PersistentClient` |
| `resume_parser.py` returned `ValueError` instead of raising | `raise ValueError(...)` |
| `SimilaritySearch.search()` had no `return` statement | Added `return results` |
| `shap.Explainer` fails for LightGBM | Changed to `shap.TreeExplainer` |
| `rank_candidate()` returned tuple but caller assigned to score directly | Callers now unpack `score, shap = ranker.rank_candidate(c)` |
| `RagPipeline` crashed on empty collection | Guard added, returns graceful message |
| Missing `__init__.py` in all `src/` subdirectories | All created |
| All test files had `if __name__ == "__main__"` blocks | Converted to proper `pytest` functions |
