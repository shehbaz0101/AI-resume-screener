"""
Central configuration — single source of truth.
All values load from environment variables / .env via pydantic-settings.
Import `settings` from this module everywhere. No hardcoded values in the codebase.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    groq_api_key: str = Field(..., description="Groq API key")
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = Field(2048, ge=64, le=8192)
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_retries: int = Field(3, ge=1, le=10)
    llm_retry_wait_seconds: float = Field(2.0, ge=0.5)

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    chroma_collection_name: str = "resumes"

    # RAG
    rag_top_k: int = Field(5, ge=1, le=20)

    # Ranking
    rank_model_path: str = str(BASE_DIR / "models" / "rank_model.pkl")
    rank_job_skills: list[str] = ["python", "machine learning", "sql", "deep learning"]

    # Paths
    data_dir: str = str(BASE_DIR / "data")
    models_dir: str = str(BASE_DIR / "models")
    logs_dir: str = str(BASE_DIR / "logs")

    # Logging
    log_level: str = "INFO"


settings = Settings()
