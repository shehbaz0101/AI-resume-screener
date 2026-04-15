"""
Shared pytest fixtures.

All fixtures use isolated in-memory state — no disk writes, no .env needed.
"""
import uuid
import numpy as np
import pytest
import chromadb


# ── Paths ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_resume_path():
    return "data/sample_resumes/sufyanresume.pdf"


@pytest.fixture(scope="session")
def sample_job_description():
    return (
        "We are looking for a Machine Learning Engineer with 3+ years of experience "
        "in Python, deep learning, SQL, and PyTorch."
    )


# ── In-memory ChromaDB (isolated per test session) ────────────────────────────

@pytest.fixture(scope="session")
def in_memory_client():
    return chromadb.Client()


@pytest.fixture
def empty_collection(in_memory_client):
    """Fresh empty collection per test."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    return in_memory_client.get_or_create_collection(name=name)


@pytest.fixture
def populated_collection(empty_collection):
    """Collection pre-loaded with 3 synthetic candidates."""
    candidates = [
        {"id": "c1", "name": "Alice Chen",  "skills": "python, machine learning, deep learning, sql"},
        {"id": "c2", "name": "Bob Smith",   "skills": "java, sql, data engineering"},
        {"id": "c3", "name": "Priya Patel", "skills": "python, pytorch, nlp, transformers"},
    ]
    rng = np.random.default_rng(42)
    for c in candidates:
        empty_collection.add(
            ids=[c["id"]],
            embeddings=[rng.random(384).tolist()],
            metadatas=[{"name": c["name"], "skills": c["skills"], "email": ""}],
        )
    return empty_collection


# ── Dummy LLM client ──────────────────────────────────────────────────────────

class DummyLLMClient:
    """Returns a deterministic string — no API calls in tests."""
    def generate(self, prompt: str) -> str:
        return f"[DUMMY LLM RESPONSE] Received prompt of {len(prompt)} chars."


@pytest.fixture
def dummy_llm():
    return DummyLLMClient()


# ── Embedding fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def embedding_model():
    from src.embeddings.embedding_model import EmbeddingModel
    return EmbeddingModel().get_model()


@pytest.fixture(scope="session")
def embedder(embedding_model):
    from src.embeddings.embedding_generator import EmbeddingGenerator
    return EmbeddingGenerator(embedding_model)
