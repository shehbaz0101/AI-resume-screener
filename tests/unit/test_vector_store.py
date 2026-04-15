"""Unit tests for vector store layer."""
import numpy as np
import pytest
from src.vector_store.resume_index import ResumeIndex
from src.vector_store.similarity_search import SimilaritySearch
from src.exceptions import VectorStoreError, EmptyCollectionError


class TestResumeIndex:
    def test_add_resume_increments_count(self, empty_collection):
        idx = ResumeIndex.__new__(ResumeIndex)
        idx.collection = empty_collection
        vec = np.random.rand(384).tolist()
        idx.add_resume("test-001", vec, {"name": "Alice", "skills": ["python", "sql"]})
        assert idx.count() == 1

    def test_skills_list_stored_as_string(self, empty_collection):
        idx = ResumeIndex.__new__(ResumeIndex)
        idx.collection = empty_collection
        vec = np.random.rand(384).tolist()
        idx.add_resume("test-002", vec, {"name": "Bob", "skills": ["python", "pytorch"]})
        result = empty_collection.get(ids=["test-002"])
        stored_skills = result["metadatas"][0]["skills"]
        assert isinstance(stored_skills, str), "Skills must be stored as string"

    def test_upsert_updates_existing(self, empty_collection):
        idx = ResumeIndex.__new__(ResumeIndex)
        idx.collection = empty_collection
        vec = np.random.rand(384).tolist()
        idx.add_resume("test-003", vec, {"name": "Carol", "skills": "python"})
        idx.add_resume("test-003", vec, {"name": "Carol Updated", "skills": "python, sql"})
        assert idx.count() == 1  # still 1 — upserted, not duplicated

    def test_none_id_auto_generates(self, empty_collection):
        idx = ResumeIndex.__new__(ResumeIndex)
        idx.collection = empty_collection
        vec = np.random.rand(384).tolist()
        idx.add_resume(None, vec, {"name": "Dan", "skills": []})
        assert idx.count() == 1

    def test_empty_skills_default_to_general(self, empty_collection):
        idx = ResumeIndex.__new__(ResumeIndex)
        idx.collection = empty_collection
        vec = np.random.rand(384).tolist()
        idx.add_resume("test-004", vec, {"name": "Eve", "skills": []})
        result = empty_collection.get(ids=["test-004"])
        assert result["metadatas"][0]["skills"] == "general"


class TestSimilaritySearch:
    def test_raises_on_empty_collection(self, empty_collection):
        searcher = SimilaritySearch(empty_collection)
        with pytest.raises(EmptyCollectionError):
            searcher.search(np.random.rand(384).tolist())

    def test_returns_results_from_populated_collection(self, populated_collection):
        searcher = SimilaritySearch(populated_collection)
        results = searcher.search(np.random.rand(384).tolist(), top_k=2)
        assert results is not None
        assert len(results["ids"][0]) <= 2

    def test_top_k_capped_at_collection_size(self, populated_collection):
        searcher = SimilaritySearch(populated_collection)
        results = searcher.search(np.random.rand(384).tolist(), top_k=100)
        assert len(results["ids"][0]) <= populated_collection.count()
