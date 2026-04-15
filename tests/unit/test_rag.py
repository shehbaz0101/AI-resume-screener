"""Unit tests for RAG pipeline."""
import numpy as np
import pytest
from src.rag.rag_pipeline import RagPipeline
from src.models import RagResponse
from src.exceptions import EmptyCollectionError


class TestRagPipeline:
    def test_returns_rag_response_type(self, populated_collection, embedder, dummy_llm):
        rag = RagPipeline(embedder, populated_collection, dummy_llm)
        result = rag.run("Looking for a Python ML engineer")
        assert isinstance(result, RagResponse)

    def test_response_has_llm_text(self, populated_collection, embedder, dummy_llm):
        rag = RagPipeline(embedder, populated_collection, dummy_llm)
        result = rag.run("Python developer with SQL skills")
        assert len(result.llm_response) > 0

    def test_candidates_retrieved_count(self, populated_collection, embedder, dummy_llm):
        rag = RagPipeline(embedder, populated_collection, dummy_llm)
        result = rag.run("Machine learning engineer")
        assert result.candidates_retrieved > 0

    def test_empty_collection_returns_graceful_message(self, empty_collection, embedder, dummy_llm):
        rag = RagPipeline(embedder, empty_collection, dummy_llm)
        result = rag.run("Any job query")
        assert result.candidates_retrieved == 0
        assert "no candidates" in result.llm_response.lower() or "upload" in result.llm_response.lower()

    def test_query_stored_in_response(self, populated_collection, embedder, dummy_llm):
        rag = RagPipeline(embedder, populated_collection, dummy_llm)
        query = "Senior data scientist"
        result = rag.run(query)
        assert result.query == query
