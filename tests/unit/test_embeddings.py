"""Unit tests for embedding generation."""
import numpy as np
import pytest
from src.exceptions import EmbeddingError


class TestEmbeddingGenerator:
    def test_returns_numpy_array(self, embedder):
        vec = embedder.generate("Python machine learning engineer")
        assert isinstance(vec, np.ndarray)

    def test_embedding_has_nonzero_length(self, embedder):
        vec = embedder.generate("Some resume text")
        assert vec.shape[0] > 0

    def test_raises_on_empty_text(self, embedder):
        with pytest.raises(EmbeddingError, match="empty"):
            embedder.generate("")

    def test_raises_on_whitespace_only(self, embedder):
        with pytest.raises(EmbeddingError, match="empty"):
            embedder.generate("   ")

    def test_batch_returns_correct_count(self, embedder):
        texts = ["resume one", "resume two", "resume three"]
        results = embedder.generate_batch(texts)
        assert len(results) == 3

    def test_batch_empty_input(self, embedder):
        assert embedder.generate_batch([]) == []

    def test_different_texts_give_different_embeddings(self, embedder):
        v1 = embedder.generate("Python developer with ML experience")
        v2 = embedder.generate("Marketing manager with brand strategy")
        assert not np.allclose(v1, v2)
