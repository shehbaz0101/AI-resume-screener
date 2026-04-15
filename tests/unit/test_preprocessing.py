"""Unit tests for text cleaning and skill normalisation."""
import pytest
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_lowercases_text(self):
        assert self.cleaner.clean("Hello World") == "hello world"

    def test_removes_non_alpha(self):
        result = self.cleaner.clean("Python 3.9, TensorFlow!")
        assert "3" not in result
        assert "!" not in result
        assert "," not in result

    def test_preserves_letters(self):
        # Critical regression test for the [a-zA-Z\s] → [^a-zA-Z\s] bug fix
        result = self.cleaner.clean("Python machine learning")
        assert "python" in result
        assert "machine" in result
        assert "learning" in result

    def test_collapses_whitespace(self):
        result = self.cleaner.clean("hello    world\n\nfoo")
        assert "  " not in result

    def test_empty_string(self):
        assert self.cleaner.clean("") == ""

    def test_strips_leading_trailing(self):
        assert self.cleaner.clean("  hello  ") == "hello"


class TestSkillNormalizer:
    def setup_method(self):
        self.norm = SkillNormalizer()

    def test_maps_pytorch_to_deep_learning(self):
        result = self.norm.normalize(["pytorch"])
        assert "deep learning" in result

    def test_maps_sklearn_to_machine_learning(self):
        result = self.norm.normalize(["sklearn"])
        assert "machine learning" in result

    def test_deduplicates(self):
        result = self.norm.normalize(["python", "python", "Python"])
        assert result.count("python") == 1

    def test_preserves_unknown_skills(self):
        result = self.norm.normalize(["kubernetes"])
        assert "kubernetes" in result

    def test_empty_list(self):
        assert self.norm.normalize([]) == []
