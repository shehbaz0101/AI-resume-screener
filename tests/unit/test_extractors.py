"""Unit tests for contact, entity, and skill extractors."""
import pytest
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.skill_extractor import SkillExtractor


class TestContactExtractor:
    def setup_method(self):
        self.extractor = ContactExtractor()

    def test_extracts_email(self):
        text = "Contact me at john.doe@example.com for more info."
        assert self.extractor.extract_email(text) == "john.doe@example.com"

    def test_returns_none_no_email(self):
        assert self.extractor.extract_email("No email here.") is None

    def test_extracts_phone(self):
        text = "Call me at +1 234 567 8901"
        result = self.extractor.extract_phone(text)
        assert result is not None
        assert "234" in result

    def test_returns_none_no_phone(self):
        assert self.extractor.extract_phone("No phone here.") is None


class TestSkillExtractor:
    def setup_method(self):
        self.extractor = SkillExtractor()

    def test_extracts_python(self):
        assert "python" in self.extractor.extract_skills("Proficient in Python and SQL")

    def test_extracts_sql(self):
        assert "sql" in self.extractor.extract_skills("experience with SQL databases")

    def test_case_insensitive(self):
        assert "python" in self.extractor.extract_skills("PYTHON developer")

    def test_extracts_multi_word_skill(self):
        assert "machine learning" in self.extractor.extract_skills(
            "worked on machine learning projects"
        )

    def test_returns_empty_for_no_match(self):
        assert self.extractor.extract_skills("I like cooking and hiking") == []

    def test_no_duplicates(self):
        result = self.extractor.extract_skills("python python python")
        assert result.count("python") == 1
