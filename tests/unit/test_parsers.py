"""Unit tests for resume parsers."""
import pytest
from src.parsers.resume_parser import ResumeParser
from src.exceptions import UnsupportedFileFormatError


class TestResumeParser:
    def setup_method(self):
        self.parser = ResumeParser()

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            self.parser.parse("nonexistent_path/resume.pdf")

    def test_raises_unsupported_format(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("some content")
        with pytest.raises(UnsupportedFileFormatError, match="Unsupported format"):
            self.parser.parse(str(f))

    def test_parse_pdf(self, sample_resume_path):
        text = self.parser.parse(sample_resume_path)
        assert isinstance(text, str)
        assert len(text) > 100, "Expected non-trivial text from PDF"

    def test_parse_returns_string_type(self, sample_resume_path):
        result = self.parser.parse(sample_resume_path)
        assert isinstance(result, str)
