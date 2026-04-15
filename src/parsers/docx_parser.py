import logging
from docx import Document
from src.parsers.base import BaseParser
from src.exceptions import ParseError

logger = logging.getLogger(__name__)


class DOCXParser(BaseParser):
    def extract_text(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            logger.debug("DOCX parsed: %d chars from '%s'", len(text), file_path)
            return text
        except Exception as exc:
            raise ParseError(f"DOCX parse failed '{file_path}': {exc}") from exc
