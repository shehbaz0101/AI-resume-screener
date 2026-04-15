import logging
import fitz
from src.parsers.base import BaseParser
from src.exceptions import ParseError

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    def extract_text(self, file_path: str) -> str:
        try:
            parts: list[str] = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    parts.append(page.get_text())
            text = "\n".join(parts)
            logger.debug("PDF parsed: %d chars from '%s'", len(text), file_path)
            return text
        except Exception as exc:
            raise ParseError(f"PDF parse failed '{file_path}': {exc}") from exc
