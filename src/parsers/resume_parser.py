import logging
import os
from src.parsers.pdf_parser import PDFParser
from src.parsers.docx_parser import DOCXParser
from src.exceptions import UnsupportedFileFormatError, ParseError

logger = logging.getLogger(__name__)

_PARSERS = {".pdf": PDFParser, ".docx": DOCXParser}


class ResumeParser:
    def __init__(self) -> None:
        self._parsers = {ext: cls() for ext, cls in _PARSERS.items()}

    def parse(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: '{file_path}'")
        ext = os.path.splitext(file_path)[1].lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise UnsupportedFileFormatError(
                f"Unsupported format '{ext}'. Supported: {list(_PARSERS)}"
            )
        logger.info("Parsing resume: %s", os.path.basename(file_path))
        return parser.extract_text(file_path)
