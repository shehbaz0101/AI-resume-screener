import logging
import re

logger = logging.getLogger(__name__)


class TextCleaner:
    """Normalise raw resume text before downstream processing."""

    _NEWLINES   = re.compile(r"\n+")
    _NON_ALPHA  = re.compile(r"[^a-zA-Z\s]")  # FIX: original [a-zA-Z\s] stripped ALL letters
    _WHITESPACE = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self._NEWLINES.sub(" ", text)
        text = self._NON_ALPHA.sub(" ", text)
        text = self._WHITESPACE.sub(" ", text)
        result = text.strip()
        logger.debug("Text cleaned: %d chars", len(result))
        return result
