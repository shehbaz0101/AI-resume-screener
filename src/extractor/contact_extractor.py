import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ContactExtractor:
    _EMAIL = re.compile(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    _PHONE = re.compile(r"\+?[\d][\d\s\-().]{7,14}[\d]")

    def extract_email(self, text: str) -> Optional[str]:
        m = self._EMAIL.search(text)
        return m.group() if m else None

    def extract_phone(self, text: str) -> Optional[str]:
        m = self._PHONE.search(text)
        return m.group() if m else None
