import logging
from functools import lru_cache
from typing import Optional
import spacy

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_nlp():
    return spacy.load("en_core_web_sm")


class EntityExtractor:
    def __init__(self) -> None:
        self.nlp = _load_nlp()

    def extract_name(self, text: str) -> Optional[str]:
        """Return first PERSON entity from first 500 chars (name is always near top)."""
        doc = self.nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None
