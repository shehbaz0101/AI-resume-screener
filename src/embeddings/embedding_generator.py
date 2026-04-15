import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.exceptions import EmbeddingError
#import logging
#from typing import List
#import numpy as np
#from src.exceptions import EmbeddingError
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def generate(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            vec = self.model.encode(text, show_progress_bar=False)
            logger.debug("Embedding shape: %s", vec.shape)
            return vec
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        try:
            return list(self.model.encode(texts, show_progress_bar=False, batch_size=32))
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc
