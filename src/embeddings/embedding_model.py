import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


class EmbeddingModel:
    def __init__(self, model_name: str = settings.embedding_model_name) -> None:
        self._name = model_name

    def get_model(self) -> SentenceTransformer:
        return _load_model(self._name)
