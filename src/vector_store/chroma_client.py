import logging
import os
import chromadb
from src.config import settings

logger = logging.getLogger(__name__)


class ChromaClient:
    def __init__(self, persist_dir: str = settings.chroma_persist_dir) -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB initialised at: %s", persist_dir)

    def get_client(self) -> chromadb.PersistentClient:
        return self._client
