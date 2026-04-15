import logging
from typing import Optional
from src.exceptions import EmptyCollectionError

logger = logging.getLogger(__name__)


class SimilaritySearch:
    def __init__(self, collection) -> None:
        self.collection = collection

    def search(self, query_embedding: list, top_k: int = 5) -> Optional[dict]:
        count = self.collection.count()
        if count == 0:
            raise EmptyCollectionError(
                "No resumes indexed yet. Upload at least one resume first."
            )
        actual_k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
        )
        logger.debug("Similarity search returned %d results", len(results["ids"][0]))
        return results
