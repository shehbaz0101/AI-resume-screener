"""Vector similarity matching between job descriptions and indexed resumes."""
import logging
from src.config import settings
from src.exceptions import EmptyCollectionError

logger = logging.getLogger(__name__)


class JobMatcher:
    def __init__(self, embedding_generator, collection) -> None:
        self._embedder = embedding_generator
        self._collection = collection

    def match(self, job_description: str, top_k: int = settings.rag_top_k) -> dict:
        """Return top-k resume matches for a job description.

        Args:
            job_description: Raw job description text.
            top_k:           Number of candidates to return.

        Returns:
            ChromaDB query result dict with metadatas, distances, ids.

        Raises:
            EmptyCollectionError: If no resumes are indexed.
        """
        count = self._collection.count()
        if count == 0:
            raise EmptyCollectionError(
                "No resumes indexed yet. Upload at least one resume first."
            )

        actual_k = min(top_k, count)
        job_embedding = self._embedder.generate(job_description)

        results = self._collection.query(
            query_embeddings=[job_embedding.tolist()],
            n_results=actual_k,
        )
        logger.info(
            "JobMatcher: matched %d candidates for query (len=%d)",
            len(results["ids"][0]),
            len(job_description),
        )
        return results
