"""
RAG pipeline: retrieve candidates from vector store → generate LLM ranking.

Flow:
    query → embed → ChromaDB similarity search → build prompt → LLM → response
"""
import logging
from src.config import settings
from src.exceptions import EmptyCollectionError, LLMError
from src.models import RagResponse

logger = logging.getLogger(__name__)

_RANKING_PROMPT = """\
You are an expert AI hiring assistant. Your job is to rank candidates for a role.

Job Description:
{query}

Retrieved Candidates:
{candidates}

Instructions:
1. Rank all candidates from most to least suitable.
2. For each candidate provide:
   - Rank number
   - Name
   - Key matching skills
   - One sentence on why they fit (or don't fit)
3. Be concise, structured, and professional.
4. If no candidates were provided, say so clearly.

Provide your ranked list below:
"""


class RagPipeline:
    def __init__(self, embedding_generator, collection, llm_client) -> None:
        self._embedder = embedding_generator
        self._collection = collection
        self._llm = llm_client

    def retrieve(self, query: str, top_k: int = settings.rag_top_k) -> dict:
        """Embed query and fetch top-k candidates from vector store.

        Raises:
            EmptyCollectionError: If no resumes are indexed.
        """
        count = self._collection.count()
        if count == 0:
            raise EmptyCollectionError(
                "No resumes indexed. Upload resumes before running a search."
            )
        actual_k = min(top_k, count)
        query_vec = self._embedder.generate(query)
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=actual_k,
        )
        logger.info(
            "RAG retrieved %d candidates for query (len=%d)",
            len(results["ids"][0]),
            len(query),
        )
        return results

    def _format_candidates(self, retrieved: dict) -> str:
        """Format raw ChromaDB results into a readable prompt block."""
        candidates = retrieved.get("metadatas", [[]])[0]
        if not candidates:
            return "No candidates found."
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"  {i}. Name: {c.get('name', 'Unknown')} "
                f"| Skills: {c.get('skills', 'N/A')}"
            )
        return "\n".join(lines)

    def generate_response(self, query: str, retrieved: dict) -> str:
        """Build a prompt from retrieved candidates and call the LLM."""
        candidates_block = self._format_candidates(retrieved)
        prompt = _RANKING_PROMPT.format(
            query=query[:1500],
            candidates=candidates_block,
        )
        try:
            return self._llm.generate(prompt)
        except Exception as exc:
            raise LLMError(f"RAG response generation failed: {exc}") from exc

    def run(self, query: str) -> RagResponse:
        """End-to-end RAG: retrieve + generate.

        Returns:
            RagResponse with query, count, and llm_response fields.
        """
        try:
            retrieved = self.retrieve(query)
        except EmptyCollectionError:
            return RagResponse(
                query=query,
                candidates_retrieved=0,
                llm_response=(
                    "⚠ No candidates found in the database. "
                    "Please upload resumes before running a search."
                ),
            )

        n = len(retrieved.get("ids", [[]])[0])
        response_text = self.generate_response(query, retrieved)

        return RagResponse(
            query=query,
            candidates_retrieved=n,
            llm_response=response_text,
        )
