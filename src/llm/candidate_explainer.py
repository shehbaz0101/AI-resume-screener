"""Use the LLM to explain why a candidate fits a job description."""
import logging
from src.exceptions import LLMError

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a professional AI hiring assistant.

Job Description:
{job_description}

Candidate Profile:
{candidate}

Write a concise (3-4 sentence) professional explanation of why this candidate
is or is not a good fit for this role. Be specific about matching and missing skills.
"""


class CandidateExplainer:
    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def explain(self, candidate: dict, job_description: str) -> str:
        """Return an LLM-generated fit explanation for a candidate.

        Args:
            candidate:       Metadata dict with 'name' and 'skills'.
            job_description: Raw job description text.

        Returns:
            Explanation string.
        """
        prompt = _PROMPT_TEMPLATE.format(
            job_description=job_description[:1000],
            candidate=str(candidate),
        )
        try:
            result = self._llm.generate(prompt)
            logger.debug("Explanation generated for: %s", candidate.get("name"))
            return result
        except Exception as exc:
            raise LLMError(f"Explanation generation failed: {exc}") from exc
