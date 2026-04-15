"""Use the LLM to extract structured information from a raw resume."""
import logging
from src.exceptions import LLMError

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an expert resume parser. Extract structured information from the resume below.

Return ONLY valid JSON with these keys (no markdown, no preamble):
{{
  "name": string or null,
  "email": string or null,
  "skills": [list of strings],
  "years_of_experience": number or null,
  "education": string or null,
  "summary": string (2-3 sentences max)
}}

Resume:
{text}
"""


class ResumeAnalyzer:
    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def analyze(self, text: str) -> str:
        """Return LLM-extracted structured JSON string for the given resume text.

        Args:
            text: Raw or lightly cleaned resume text.

        Returns:
            JSON string with extracted fields.
        """
        if not text.strip():
            raise ValueError("Cannot analyse empty resume text.")
        prompt = _PROMPT_TEMPLATE.format(text=text[:4000])  # stay within context window
        logger.info("Analysing resume with LLM (text length=%d)", len(text))
        try:
            return self._llm.generate(prompt)
        except Exception as exc:
            raise LLMError(f"Resume analysis failed: {exc}") from exc
