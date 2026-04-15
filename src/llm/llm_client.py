"""
LLM client wrapping Groq API.

Features:
- Exponential backoff retry on transient failures
- Rate limit detection with dedicated exception
- Temperature / max_tokens driven by central config
- Structured logging on every call
"""
import logging
import time
from groq import Groq, RateLimitError, APIError
from src.config import settings
from src.exceptions import LLMError, LLMRateLimitError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._max_retries = settings.llm_max_retries
        self._retry_wait = settings.llm_retry_wait_seconds

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text.

        Retries up to ``settings.llm_max_retries`` times with exponential
        backoff on transient API errors.

        Raises:
            LLMRateLimitError: On persistent rate limiting.
            LLMError: On any other failure after all retries.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "LLM call attempt %d/%d | model=%s | prompt_len=%d",
                    attempt, self._max_retries, self._model, len(prompt),
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                logger.debug("LLM response received: %d chars", len(content))
                return content

            except RateLimitError as exc:
                last_exc = exc
                wait = self._retry_wait * (2 ** (attempt - 1))
                logger.warning("Rate limit hit. Retrying in %.1fs (attempt %d).", wait, attempt)
                time.sleep(wait)

            except APIError as exc:
                last_exc = exc
                wait = self._retry_wait * (2 ** (attempt - 1))
                logger.warning("API error: %s. Retrying in %.1fs.", exc, wait)
                time.sleep(wait)

        if isinstance(last_exc, RateLimitError):
            raise LLMRateLimitError(
                f"Rate limit exceeded after {self._max_retries} retries."
            ) from last_exc

        raise LLMError(
            f"LLM call failed after {self._max_retries} retries: {last_exc}"
        ) from last_exc
