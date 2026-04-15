"""
Custom exception hierarchy.
Catch specific exceptions instead of bare Exception everywhere.
"""


class ResumeScreenerError(Exception):
    """Base for all project exceptions."""


# Parsing
class ParseError(ResumeScreenerError):
    """Resume parsing failed."""

class UnsupportedFileFormatError(ParseError):
    """File extension not supported."""


# Embeddings
class EmbeddingError(ResumeScreenerError):
    """Embedding generation failed."""


# Vector store
class VectorStoreError(ResumeScreenerError):
    """Vector store operation failed."""

class EmptyCollectionError(VectorStoreError):
    """Query attempted on empty collection."""


# LLM
class LLMError(ResumeScreenerError):
    """LLM call failed after all retries."""

class LLMRateLimitError(LLMError):
    """LLM rate limit hit."""


# Ranking / Models
class ModelNotFoundError(ResumeScreenerError):
    """Serialised model file not found on disk."""

class RankingError(ResumeScreenerError):
    """Candidate ranking failed."""
