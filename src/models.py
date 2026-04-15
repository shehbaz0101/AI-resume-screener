"""
Pydantic data models — enforce types and validation at every pipeline boundary.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    """Parsed and extracted data from a single resume."""
    resume_id: str
    raw_text: str
    clean_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None


class CandidateMatch(BaseModel):
    """A single candidate returned from vector similarity search."""
    name: str = "Unknown"
    skills: str = ""
    distance: float = 0.0
    rank_score: Optional[float] = None
    explanation: Optional[str] = None


class RagResponse(BaseModel):
    """Final structured output from the RAG pipeline."""
    query: str
    candidates_retrieved: int
    llm_response: str
