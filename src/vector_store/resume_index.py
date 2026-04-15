"""
ResumeIndex — wraps ChromaDB collection with safe add/upsert logic.

Key fixes vs original:
- Skills stored as comma-joined STRING (ChromaDB rejects lists)
- Upsert semantics: update if ID exists, add if new (prevents crash on duplicate upload)
- All metadata values explicitly cast to str
"""
import logging
import uuid
from typing import Optional
from src.config import settings
from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class ResumeIndex:
    def __init__(self, client) -> None:
        self.collection = client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        logger.info(
            "ResumeIndex ready. Collection '%s' has %d docs.",
            settings.chroma_collection_name,
            self.collection.count(),
        )

    def add_resume(
        self,
        resume_id: Optional[str],
        embedding: list,
        metadata: dict,
    ) -> None:
        """Upsert a resume into the vector store.

        Args:
            resume_id: Unique string ID. Auto-generated if None.
            embedding:  List of floats (dense vector).
            metadata:   Dict with candidate info. Skills must be a list[str] or str.
        """
        doc_id = str(resume_id) if resume_id else str(uuid.uuid4())

        # FIX: ChromaDB metadata only accepts scalar types — join list to string
        skills_raw = metadata.get("skills", [])
        if isinstance(skills_raw, list):
            skills_str = ", ".join(str(s) for s in skills_raw) or "general"
        else:
            skills_str = str(skills_raw).strip() or "general"

        clean_meta = {
            "name":   str(metadata.get("name") or "unknown"),
            "skills": skills_str,
            "email":  str(metadata.get("email") or ""),
        }

        try:
            existing = self.collection.get(ids=[doc_id])
            if existing["ids"]:
                self.collection.update(
                    ids=[doc_id], embeddings=[embedding], metadatas=[clean_meta]
                )
                logger.info("Updated resume: %s", doc_id)
            else:
                self.collection.add(
                    ids=[doc_id], embeddings=[embedding], metadatas=[clean_meta]
                )
                logger.info("Added resume: %s | name=%s | skills=%s",
                            doc_id, clean_meta["name"], skills_str[:80])
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert resume '{doc_id}': {exc}") from exc

    def count(self) -> int:
        return self.collection.count()

    def get_all(self) -> dict:
        return self.collection.get()

    def delete(self, resume_id: str) -> None:
        self.collection.delete(ids=[resume_id])
        logger.info("Deleted resume: %s", resume_id)
