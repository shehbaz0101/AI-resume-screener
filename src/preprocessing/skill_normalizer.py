import logging
from typing import List

logger = logging.getLogger(__name__)

_SKILL_MAP: dict[str, str] = {
    "python3": "python", "python programming": "python",
    "tensorflow": "deep learning", "pytorch": "deep learning",
    "scikit learn": "machine learning", "sklearn": "machine learning",
    "scikit-learn": "machine learning", "nlp": "natural language processing",
    "cv": "computer vision", "ml": "machine learning", "dl": "deep learning",
}


class SkillNormalizer:
    def normalize(self, skills: List[str]) -> List[str]:
        normalised = {
            _SKILL_MAP.get(s.lower().strip(), s.lower().strip())
            for s in skills if s.strip()
        }
        result = sorted(normalised)
        logger.debug("Normalised %d → %d skills", len(skills), len(result))
        return result
