"""
RankIntegration — load trained model and score a candidate.

Fixes vs original:
- rank_candidate() returns (score, shap_values) tuple — callers must unpack
- Skills stored as comma-separated string in ChromaDB — parsed back to list here
- model_path sourced from central config
- ModelNotFoundError raised with clear message if model missing
"""
import logging
from typing import Tuple

import joblib
import numpy as np

from src.config import settings
from src.ranking.feature_builder import FeatureBuilder
from src.explainability.shap_explainer import ShapExplainer
from src.exceptions import ModelNotFoundError, RankingError

logger = logging.getLogger(__name__)


class RankIntegration:
    def __init__(self, model_path: str = settings.rank_model_path) -> None:
        import os
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Rank model not found at '{model_path}'. "
                "Run: python -m src.ranking.train_model"
            )
        self._model = joblib.load(model_path)
        self._feature_builder = FeatureBuilder()
        self._explainer = ShapExplainer(self._model)
        self._job_skills: list[str] = settings.rank_job_skills
        logger.info("RankIntegration loaded model from '%s'", model_path)

    def rank_candidate(self, candidate_metadata: dict) -> Tuple[float, np.ndarray]:
        """Score a candidate against the trained ranking model.

        Args:
            candidate_metadata: Dict with keys 'skills', 'experience', 'projects'.
                                 Skills may be a comma-separated string (from ChromaDB)
                                 or a list.

        Returns:
            Tuple of (score: float, shap_values: np.ndarray).
            Callers must unpack: ``score, shap = ranker.rank_candidate(c)``

        Raises:
            RankingError: On any prediction failure.
        """
        try:
            # FIX: skills come back as "python, sql, ..." string from ChromaDB
            skills_raw = candidate_metadata.get("skills", "")
            if isinstance(skills_raw, list):
                skills = [s.lower().strip() for s in skills_raw]
            else:
                skills = [s.lower().strip() for s in skills_raw.split(",") if s.strip()]

            experience = float(candidate_metadata.get("experience", 0) or 0)
            projects   = float(candidate_metadata.get("projects",   0) or 0)
            skill_overlap = self._feature_builder.skill_overlap(skills, self._job_skills)

            features = np.array([[experience, projects, skill_overlap]])
            score = float(self._model.predict(features)[0])
            shap_values = self._explainer.explain(features)

            logger.debug(
                "Ranked candidate '%s': score=%.2f overlap=%.2f",
                candidate_metadata.get("name", "?"), score, skill_overlap,
            )
            return score, shap_values

        except Exception as exc:
            raise RankingError(f"Ranking failed for candidate: {exc}") from exc
