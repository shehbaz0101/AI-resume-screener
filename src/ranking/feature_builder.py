"""Feature engineering for the ranking model."""
from typing import List


class FeatureBuilder:
    def skill_overlap(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Jaccard-style overlap: |intersection| / |job_skills|.

        Returns a float in [0.0, 1.0].
        """
        if not job_skills:
            return 0.0
        resume_set = {s.lower().strip() for s in resume_skills}
        job_set = {s.lower().strip() for s in job_skills}
        return len(resume_set & job_set) / len(job_set)
