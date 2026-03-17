class FeatureBuilder:
    def skill_overlap(self,resume_skills, job_skills):
        overlap = set(resume_skills).intersection(set(job_skills))
        score = len(overlap) / max(len(job_skills), 1)
        
        return score