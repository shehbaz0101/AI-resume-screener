import joblib
from src.ranking.feature_builder import FeatureBuilder

class RankIntegration:
    def __init__(self):
        
        self.model = joblib.load("models/rank_model.pkl")
        self.feature_builder = FeatureBuilder()
        
        #example job requirement
        
        self.job_skills = ["python", "machine learning", "sql", "deep leanrning"]
    
    def rank_candidate(self, candidate_metadata):
        
        #Extract candidate data
        
        skills = candidate_metadata.get("skills", [])
        experience = candidate_metadata.get("experience", 0)
        projects = candidate_metadata.get("projects", 0)
        
        #feature engineering
        
        skill_overlap = self.feature_builder.skill_overlap(
            skills, self.job_skills
        )
        
        features = [[experience, projects, skill_overlap]]
        
        # predict score
        
        score = self.model.predict(features)[0]
        
        return score