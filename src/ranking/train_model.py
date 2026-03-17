import pandas as pd
import joblib

from src.ranking.rank_model import RankModel
from src.ranking.feature_builder import FeatureBuilder

def train():
    df = pd.read_csv("data/raw/AI_Resume_Screening.csv")

    feature_builder = FeatureBuilder()
    
    X = []
    y = []
    
    job_skills = ["python", "machine learning", "sql", "deep learning"]
    
    for _, row in df.iterrows():
        
        #extract features
        
        experience = row["Experience (Years)"]
        projects = row["Projects Count"]
        
        #convert skills string to list
        resume_skills = [s.strip().lower() for s in str(row["Skills"]).split(",")]
        
        #skill overlap
        skill_overlap = feature_builder.skill_overlap(resume_skills, job_skills)
        
        X.append([experience, projects, skill_overlap])
        
        y.append(row["AI Score (0-100)"])
        
        #Train model
    model = RankModel()
    model.train(X,y)
        
        #save model
        
    joblib.dump(model.model, "models/rank_model.pkl")
        
    print(" Model trained and saved successfully")
        
if __name__ == "__main__":
    
    train()

        
    
    