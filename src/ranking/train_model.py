"""
Training script for the LightGBM ranking model.

Usage:
    python -m src.ranking.train_model
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.ranking.rank_model import RankModel
from src.ranking.feature_builder import FeatureBuilder
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def train(
    data_path: str = os.path.join(settings.data_dir, "raw", "AI_Resume_Screening.csv"),
    output_path: str = settings.rank_model_path,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: '{data_path}'")

    logger.info("Loading training data from '%s'", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset shape: %s", df.shape)

    fb = FeatureBuilder()
    job_skills = settings.rank_job_skills

    X, y = [], []
    for _, row in df.iterrows():
        experience = float(row["Experience (Years)"])
        projects   = float(row["Projects Count"])
        resume_skills = [s.strip().lower() for s in str(row["Skills"]).split(",")]
        overlap = fb.skill_overlap(resume_skills, job_skills)
        X.append([experience, projects, overlap])
        y.append(float(row["AI Score (0-100)"]))

    X_arr, y_arr = np.array(X), np.array(y)
    logger.info("Training on %d samples with %d features", len(y_arr), X_arr.shape[1])

    model = RankModel()
    model.train(X_arr, y_arr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model.model, output_path)
    logger.info("Model saved to '%s'", output_path)
    print(f"✅ Model trained on {len(y_arr)} samples → saved to '{output_path}'")


if __name__ == "__main__":
    train()
