import logging
from typing import List

logger = logging.getLogger(__name__)

# Longest-first so multi-word phrases match before substrings
_TAXONOMY: list[str] = sorted([
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "scala", "r", "swift", "kotlin", "php", "ruby", "matlab",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "transfer learning",
    "neural networks", "transformers", "large language models",
    "generative ai", "fine-tuning", "rag",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
    "lightgbm", "hugging face", "langchain", "spacy", "nltk",
    "pandas", "numpy", "scipy", "matplotlib", "plotly",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "apache spark", "hadoop", "dbt", "apache airflow", "kafka",
    "aws", "gcp", "azure", "docker", "kubernetes", "git", "ci/cd",
    "mlflow", "wandb", "terraform",
    "rest api", "fastapi", "flask", "django", "streamlit",
    "data science", "data engineering", "feature engineering",
    "statistics", "a/b testing", "time series", "forecasting",
], key=len, reverse=True)


class SkillExtractor:
    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = sorted({s for s in _TAXONOMY if s in text_lower})
        logger.debug("Found %d skills", len(found))
        return found
