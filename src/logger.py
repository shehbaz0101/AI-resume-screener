"""
Logging configuration.
Call setup_logging() once at entry point (dashboard.py / scripts).
All other modules: logger = logging.getLogger(__name__)
"""
import logging
import logging.handlers
import os
from src.config import settings


def setup_logging() -> None:
    os.makedirs(settings.logs_dir, exist_ok=True)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    fh = logging.handlers.RotatingFileHandler(
        os.path.join(settings.logs_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for name in ("httpx", "chromadb", "sentence_transformers", "groq"):
        logging.getLogger(name).setLevel(logging.WARNING)
