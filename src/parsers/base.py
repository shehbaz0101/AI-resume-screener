from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract raw text from file_path."""
