import re

class TextCleaner:
    def __init__(self):
        pass
    def clean(self, text):
        text = text.lower()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
        
        