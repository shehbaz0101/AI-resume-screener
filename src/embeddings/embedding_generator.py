from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model):
        self.model = model
        
    def generate(self, text):
        embedding = self.model.encode(text)
        
        return embedding