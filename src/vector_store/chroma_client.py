import chromadb

class ChromaClient:
    def __init__(self):
        self.client = chromadb.Client()
    def get_client(self):
        return self.client