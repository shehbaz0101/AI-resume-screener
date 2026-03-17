import uuid
class ResumeIndex:
    def __init__(self, client):
        self.collection = client.get_or_create_collection(name = "resume")
        
    def add_resume(self, resume_id, embedding, metadata):
        self.collection.add(
            ids = [resume_id],
            embeddings = [embedding],
            metadatas = [metadata]
        )
        
        