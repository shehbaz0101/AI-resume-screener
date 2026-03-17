class JobMatcher:
    def __init__(self, embedding_generator, collection):
        self.embedding_generator = embedding_generator
        self.collection = collection
        
    def match(self, job_description, top_k = 5):
        #convert job description to embedding
        job_embedding = self.embedding_generator.generate(job_description)
        results = self.collection.query(
            query_embeddings = [job_embedding.tolist()],
            n_results =top_k
        )
        return results        