class SimilaritySearch:
    def __init__(self, collection):
        self.collection = collection
        
    def search(self, query_embedding, top_k = 5):
        results  = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k
        )
        