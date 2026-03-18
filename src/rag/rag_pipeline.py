class RagPipeline:
    
    def __init__(self, embedding_generator, collection, llm_client):
        self.embedding_generator = embedding_generator
        self.collection = collection
        self.llm_client = llm_client
        
    def retrieve(self, query, top_k = 5):
        query_embeddings = self.embedding_generator.generate(query)
        
        results = self.collection.query(
            query_embeddings = [query_embeddings.tolist()],
            n_results = top_k            
        )
        
        return results
    def generate_response(self, query, retrieved_data):
        
        candidates = retrieved_data["metadatas"][0]
        candidates_formatted = "\n".join([str(c) for c in candidates])
        
        prompt = f"""
        you are an AI hiring assistant
        
        job_description:
        {query}
        
        candidates:
        {candidates_formatted}
        
        -Select best candidates
        -Rank them
        -Explain why they are suitable
        Return clear structured answers
        """

        response =self.llm_client.generate(prompt)
        
        return response
    
    def run(self, query):
        retrieved = self.retrieve(query)
        response = self.generate_response(query, retrieved)
        
        return response
        
        
        