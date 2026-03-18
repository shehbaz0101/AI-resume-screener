from src.rag.rag_pipeline import RAGPipeline
from src.llm.llm_client import LLMClient
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.chroma_client import ChromaClient


if __name__ == "__main__":

    # Initialize embedding
    model = EmbeddingModel().get_model()
    embedder = EmbeddingGenerator(model)

    # Load vector DB
    chroma_client = ChromaClient().get_client()
    collection = chroma_client.get_or_create_collection(name="resumes")

    # LLM
    llm_client = LLMClient()

    # RAG
    rag = RAGPipeline(embedder, collection, llm_client)

    # 🔥 QUERY HERE
    query = """
    Looking for a Machine Learning Engineer
    with Python, Deep Learning, SQL skills
    """

    response = rag.run(query)

    print("\n===== RAG RESPONSE =====\n")
    print(response)