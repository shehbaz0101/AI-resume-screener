import streamlit as st

from src.parsers.resume_parser import ResumeParser
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.entity_extractor import EntityExtractor
from src.extractor.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.chroma_client import ChromaClient
from src.vector_store.resume_index import ResumeIndex
from src.rag.rag_pipeline import RagPipeline
from src.llm.llm_client import LLMClient

import tempfile


# Initialize components
parser = ResumeParser()
contact = ContactExtractor()
entity = EntityExtractor()
skills_extractor = SkillExtractor()
cleaner = TextCleaner()
normalizer = SkillNormalizer()

model = EmbeddingModel().get_model()
embedder = EmbeddingGenerator(model)

chroma_client = ChromaClient().get_client()
collection = chroma_client.get_or_create_collection(name="resumes")
resume_index = ResumeIndex(chroma_client)

llm_client = LLMClient()
rag = RagPipeline(embedder, collection, llm_client)


# UI Title
st.title("🚀 AI Resume Screener")


# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])


if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Parse
    text = parser.parse(file_path)

    # Clean
    clean_text = cleaner.clean(text)

    # Extract
    name = entity.extract_name(clean_text)
    email = contact.extract_email(clean_text)
    extracted_skills = skills_extractor.extract_skills(clean_text)
    extracted_skills = normalizer.normalize(extracted_skills)

    # Embedding
    embedding = embedder.generate(clean_text)

    # Store in vector DB
    resume_index.add_resume(
        resume_id=name if name else "unknown",
        embedding=embedding.tolist(),
        metadata={
            "name": name,
            "skills": extracted_skills
        }
    )

    st.success(f"Resume processed for: {name}")

    st.write("Skills:", extracted_skills)


# Job Description Input
job_description = st.text_area("Enter Job Description")


if st.button("Find Best Candidates"):

    if job_description:

        response = rag.run(job_description)

        st.subheader("🎯 Top Candidates")
        st.write(response)

    else:
        st.warning("Please enter job description")