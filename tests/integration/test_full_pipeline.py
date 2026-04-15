"""
Integration test — full pipeline from PDF parse → index → RAG query.

Requires:
- data/sample_resumes/sufyanresume.pdf to exist
- No LLM API calls (uses DummyLLMClient)
"""
import pytest
from src.parsers.resume_parser import ResumeParser
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.entity_extractor import EntityExtractor
from src.extractor.skill_extractor import SkillExtractor
from src.vector_store.resume_index import ResumeIndex
from src.rag.rag_pipeline import RagPipeline
from src.models import RagResponse
import uuid


class TestFullPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, empty_collection, embedder, dummy_llm):
        self.collection = empty_collection
        self.embedder = embedder
        self.llm = dummy_llm
        self.parser = ResumeParser()
        self.cleaner = TextCleaner()
        self.normalizer = SkillNormalizer()
        self.contact_ext = ContactExtractor()
        self.entity_ext = EntityExtractor()
        self.skill_ext = SkillExtractor()
        self.index = ResumeIndex.__new__(ResumeIndex)
        self.index.collection = self.collection

    def test_parse_extract_index_query(self, sample_resume_path, sample_job_description):
        # 1. Parse
        raw_text = self.parser.parse(sample_resume_path)
        assert len(raw_text) > 50

        # 2. Extract
        clean_text = self.cleaner.clean(raw_text)
        name = self.entity_ext.extract_name(raw_text)
        email = self.contact_ext.extract_email(raw_text)
        skills = self.skill_ext.extract_skills(clean_text)
        skills = self.normalizer.normalize(skills)

        # 3. Embed + index
        embedding = self.embedder.generate(clean_text)
        self.index.add_resume(
            resume_id=str(uuid.uuid4()),
            embedding=embedding.tolist(),
            metadata={"name": name, "skills": skills, "email": email},
        )
        assert self.index.count() == 1

        # 4. RAG query
        rag = RagPipeline(self.embedder, self.collection, self.llm)
        result = rag.run(sample_job_description)

        assert isinstance(result, RagResponse)
        assert result.candidates_retrieved == 1
        assert len(result.llm_response) > 0

    def test_multiple_resumes_ranked(self, sample_resume_path, sample_job_description):
        """Index the same resume twice under different IDs, verify 2 retrieved."""
        raw_text = self.parser.parse(sample_resume_path)
        clean_text = self.cleaner.clean(raw_text)
        skills = self.normalizer.normalize(self.skill_ext.extract_skills(clean_text))
        embedding = self.embedder.generate(clean_text)

        for i in range(2):
            self.index.add_resume(
                resume_id=f"candidate-{i}",
                embedding=embedding.tolist(),
                metadata={"name": f"Candidate {i}", "skills": skills},
            )

        rag = RagPipeline(self.embedder, self.collection, self.llm)
        result = rag.run(sample_job_description)
        assert result.candidates_retrieved == 2
