from src.parsers.resume_parser import ResumeParser
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer
from src.extractor.entity_extractor import EntityExtractor
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.skill_extractor import SkillExtractor
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.chroma_client import ChromaClient
from src.vector_store.resume_index import ResumeIndex
from src.matching.job_matcher import JobMatcher
from src.llm.llm_client import LLMClient
from src.llm.resume_analyzer import ResumeAnalyzer
from src.llm.candidate_explainer import CandidateExplainer
import uuid
resume_id = str(uuid.uuid4())
from src.ranking.rank_integration import RankIntegration 


class ResumePipeline:
    
    def __init__(self, file_path):
        
        self.cleaner = TextCleaner()
        self.normalizer = SkillNormalizer()
        self.file_path = file_path
        
        self.parser = ResumeParser()
        self.entity_extractor = EntityExtractor()
        self.contact_extractor = ContactExtractor()
        self.skill_extractor = SkillExtractor()
        self.embedding_model = EmbeddingModel().get_model()
        self.embedding_generator = EmbeddingGenerator(self.embedding_model)
        self.chroma_client = ChromaClient().get_client()
        self.resume_index = ResumeIndex(self.chroma_client)
        self.collection = self.chroma_client.get_or_create_collection(name = "resumes")
        self.matcher = JobMatcher(self.embedding_generator, self.collection)
        
        #rank intgration
        self.ranker = RankIntegration()
        
        #llms
        self.llm_client = LLMClient()
        self.resume_analyzer = ResumeAnalyzer(self.llm_client)
        self.explainer = CandidateExplainer(self.llm_client)
        
        
        
        
        
    def run(self):
        #here i am using resume parser
        text = self.parser.parse(self.file_path)

        clean_text = self.cleaner.clean(text)

        skills = self.skill_extractor.extract_skills(clean_text)

        skills = self.normalizer.normalize(skills)
        
        #llm analyzer
        llm_output = self.resume_analyzer.analyze(text)
        
        print("\n LLM Resume Analysis")
        print(llm_output)
        
        #here i am extracting information
        
        email = self.contact_extractor.extract_email(text)
        name = self.entity_extractor.extract_name(text)
        phone = self.contact_extractor.extract_phone(text)
        skill = self.skill_extractor.extract_skills(text)
        
        #embedding
        embedding = self.embedding_generator.generate(clean_text)
        
        #store in vector database
        self.resume_index.add_resume(
            resume_id = resume_id,
            embedding=embedding.tolist(),
            metadata={
                "name": name,
                "skills": ", ".join(skills) if skills else "None"
            }
        )
        result = {
            "email" : email,
            "name" : name,
            "phone" : phone,
            "skills" : skill,
            "embedding" : embedding
        }
        return result
    
    
if __name__ == "__main__":
    pipeline = ResumePipeline("data/sample_resumes/sufyanresume.pdf")
    output = pipeline.run()
    print("\n ===Resume  Extraction Result===\n")
    for key, value in output.items():
        print(f"{key} : {value}")




    job_description = """
    We are looking for a Machine Learning Engineer
    with experience in Python, Machine Learning,
    Deep Learning and SQL.
    """


    matcher = JobMatcher(
        pipeline.embedding_generator,
        pipeline.resume_index.collection
    )

    
    results = matcher.match(job_description)


    candidates = results["metadatas"][0]
    ranked_candidates = []
    ranker = RankIntegration()
    
    for candidate in candidates:
        score = ranker.rank_candidate(candidate)
        candidate["score"] = score
        ranked_candidates.append(candidate)
        
    #sort candidates by score
    ranked_candidates = sorted(
        ranked_candidates,
        key = lambda x: x["score"],
        reverse = True
    )
    
    
    print("\nFinal Ranked Candidates:\n")
    
    for c in ranked_candidates:
        explanation = pipeline.explainer.explain(c, job_description)
        
        c["explanation"] = explanation