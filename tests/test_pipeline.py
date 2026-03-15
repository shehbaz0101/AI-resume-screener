from src.parsers.resume_parser import ResumeParser
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer
from src.extractor.entity_extractor import EntityExtractor
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.skill_extractor import SkillExtractor

class ResumePipeline:
    
    def __init__(self, file_path):
        self.cleaner = TextCleaner()
        self.normalizer = SkillNormalizer()
        self.file_path = file_path
        
        self.parser = ResumeParser()
        self.entity_extractor = EntityExtractor()
        self.contact_extractor = ContactExtractor()
        self.skill_extractor = SkillExtractor()
        
    def run(self):
        #here i am using resume parser
        text = self.parser.parse(self.file_path)
        text = self.parser.parse(self.file_path)

        clean_text = self.cleaner.clean(text)

        skills = self.skill_extractor.extract_skills(clean_text)

        skills = self.normalizer.normalize(skills)
        
        #here i am extracting information
        
        email = self.contact_extractor.extract_email(text)
        name = self.entity_extractor.extract_name(text)
        phone = self.contact_extractor.extract_phone(text)
        skills = self.skill_extractor.extract_skills(text)
        
        result = {
            "email" : email,
            "name" : name,
            "phone" : phone,
            "skills" : skills
        }
        return result
    
if __name__ == "__main__":
    pipeline = ResumePipeline("data/sample_resumes/sufyanresume.pdf")
    output = pipeline.run()
    print("\n ===Resume  Extraction Result===\n")
    
    for key, value in output.items():
        print(f"{key} : {value}")