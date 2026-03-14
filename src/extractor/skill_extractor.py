class SkillExtractor:
    def __init__(self):
        self.skills = {
            "python",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "sql",
            "nlp",
            "data science"
        }
        
    def extract_skills(self, text):
        text = text.lower()
        found = []
        
        for skill in self.skills:
            if skill in text:
                found.append(skill)
                
        return found
                
        