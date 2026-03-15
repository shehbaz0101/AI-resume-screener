class SkillNormalizer:
    def __init__(self):
        self.skill_map = {"python3": "python",
            "python programming": "python",
            "tensorflow": "deep learning",
            "pytorch": "deep learning",
            "scikit learn": "machine learning",
            "sklearn": "machine learning",
            "nlp": "natural language processing"
        }
    def normalize(self, skills):

        normalized = []

        for skill in skills:

            skill = skill.lower()

            if skill in self.skill_map:
                normalized.append(self.skill_map[skill])
            else:
                normalized.append(skill)

        return list(set(normalized))