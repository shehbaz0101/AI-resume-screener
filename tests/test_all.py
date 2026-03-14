from src.parsers.resume_parser import ResumeParser

parser = ResumeParser()

text = parser.parse("data/sample_resumes/sufyanresume.pdf")

print(text[:3000])