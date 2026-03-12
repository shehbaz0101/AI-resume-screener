from src.parser.resume_parser import ResumeParser

parser = ResumeParser()

text = parser.parse("data/sample_resume.pdf")

print(text[:1000])