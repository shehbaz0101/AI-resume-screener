import os

from src.parser.pdf_parser import PDFParser
from src.parser.docx_parser import DOCXParser

class ResumeParser:
    
    def __init__(self):
        
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        
    def parse(self, file_path):
        
            extension = os.path.splitext(file_path)[1]
            
            if extension == ".pdf":
                
                return self.pdf_parser.extract_text(file_path)
            
            elif extension == ".docx":
                
                return self.docx_parser.extract_text(file_path)
            else:
                
                return ValueError("unsupported file format")
            
            