from docx import Document

class DOCXParser :
    def extract_text(self, file_path):
        
        doc = Document(file_path)
        
        text = []
        
        for paragraph in doc.paragraphs :
            
            text.append(paragraph.text)
            
        return "\n".join(text)
    