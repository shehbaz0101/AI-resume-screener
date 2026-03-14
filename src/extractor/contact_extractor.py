import re

class ContactExtractor:
    
    def extract_email(self, text):
        
        pattern = r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+"
        
        match= re.search(pattern, text)
        
        if match:
            return match.group()
        
        return None
    
    def extract_phone(self, text):
        
        pattern = r"\+?\d[\d -]{8,12}\d"
        
        match = re.search(pattern, text)
        
        if match:
            return match.group()
        
        return None
    