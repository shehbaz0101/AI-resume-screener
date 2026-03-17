class ResumeAnalyzer:
    
    def __init__(self, llm_client):
        
        self.llm = llm_client
        
    def analyze(self, text):
        
        prompt = f"""
        Extract structured information from this resume.property
        
        Return JSON with:
        - name
        - skills 
        - years of experience
        - education
        - summary
        
        Resume :
        {text}
        """
        
        result = self.llm.generate(prompt)
        
        return result