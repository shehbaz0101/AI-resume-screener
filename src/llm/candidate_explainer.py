class CandidateExplainer:
    
    def __init__(self,llm_client):
        self.llm = llm_client
        
    def explain(self, candidate, job_description):
        prompt = f"""
        Explain why this candidate is suitable for the job.
        
        Job Description:
        {job_description}
        
        Candidate Info:
        {candidate}
        
        Give a short professional explanation.
        """
        
        return self.llm.generate(prompt)