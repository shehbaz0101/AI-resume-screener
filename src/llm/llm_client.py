from groq import Groq
import os

from dotenv import load_dotenv

# This is the "magic" line that reads your .env file
load_dotenv() 

class LLMClient:
    def __init__(self):
        # Now os.getenv will actually find the key you hid from me!
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content