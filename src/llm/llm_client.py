from openai import OpenAI
import os
from dotenv import load_dotenv

# This is the "magic" line that reads your .env file
load_dotenv() 

class LLMClient:
    def __init__(self):
        # Now os.getenv will actually find the key you hid from me!
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content