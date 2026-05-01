import os
from dotenv import load_dotenv

load_dotenv()


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# -------- OPENAI --------
class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# -------- GEMINI --------
class GeminiLLM(BaseLLM):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-flash-latest")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


# -------- FACTORY --------
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "gemini":
        return GeminiLLM()
    return OpenAILLM()