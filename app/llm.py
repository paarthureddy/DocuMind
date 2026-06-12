import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

class AzureRESTLLM:
    """A drop-in replacement for LangChain ChatOllama targeting Azure OpenAI."""
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5-mini")
        
    def invoke(self, prompt: str):
        if not self.endpoint or not self.api_key:
            return self._response("Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.")

        base_url = self.endpoint.rstrip('/')
        url = f"{base_url}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        data = {
            "messages": [
                {"role": "user", "content": str(prompt)}
            ],
            "temperature": 0.3
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return self._response(res_data['choices'][0]['message']['content'])
        except Exception as e:
            print(f"AzureLLM Error: {e}")
            return self._response(f"Error: {str(e)}")

    def _response(self, content: str):
        class LLMResponse:
            def __init__(self, content):
                self.content = content

        return LLMResponse(content)

def get_llm():
    return AzureRESTLLM()
