import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

def get_llm():
    # Priority: ollama model from env > default
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    llm = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.3,
        num_ctx=1024,  # Reduced memory footprint for faster local responses
        num_predict=512,
        top_p=0.9,
    )
    return llm