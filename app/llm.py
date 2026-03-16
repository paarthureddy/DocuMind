from langchain_ollama import ChatOllama


def get_llm():
    llm = ChatOllama(
        model="llama3",
        temperature=0.3,
    )
    return llm