import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from vector_store import load_vector_store
from llm import get_llm

# Load vector database
vector_db = load_vector_store()

# Load LLM
llm = get_llm()

query = input("Ask a question: ")

# Retrieve relevant chunks
docs = vector_db.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAI Answer:\n")
print(response.content)