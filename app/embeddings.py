import os
import warnings
warnings.filterwarnings("ignore")

# Use the token from env if it exists
token = os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        multi_process=False, # Faster and avoids Windows issues
    )
    return embeddings