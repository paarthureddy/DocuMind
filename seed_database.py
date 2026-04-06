import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from document_loader import load_and_split_document
from vector_store import create_vector_store

def seed():
    dataset_path = "data/1500_profiles.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print("Loading profiles...")
    chunks = load_and_split_document(dataset_path)
    print(f"Loaded {len(chunks)} profiles.")
    
    print("Creating vector embeddings... (This might take a minute)")
    create_vector_store(chunks)
    print("Successfully built the FAISS index for all profiles!")

if __name__ == "__main__":
    seed()
