import os
from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings

FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def create_vector_store(chunks):
    embeddings = get_embeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_INDEX_DIR)
    return vector_db


def add_to_vector_store(chunks):
    """Add new document chunks to existing vector store, or create new one."""
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        vector_db = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        new_db = FAISS.from_documents(chunks, embeddings)
        vector_db.merge_from(new_db)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)

    vector_db.save_local(FAISS_INDEX_DIR)
    return vector_db


def load_vector_store():
    embeddings = get_embeddings()
    vector_db = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db


def vector_store_exists():
    return os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))