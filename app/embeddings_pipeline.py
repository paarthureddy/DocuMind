"""
Multi-embedding pipeline for talent search engine.
Handles generation and storage of embeddings for semantic search.
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL, 
    EMBEDDING_DIMENSION, 
    FAISS_INDEX_DIR,
    PROCESSED_PROFILES_PATH
)

class EmbeddingsPipeline:
    """Manages embeddings generation and FAISS vector storage."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            multi_process=False,
        )
        self.vector_store = None
        self.profile_metadata = {}
    
    def create_profile_document(self, profile: Dict[str, Any]) -> Document:
        """Create a LangChain document from a profile."""
        past_roles = profile.get("past_roles", [])
        subcrafts = profile.get("subcrafts", [])
        crafts = [profile.get("craft", "")] + subcrafts
        crafts = [c for c in crafts if c]
        appearance = profile.get("appearance_tags_enhanced", []) + profile.get("appearance_tags", [])
        
        formatted_prefix = f"Crafts: {', '.join(crafts)}. Past Roles: {', '.join(past_roles)}. Appearance: {', '.join(appearance)}."
        
        # Combine all searchable text
        searchable_text = [
            formatted_prefix,
            profile.get("semantic_profile", ""),
            " ".join(profile.get("casting_tags", [])),
            " ".join(profile.get("personality_tags", [])),
            " ".join(profile.get("skills", [])),
            profile.get("name", "")
        ]
        
        content = " ".join(filter(None, searchable_text))
        
        # Create metadata for filtering and ranking
        metadata = {
            "profile_id": profile.get("id"),
            "name": profile.get("name", ""),
            "gender": profile.get("gender", ""),
            "height_cm": profile.get("height_info", {}).get("height_cm"),
            "height_bucket": profile.get("height_info", {}).get("height_bucket"),
            "complexion": profile.get("complexion_standard", ""),
            "craft": profile.get("craft", ""),
            "experience_years": profile.get("experience_years", 0),
            "rating_average": profile.get("rating", {}).get("average", 0),
            "engagement_rate": profile.get("creator_metrics", {}).get("engagement_rate", 0),
            "followers": profile.get("creator_metrics", {}).get("followers", 0),
            "casting_tags": profile.get("casting_tags", []),
            "personality_tags": profile.get("personality_tags", []),
            "skills": profile.get("skills", []),
            "semantic_profile": profile.get("semantic_profile", ""),
            "past_roles": profile.get("past_roles", []),
            "subcrafts": profile.get("subcrafts", [])
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def generate_embeddings_batch(self, profiles: List[Dict[str, Any]]) -> List[Document]:
        """Generate embeddings for a batch of profiles."""
        documents = []
        
        print(f"Creating documents for {len(profiles)} profiles...")
        for i, profile in enumerate(profiles):
            if i % 50 == 0:
                print(f"Processing document {i+1}/{len(profiles)}")
            
            try:
                doc = self.create_profile_document(profile)
                documents.append(doc)
                
                # Store profile metadata for later access
                self.profile_metadata[profile.get("id")] = profile
                
            except Exception as e:
                print(f"Error creating document for profile {i}: {e}")
                continue
        
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        print(f"Creating FAISS vector store with {len(documents)} documents...")
        
        if not documents:
            raise ValueError("No documents to create vector store")
        
        # Create FAISS index
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        vector_store.save_local(str(FAISS_INDEX_DIR))
        
        # Save metadata separately
        metadata_path = FAISS_INDEX_DIR / "profile_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.profile_metadata, f)
        
        print(f"Vector store saved to {FAISS_INDEX_DIR}")
        self.vector_store = vector_store
        
        return vector_store
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load existing FAISS vector store."""
        if not FAISS_INDEX_DIR.exists():
            return None
        
        try:
            # Load FAISS index
            vector_store = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = FAISS_INDEX_DIR / "profile_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.profile_metadata = pickle.load(f)
            
            print(f"Loaded vector store with {len(self.profile_metadata)} profiles")
            self.vector_store = vector_store
            return vector_store
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def vector_store_exists(self) -> bool:
        """Check if vector store exists."""
        return FAISS_INDEX_DIR.exists() and (FAISS_INDEX_DIR / "index.faiss").exists()
    
    def build_from_profiles(self, profiles: List[Dict[str, Any]]) -> FAISS:
        """Build complete vector store from profiles."""
        documents = self.generate_embeddings_batch(profiles)
        return self.create_vector_store(documents)
    
    def build_from_processed_file(self) -> Optional[FAISS]:
        """Build vector store from processed profiles file."""
        if not PROCESSED_PROFILES_PATH.exists():
            print(f"Processed profiles file not found: {PROCESSED_PROFILES_PATH}")
            return None
        
        print(f"Loading processed profiles from {PROCESSED_PROFILES_PATH}")
        with open(PROCESSED_PROFILES_PATH, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        return self.build_from_profiles(profiles)
    
    def get_profile_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get full profile data by ID."""
        return self.profile_metadata.get(profile_id)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search. Filtering is handled downstream by the ranking engine."""
        if not self.vector_store:
            raise ValueError("Vector store not loaded")
        
        # Fetch all available profiles for downstream scoring/filtering
        try:
            total = self.vector_store.index.ntotal
            fetch_k = min(k, total) if total > 0 else k
        except Exception:
            fetch_k = k
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=fetch_k)
        return docs_with_scores
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        if not self.vector_store:
            return {"error": "Vector store not loaded"}
        
        return {
            "total_profiles": len(self.profile_metadata),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "model": EMBEDDING_MODEL,
            "index_path": str(FAISS_INDEX_DIR)
        }
