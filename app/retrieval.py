"""
Hybrid retrieval engine combining vector search with structured filtering.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from embeddings_pipeline import EmbeddingsPipeline
from query_parser import QueryParser
from ranking import RankingEngine
from config import MAX_SEARCH_RESULTS, TOP_K_RERANK

class RetrievalEngine:
    """Hybrid retrieval engine for talent search."""
    
    def __init__(self):
        self.embeddings_pipeline = EmbeddingsPipeline()
        self.query_parser = QueryParser()
        self.ranking_engine = RankingEngine()
        
        # Load existing vector store if available
        if self.embeddings_pipeline.vector_store_exists():
            self.embeddings_pipeline.load_vector_store()
        else:
            print("Warning: No vector store found. Please build the index first.")
    
    def search_profiles(
        self, 
        query: str, 
        max_results: int = MAX_SEARCH_RESULTS,
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Perform hybrid search for talent profiles.
        
        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            enable_reranking: Whether to apply advanced ranking
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.embeddings_pipeline.vector_store:
            return {
                "success": False,
                "error": "Vector store not loaded. Please build the index first.",
                "results": []
            }
        
        # Parse query
        parsed_query = self.query_parser.parse_query(query)
        
        # Perform vector search with filters
        try:
            # Retrieve ALL profiles from FAISS for full scoring
            docs_with_scores = self.embeddings_pipeline.similarity_search(
                query=parsed_query["semantic_query"],
                k=10000,
                filter_dict=parsed_query["filters"]
            )
            
            if not docs_with_scores:
                return {
                    "success": True,
                    "query": query,
                    "parsed_query": parsed_query,
                    "results": [],
                    "total_found": 0
                }
            
            # Convert to profile format
            candidates = []
            for doc, similarity_score in docs_with_scores:
                profile_data = {
                    "document": doc,
                    "similarity_score": float(similarity_score),
                    "metadata": doc.metadata
                }
                candidates.append(profile_data)
            
            # Apply ranking
            if enable_reranking:
                ranked_results = self.ranking_engine.rank_candidates(
                    candidates, 
                    parsed_query
                )
            else:
                # Simple ranking by similarity score
                ranked_results = sorted(
                    candidates, 
                    key=lambda x: x["similarity_score"], 
                    reverse=True
                )
            
            # Return top 50 ranked results
            top_results = ranked_results[:50]
            
            # Generate explanations
            explained_results = []
            for result in top_results:
                explanation = self.ranking_engine.generate_explanation(
                    result, 
                    parsed_query
                )
                
                meta = result.get("metadata", {})
                explained_results.append({
                    "profile_id": meta.get("profile_id", ""),
                    "name": meta.get("name", "Unknown"),
                    "score": result.get("final_score_pct", result.get("final_score", 0)),
                    "final_score": result.get("final_score", 0),
                    "similarity_score": result.get("similarity_score", 0),
                    "explanation": explanation,
                    "metadata": {
                        "gender": meta.get("gender", ""),
                        "age": meta.get("age"),
                        "height_cm": meta.get("height_cm"),
                        "height_bucket": meta.get("height_bucket", ""),
                        "complexion": meta.get("complexion", ""),
                        "craft": meta.get("craft", ""),
                        "experience_years": meta.get("experience_years", 0),
                        "rating_average": meta.get("rating_average", 0),
                        "casting_tags": meta.get("casting_tags", []),
                        "personality_tags": meta.get("personality_tags", []),
                        "skills": meta.get("skills", []),
                        "past_roles": meta.get("past_roles", []),
                        "subcrafts": meta.get("subcrafts", [])
                    }
                })
            
            return {
                "success": True,
                "query": query,
                "parsed_query": parsed_query,
                "results": explained_results,
                "total_found": len(docs_with_scores),
                "returned": len(explained_results)
            }
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return {
                "success": False,
                "error": f"Search failed: {str(e)}\n{tb}",
                "results": []
            }
    
    def get_similar_profiles(
        self, 
        profile_id: str, 
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Find profiles similar to a given profile.
        
        Args:
            profile_id: ID of the reference profile
            max_results: Maximum number of similar profiles to return
            
        Returns:
            Dictionary with similar profiles
        """
        if not self.embeddings_pipeline.vector_store:
            return {
                "success": False,
                "error": "Vector store not loaded",
                "results": []
            }
        
        # Get the reference profile
        reference_profile = self.embeddings_pipeline.get_profile_by_id(profile_id)
        if not reference_profile:
            return {
                "success": False,
                "error": f"Profile {profile_id} not found",
                "results": []
            }
        
        # Create search query from profile
        search_query = " ".join([
            reference_profile.get("semantic_profile", ""),
            " ".join(reference_profile.get("casting_tags", [])),
            " ".join(reference_profile.get("personality_tags", [])),
            " ".join(reference_profile.get("skills", []))
        ])
        
        # Perform similarity search
        try:
            docs_with_scores = self.embeddings_pipeline.similarity_search(
                query=search_query,
                k=max_results + 1  # +1 to exclude the reference profile
            )
            
            # Filter out the reference profile
            similar_profiles = []
            for doc, similarity_score in docs_with_scores:
                if doc.metadata["profile_id"] != profile_id:
                    similar_profiles.append({
                        "profile_id": doc.metadata["profile_id"],
                        "name": doc.metadata["name"],
                        "similarity_score": float(similarity_score),
                        "metadata": {
                            "gender": doc.metadata["gender"],
                            "height_bucket": doc.metadata["height_bucket"],
                            "complexion": doc.metadata["complexion"],
                            "craft": doc.metadata["craft"],
                            "casting_tags": doc.metadata["casting_tags"],
                            "personality_tags": doc.metadata["personality_tags"]
                        }
                    })
            
            return {
                "success": True,
                "reference_profile_id": profile_id,
                "reference_profile_name": reference_profile.get("name"),
                "results": similar_profiles[:max_results],
                "total_found": len(similar_profiles)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Similarity search failed: {str(e)}",
                "results": []
            }
    
    def filter_search(
        self, 
        filters: Dict[str, Any], 
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Perform search using only structured filters (no semantic search).
        
        Args:
            filters: Dictionary of filter criteria
            max_results: Maximum number of results
            
        Returns:
            Dictionary with filtered results
        """
        if not self.embeddings_pipeline.vector_store:
            return {
                "success": False,
                "error": "Vector store not loaded",
                "results": []
            }
        
        # Use a generic query to get all documents, then filter
        try:
            docs_with_scores = self.embeddings_pipeline.similarity_search(
                query="talent profile",  # Generic query
                k=1000,  # Get many results for filtering
                filter_dict=filters
            )
            
            # Convert to profile format
            filtered_results = []
            for doc, similarity_score in docs_with_scores:
                filtered_results.append({
                    "profile_id": doc.metadata["profile_id"],
                    "name": doc.metadata["name"],
                    "metadata": {
                        "gender": doc.metadata["gender"],
                        "height_cm": doc.metadata["height_cm"],
                        "height_bucket": doc.metadata["height_bucket"],
                        "complexion": doc.metadata["complexion"],
                        "craft": doc.metadata["craft"],
                        "experience_years": doc.metadata["experience_years"],
                        "rating": doc.metadata["rating_average"],
                        "casting_tags": doc.metadata["casting_tags"],
                        "personality_tags": doc.metadata["personality_tags"],
                        "skills": doc.metadata["skills"]
                    }
                })
            
            return {
                "success": True,
                "filters": filters,
                "results": filtered_results[:max_results],
                "total_found": len(filtered_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Filter search failed: {str(e)}",
                "results": []
            }
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build the search index from processed profiles.
        
        Args:
            force_rebuild: Whether to rebuild even if index exists
            
        Returns:
            Dictionary with build status
        """
        if self.embeddings_pipeline.vector_store_exists() and not force_rebuild:
            return {
                "success": False,
                "message": "Index already exists. Use force_rebuild=True to rebuild."
            }
        
        try:
            vector_store = self.embeddings_pipeline.build_from_processed_file()
            if vector_store:
                stats = self.embeddings_pipeline.get_embedding_stats()
                return {
                    "success": True,
                    "message": "Index built successfully",
                    "stats": stats
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to build index. Check if processed profiles file exists."
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Index build failed: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        stats = self.embeddings_pipeline.get_embedding_stats()
        
        if "error" not in stats:
            stats["search_engine_ready"] = True
        else:
            stats["search_engine_ready"] = False
        
        return stats
