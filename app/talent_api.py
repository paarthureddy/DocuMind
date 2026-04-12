"""
FastAPI endpoints for the talent search engine.
Provides REST API for talent search and management.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import DataPipeline
from embeddings_pipeline import EmbeddingsPipeline
from retrieval import RetrievalEngine
from talent_agent import run_agent
from intent_classifier import classify_intent
from config import PROFILES_JSON_PATH, PROCESSED_PROFILES_PATH

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    enable_reranking: bool = True

class SimilarRequest(BaseModel):
    profile_id: str
    max_results: int = 10

class FilterRequest(BaseModel):
    filters: Dict[str, Any]
    max_results: int = 50

class ChatRequest(BaseModel):
    message: str

class ProcessRequest(BaseModel):
    force_reprocess: bool = False

class BuildIndexRequest(BaseModel):
    force_rebuild: bool = False

# Initialize FastAPI app
app = FastAPI(
    title="Talent Search API",
    description="AI-powered casting assistant and talent search engine",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_pipeline = None
retrieval_engine = None

def get_data_pipeline():
    """Get or create data pipeline instance."""
    global data_pipeline
    if data_pipeline is None:
        data_pipeline = DataPipeline()
    return data_pipeline

def get_retrieval_engine():
    """Get or create retrieval engine instance."""
    global retrieval_engine
    if retrieval_engine is None:
        retrieval_engine = RetrievalEngine()
    return retrieval_engine

# ─────────────────────────────────────────────
# Health and Status Endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Talent Search API",
        "version": "1.0.0",
        "description": "AI-powered casting assistant and talent search engine",
        "endpoints": {
            "search": "/search_profiles",
            "similar": "/similar_profiles",
            "filter": "/filter_profiles",
            "chat": "/chat",
            "stats": "/stats",
            "process": "/process_data",
            "build_index": "/build_index"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine = get_retrieval_engine()
    stats = engine.get_stats()
    
    return {
        "status": "ok",
        "search_engine_ready": stats.get("search_engine_ready", False),
        "data_processed": PROCESSED_PROFILES_PATH.exists(),
        "source_data_exists": os.path.exists(PROFILES_JSON_PATH),
        "stats": stats
    }

@app.get("/stats")
async def get_stats():
    """Get detailed system statistics."""
    engine = get_retrieval_engine()
    stats = engine.get_stats()
    
    # Add data processing stats
    if PROCESSED_PROFILES_PATH.exists():
        try:
            with open(PROCESSED_PROFILES_PATH, 'r', encoding='utf-8') as f:
                processed_profiles = json.load(f)
                stats["processed_profiles_count"] = len(processed_profiles)
        except:
            stats["processed_profiles_count"] = 0
    else:
        stats["processed_profiles_count"] = 0
    
    return stats

# ─────────────────────────────────────────────
# Search Endpoints
# ─────────────────────────────────────────────

@app.post("/search_profiles")
async def search_profiles(request: SearchRequest):
    """Search for talent profiles using natural language query."""
    try:
        engine = get_retrieval_engine()
        result = engine.search_profiles(
            query=request.query,
            max_results=request.max_results,
            enable_reranking=request.enable_reranking
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Search failed"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/similar_profiles")
async def get_similar_profiles(request: SimilarRequest):
    """Find profiles similar to a given profile ID."""
    try:
        engine = get_retrieval_engine()
        result = engine.get_similar_profiles(
            profile_id=request.profile_id,
            max_results=request.max_results
        )
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "Profile not found"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.post("/filter_profiles")
async def filter_profiles(request: FilterRequest):
    """Search profiles using structured filters only."""
    try:
        engine = get_retrieval_engine()
        result = engine.filter_search(
            filters=request.filters,
            max_results=request.max_results
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Filter search failed"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter search failed: {str(e)}")

@app.get("/profile/{profile_id}")
async def get_profile(profile_id: str):
    """Get detailed information about a specific profile."""
    try:
        engine = get_retrieval_engine()
        profile = engine.embeddings_pipeline.get_profile_by_id(profile_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {"success": True, "profile": profile}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")

# ─────────────────────────────────────────────
# Chat/Agent Endpoint
# ─────────────────────────────────────────────

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Intelligent chat endpoint with robust intent classification.
    
    Routes queries based on intent:
    - calculator: Pure math expressions
    - profile_search: Talent/casting queries  
    - chat: General conversation
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        query = request.message.strip()
        
        # Classify intent using robust classifier
        intent = classify_intent(query)
        
        # Route to appropriate handler based on classified intent
        if intent == "calculator":
            # Mathematical query - use calculator tool
            try:
                from talent_agent import calculator
                calc_result = calculator.invoke(query)
                result = {
                    "answer": calc_result,
                    "tools_used": ["calculator"],
                    "success": True,
                    "intent": "calculator"
                }
            except Exception as e:
                result = {
                    "answer": f"Calculation error: {str(e)}",
                    "tools_used": [],
                    "success": False,
                    "intent": "calculator"
                }
                
        elif intent == "profile_search":
            # Talent search query - use profile search
            try:
                retrieval_engine = get_retrieval_engine()
                search_result = retrieval_engine.search_profiles(
                    query=query,
                    max_results=10,
                    enable_reranking=True
                )
                
                if search_result["success"] and search_result["results"]:
                    # Format results for chat response
                    profiles_text = f"Found {len(search_result['results'])} matching profiles:\n\n"
                    for i, profile in enumerate(search_result["results"][:5], 1):
                        profiles_text += f"{i}. **{profile['name']}** (Score: {profile['score']:.2f})\n"
                        profiles_text += f"   {profile['explanation']}\n"
                        if profile['metadata'].get('gender'):
                            profiles_text += f"   Gender: {profile['metadata']['gender']}, "
                        if profile['metadata'].get('height_cm'):
                            profiles_text += f"Height: {profile['metadata']['height_cm']}cm, "
                        if profile['metadata'].get('craft'):
                            profiles_text += f"Craft: {profile['metadata']['craft']}"
                        profiles_text += "\n\n"
                    
                    result = {
                        "answer": profiles_text.strip(),
                        "tools_used": ["profile_search"],
                        "success": True,
                        "intent": "profile_search",
                        "results_count": len(search_result["results"])
                    }
                else:
                    result = {
                        "answer": f"No talent profiles found matching '{query}'. Try adjusting your search terms or contact us for expanded search options.",
                        "tools_used": ["profile_search"],
                        "success": True,
                        "intent": "profile_search",
                        "results_count": 0
                    }
                    
            except Exception as e:
                result = {
                    "answer": f"Talent search failed: {str(e)}",
                    "tools_used": [],
                    "success": False,
                    "intent": "profile_search"
                }
                
        else:
            # General chat query - use AI agent
            try:
                agent_result = run_agent(query)
                result = {
                    "answer": agent_result["answer"],
                    "tools_used": agent_result["tools_used"],
                    "success": agent_result["success"],
                    "intent": "chat"
                }
            except Exception as e:
                # Fallback to helpful message
                result = {
                    "answer": "I'm an AI-powered casting assistant. I can help you with:\n\n• 🎭 **Talent Search**: Find actors, models, singers (e.g., 'male villain actor')\n• 🧮 **Calculations**: Math expressions (e.g., '20+20')\n• 🤖 **General Chat**: Answer questions and provide assistance\n\nTry one of the above or ask me anything else!",
                    "tools_used": [],
                    "success": True,
                    "intent": "chat"
                }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# ─────────────────────────────────────────────
# Data Management Endpoints
# ─────────────────────────────────────────────

@app.post("/process_data")
async def process_data(request: ProcessRequest):
    """Process raw talent data and generate semantic enrichments."""
    try:
        if not os.path.exists(PROFILES_JSON_PATH):
            raise HTTPException(
                status_code=404, 
                detail=f"Source data file not found: {PROFILES_JSON_PATH}"
            )
        
        # Check if already processed
        if PROCESSED_PROFILES_PATH.exists() and not request.force_reprocess:
            return {
                "success": False,
                "message": "Data already processed. Use force_reprocess=True to reprocess."
            }
        
        pipeline = get_data_pipeline()
        processed_profiles = pipeline.process_dataset(PROFILES_JSON_PATH)
        
        return {
            "success": True,
            "message": f"Successfully processed {len(processed_profiles)} profiles",
            "processed_count": len(processed_profiles),
            "output_file": str(PROCESSED_PROFILES_PATH)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")

@app.post("/build_index")
async def build_search_index(request: BuildIndexRequest):
    """Build the vector search index from processed data."""
    try:
        if not PROCESSED_PROFILES_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Processed data not found. Please run /process_data first."
            )
        
        engine = get_retrieval_engine()
        result = engine.build_index(force_rebuild=request.force_rebuild)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index building failed: {str(e)}")

@app.delete("/clear_index")
async def clear_search_index():
    """Clear the search index and processed data."""
    try:
        # Clear vector store
        from config import FAISS_INDEX_DIR
        if FAISS_INDEX_DIR.exists():
            import shutil
            shutil.rmtree(FAISS_INDEX_DIR)
        
        # Clear processed data
        if PROCESSED_PROFILES_PATH.exists():
            PROCESSED_PROFILES_PATH.unlink()
        
        # Clear agent cache
        from talent_agent import clear_agent_cache
        clear_agent_cache()
        
        # Reset global instances
        global data_pipeline, retrieval_engine
        data_pipeline = None
        retrieval_engine = None
        
        return {
            "success": True,
            "message": "Search index and processed data cleared successfully."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# ─────────────────────────────────────────────
# Utility Endpoints
# ─────────────────────────────────────────────

@app.get("/query_explain")
async def explain_query(query: str = Query(..., description="Query to explain")):
    """Get explanation of how a query is parsed."""
    try:
        from query_parser import QueryParser
        parser = QueryParser()
        parsed = parser.parse_query(query)
        explanation = parser.explain_parsing(query, parsed)
        
        return {
            "query": query,
            "parsed_query": parsed,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query explanation failed: {str(e)}")

@app.get("/casting_suggestions")
async def get_casting_suggestions():
    """Get common casting suggestions and example queries."""
    suggestions = {
        "example_queries": [
            "male villain brown 6 feet intense actor",
            "female lead romantic fair complexion 5'6\" model",
            "supporting actor comic timing medium height",
            "tall male antagonist dark complexion experienced",
            "young female protagonist light complexion charming"
        ],
        "common_roles": [
            "villain", "hero", "lead", "supporting", "comic", "romantic",
            "antagonist", "protagonist", "character actor"
        ],
        "search_tips": [
            "Be specific about physical attributes (gender, height, complexion)",
            "Include personality traits (intense, charming, comic)",
            "Specify experience level and craft (actor, model, dancer)",
            "Combine multiple criteria for better results"
        ]
    }
    
    return suggestions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
