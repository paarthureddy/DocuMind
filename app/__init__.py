"""
Talent Search Engine - AI-powered casting assistant.

A production-ready semantic talent search system that transforms natural language
queries like "male villain brown 6 feet intense actor" into intelligent profile
matches using hybrid retrieval + advanced ranking.

Core Components:
- Data Pipeline: Normalization and semantic enrichment
- Embeddings Pipeline: Multi-embedding generation with FAISS storage
- Query Parser: Natural language understanding
- Retrieval Engine: Hybrid search with filtering
- Ranking System: Multi-factor scoring
- Talent Agent: LangGraph integration
- API Layer: FastAPI endpoints
- Performance: Caching and optimizations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

__version__ = "1.0.0"
__author__ = "Talent Search Engine Team"

# Core imports for easy access
from .config import *
from .data_pipeline import DataPipeline
from .embeddings_pipeline import EmbeddingsPipeline
from .query_parser import QueryParser
from .retrieval import RetrievalEngine
from .ranking import RankingEngine
from .talent_agent import run_agent, is_talent_query
from .talent_api import app
from .performance import performance_optimizer

__all__ = [
    # Configuration
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION", 
    "SCORING_WEIGHTS",
    
    # Core Classes
    "DataPipeline",
    "EmbeddingsPipeline", 
    "QueryParser",
    "RetrievalEngine",
    "RankingEngine",
    
    # Agent Functions
    "run_agent",
    "is_talent_query",
    
    # API
    "app",
    
    # Performance
    "performance_optimizer"
]
