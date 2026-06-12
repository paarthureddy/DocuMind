"""
Configuration settings for the talent search engine.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Data source
PROFILES_JSON_PATH = BASE_DIR / "data" / "1500_profiles.json"
PROCESSED_PROFILES_PATH = DATA_DIR / "processed_profiles.json"

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Search settings
MAX_SEARCH_RESULTS = 100
TOP_K_RERANK = 20

# Scoring weights
SCORING_WEIGHTS = {
    "semantic_similarity": 0.4,
    "casting_tag_match": 0.2,
    "personality_match": 0.1,
    "experience_score": 0.1,
    "rating_score": 0.1,
    "engagement_score": 0.1
}

# Complexion mapping
COMPLEXION_MAPPING = {
    "fair": "light",
    "wheatish": "brown", 
    "dusky": "dark",
    "light": "light",
    "brown": "brown",
    "dark": "dark"
}

# Height bucket mapping
def get_height_bucket(height_cm: float) -> str:
    """Convert height in cm to bucket."""
    if height_cm < 165:
        return "short"
    elif height_cm <= 180:
        return "medium"
    else:
        return "tall"

# Feet to cm conversion
def feet_to_cm(feet: float) -> float:
    """Convert feet to centimeters."""
    return round(feet * 30.48, 1)

# Inches to cm conversion  
def inches_to_cm(inches: float) -> float:
    """Convert inches to centimeters."""
    return round(inches * 2.54, 1)

# Height parsing regex patterns
HEIGHT_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*feet(?:\s*(\d+(?:\.\d+)?)\s*inches?)?",
    r"(\d+(?:\.\d+)?)\s*ft(?:\s*(\d+(?:\.\d+)?)\s*in)?",
    r"(\d+(?:\.\d+)?)\s*'",
    r"(\d+(?:\.\d+)?)\s*cm"
]
