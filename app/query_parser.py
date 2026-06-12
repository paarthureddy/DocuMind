"""
Query understanding engine for natural language talent search.
Parses user queries and extracts structured search parameters.
"""

import re
import os
import json
import urllib.request
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5-mini")

def parse_search_query_with_llm(user_query: str) -> dict:
    """Uses Azure OpenAI via direct REST call to extract strict parameters for casting search."""
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        return {
            "target_role": None,
            "target_gender": None,
            "target_age_min": None,
            "target_age_max": None,
            "target_min_rating": None
        }

    # Ensure URL formatting handles training slashes properly
    base_url = AZURE_ENDPOINT.rstrip('/')
    url = f"{base_url}/openai/deployments/{MODEL_NAME}/chat/completions?api-version={AZURE_API_VERSION}"
    
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert entity extractor for a casting agency. "
                    "Extract the following search parameters from the user's query: "
                    "target_role (string), target_gender ('Male' or 'Female'), "
                    "target_age_min (integer), target_age_max (integer), "
                    "target_min_rating (float). "
                    "If a value is not mentioned, return null for that field. "
                    "CRITICAL: If the user's query is conversational (e.g., 'hi') or completely unrelated "
                    "to searching for talent, actors, models, or characters, you MUST set 'target_role' to 'INVALID_QUERY'. "
                    "Return ONLY a raw JSON object with exactly these keys. No markdown blocks."
                )
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        "response_format": { "type": "json_object" },
        "temperature": 0.0
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            content_str = res_data['choices'][0]['message']['content']
            return json.loads(content_str)
    except Exception as e:
        print(f"Azure OpenAI Parsing Error (REST): {e}")
        return {
            "target_role": None,
            "target_gender": None,
            "target_age_min": None,
            "target_age_max": None,
            "target_min_rating": None
        }



from config import (
    COMPLEXION_MAPPING,
    feet_to_cm,
    inches_to_cm,
    HEIGHT_PATTERNS
)

class QueryParser:
    """Parses natural language queries into structured search parameters."""
    
    def __init__(self):
        self.gender_terms = {
            "male": "male", "men": "male", "man": "male", "boy": "male",
            "female": "female", "women": "female", "woman": "female", "girl": "female"
        }
        
        self.complexion_terms = {
            "fair": "light", "light": "light", "pale": "light",
            "wheatish": "brown", "brown": "brown", "tan": "brown", "olive": "brown",
            "dusky": "dark", "dark": "dark", "deep": "dark"
        }
        
        self.craft_terms = {
            "actor": "actor", "actress": "actor", "acting": "actor",
            "model": "model", "modeling": "model",
            "dancer": "dancer", "dancing": "dancer",
            "singer": "singer", "singing": "singer",
            "influencer": "influencer", "creator": "influencer"
        }
        
        self.role_terms = {
            "villain": "villain", "antagonist": "villain", "negative": "villain", "bad guy": "villain",
            "hero": "hero", "protagonist": "hero", "lead": "hero", "main": "hero",
            "comic": "comic", "comedy": "comic", "funny": "comic", "comedian": "comic",
            "romantic": "romantic", "romance": "romantic", "lover": "romantic", "charming": "romantic",
            "supporting": "supporting", "side": "supporting", "secondary": "supporting"
        }
        
        self.personality_terms = {
            "intense": "intense", "aggressive": "intense", "dominant": "intense", "powerful": "intense",
            "soft": "soft", "gentle": "soft", "mild": "soft", "calm": "soft",
            "charming": "charming", "charismatic": "charming", "attractive": "charming",
            "funny": "funny", "humorous": "funny", "witty": "funny"
        }
    
    def parse_height(self, query: str) -> Dict[str, Optional[float]]:
        """Extract height constraints from query."""
        height_min = None
        height_max = None
        
        # Look for specific height patterns
        for pattern in HEIGHT_PATTERNS:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                groups = match.groups()
                
                if "cm" in query[match.start():match.end()]:
                    # Direct cm value
                    cm_value = float(groups[0])
                    height_min = cm_value - 5  # Allow 5cm tolerance
                    height_max = cm_value + 5
                elif "'" in query[match.start():match.end()] or "ft" in query[match.start():match.end()] or "feet" in query[match.start():match.end()]:
                    # Feet and optional inches
                    feet = float(groups[0])
                    inches = float(groups[1]) if groups[1] else 0
                    cm_value = feet_to_cm(feet) + inches_to_cm(inches)
                    height_min = cm_value - 5
                    height_max = cm_value + 5
        
        # Look for height descriptors
        height_descriptors = {
            "tall": (175, None),
            "short": (None, 165),
            "very tall": (180, None),
            "very short": (None, 160),
            "medium": (165, 180),
            "average": (165, 180)
        }
        
        for descriptor, (h_min, h_max) in height_descriptors.items():
            if descriptor in query.lower():
                if h_min and (height_min is None or h_min > height_min):
                    height_min = h_min
                if h_max and (height_max is None or h_max < height_max):
                    height_max = h_max
        
        return {
            "height_min": height_min,
            "height_max": height_max
        }
    
    def parse_gender(self, query: str) -> Optional[str]:
        """Extract gender from query."""
        query_lower = query.lower()
        # Check longer terms first (e.g., "female" before "male")
        # Use word boundary regex to prevent "male" matching inside "female"
        for term, gender in sorted(self.gender_terms.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
                return gender
        return None
    
    def parse_complexion(self, query: str) -> Optional[str]:
        """Extract complexion from query."""
        query_lower = query.lower()
        for term, complexion in self.complexion_terms.items():
            if term in query_lower:
                return complexion
        return None
    
    def parse_craft(self, query: str) -> Optional[str]:
        """Extract craft/profession from query."""
        query_lower = query.lower()
        for term, craft in self.craft_terms.items():
            if term in query_lower:
                return craft
        return None
    
    def extract_semantic_terms(self, query: str) -> List[str]:
        """Extract semantic terms for vector search."""
        semantic_terms = []
        query_lower = query.lower()
        
        # Extract role terms
        for term, normalized in self.role_terms.items():
            if term in query_lower:
                semantic_terms.append(normalized)
        
        # Extract personality terms
        for term, normalized in self.personality_terms.items():
            if term in query_lower:
                semantic_terms.append(normalized)
        
        # Extract any remaining meaningful words (filter out stop words)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "looking", "search", "find", "need", "want", "seeking"
        }
        
        words = re.findall(r'\b\w+\b', query_lower)
        meaningful_words = [
            word for word in words 
            if word not in stop_words and len(word) > 2
            and word not in semantic_terms  # Avoid duplicates
        ]
        
        semantic_terms.extend(meaningful_words)
        
        return semantic_terms
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Role expansions
        role_expansions = {
            "villain": ["antagonist", "negative role", "bad guy", "evil character"],
            "hero": ["protagonist", "lead role", "main character", "positive role"],
            "comic": ["comedy", "funny", "humorous", "comedian"],
            "romantic": ["romance", "lover", "charming", "love interest"]
        }
        
        for term, expansions in role_expansions.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
        
        # Personality expansions
        personality_expansions = {
            "intense": ["aggressive", "dominant", "powerful", "strong"],
            "soft": ["gentle", "mild", "calm", "subtle"],
            "charming": ["attractive", "charismatic", "appealing"]
        }
        
        for term, expansions in personality_expansions.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
        
        return list(set(expanded_terms))  # Remove duplicates
    
    def parse_roles(self, query: str) -> List[str]:
        """Extract specific target roles from the query."""
        query_lower = query.lower()
        extracted_roles = []
        
        extended_roles = list(self.role_terms.keys()) + [
            "comedian", "background character", "cameo", "lead role", "supporting actor",
            "dancer", "background dancer", "stand-up comedian", "extra", "lead"
        ]
        
        for role in extended_roles:
            if re.search(r'\b' + re.escape(role) + r'\b', query_lower):
                extracted_roles.append(role)
                
        return list(set(extracted_roles))
    
    def parse_age_range(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract age constraints from query."""
        query_lower = query.lower()
        if "in their 20s" in query_lower or "20s" in query_lower:
            return 20, 29
        if "in their 30s" in query_lower or "30s" in query_lower:
            return 30, 39
        if "in their 40s" in query_lower or "40s" in query_lower:
            return 40, 49
        if "in their 50s" in query_lower or "50s" in query_lower:
            return 50, 59
            
        match = re.search(r'\b(?:age\s*)?(\d{1,2})\b(?:\s*(?:years?|yrs?|yo)?)', query_lower)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 80:
                return age, age
                
        return None, None

    def parse_min_rating(self, query: str) -> Optional[float]:
        """Extract min rating from query."""
        query_lower = query.lower()
        if "highly rated" in query_lower or "top rated" in query_lower or "5 stars" in query_lower or "5 star" in query_lower:
            return 4.5
        if "4+ stars" in query_lower or "4 stars" in query_lower or "good rating" in query_lower:
            return 4.0
        
        match = re.search(r'(\d(?:\.\d)?)\+?\s*stars?', query_lower)
        if match:
            return float(match.group(1))
            
        return None

    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse complete query into structured search parameters."""
        # Clean query
        query = query.strip()
        if not query:
            return {
                "filters": {},
                "semantic_query": "",
                "expanded_terms": []
            }
        
        # Extract Target Entities via Azure LLM
        llm_extracted = parse_search_query_with_llm(query)
        
        target_role = llm_extracted.get("target_role")
        target_roles = [target_role] if target_role else []
        target_gender = llm_extracted.get("target_gender")
        local_gender = self.parse_gender(query)
        target_age_min = llm_extracted.get("target_age_min")
        target_age_max = llm_extracted.get("target_age_max")
        target_min_rating = llm_extracted.get("target_min_rating")
        
        llm_gender = target_gender.lower().strip() if target_gender and target_gender.lower().strip() in ["male", "female"] else None
        gender_filter = local_gender or llm_gender
        target_gender = gender_filter

        # Extract structured filters
        height_info = self.parse_height(query)
        filters = {
            "gender": gender_filter,
            "height_min": height_info["height_min"],
            "height_max": height_info["height_max"],
            "complexion": self.parse_complexion(query),
            "craft": self.parse_craft(query)
        }
        
        # Remove None values from filters
        filters = {k: v for k, v in filters.items() if v is not None}
        
        # Extract semantic terms
        semantic_terms = self.extract_semantic_terms(query)
        semantic_query = " ".join(semantic_terms)
        
        # Expand query terms
        expanded_terms = self.expand_query(query)
        
        return {
            "filters": filters,
            "semantic_query": semantic_query,
            "expanded_terms": expanded_terms,
            "target_roles": target_roles,
            "target_role": target_role,
            "target_gender": target_gender,
            "target_age_min": target_age_min,
            "target_age_max": target_age_max,
            "target_min_rating": target_min_rating,
            "original_query": query
        }
    
    def explain_parsing(self, query: str, parsed_result: Dict[str, Any]) -> str:
        """Generate explanation of how query was parsed."""
        explanation = f"Query: '{query}'\n\n"
        
        filters = parsed_result.get("filters", {})
        if filters:
            explanation += "Filters applied:\n"
            if "gender" in filters:
                explanation += f"- Gender: {filters['gender']}\n"
            if "height_min" in filters or "height_max" in filters:
                height_range = ""
                if filters.get("height_min"):
                    height_range += f"{filters['height_min']}cm"
                if filters.get("height_max"):
                    if height_range:
                        height_range += " - "
                    height_range += f"{filters['height_max']}cm"
                explanation += f"- Height: {height_range}\n"
            if "complexion" in filters:
                explanation += f"- Complexion: {filters['complexion']}\n"
            if "craft" in filters:
                explanation += f"- Craft: {filters['craft']}\n"
            explanation += "\n"
        
        semantic_query = parsed_result.get("semantic_query", "")
        if semantic_query:
            explanation += f"Semantic search terms: {semantic_query}\n\n"
        
        expanded_terms = parsed_result.get("expanded_terms", [])
        if len(expanded_terms) > 1:
            explanation += f"Expanded terms: {', '.join(expanded_terms[1:])}\n"
        
        return explanation
