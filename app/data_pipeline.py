"""
Data normalization and preprocessing pipeline for talent profiles.
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import (
    COMPLEXION_MAPPING, 
    get_height_bucket, 
    feet_to_cm, 
    inches_to_cm,
    HEIGHT_PATTERNS,
    PROCESSED_PROFILES_PATH
)
from llm import get_llm

class DataPipeline:
    """Handles normalization and enrichment of talent profiles."""
    
    def __init__(self):
        self.llm = get_llm()
        self.synonym_dict = self._build_synonym_dict()
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build synonym expansion dictionary."""
        return {
            "villain": ["antagonist", "negative role", "gangster", "mafia", "bad guy", "evil character"],
            "hero": ["lead role", "protagonist", "main character", "positive role", "good guy"],
            "romantic": ["charming", "lover", "romantic lead", "love interest"],
            "comic": ["funny", "comedian", "humorous", "comedy relief"],
            "supporting": ["side character", "secondary role", "supporting role"],
            "intense": ["aggressive", "dominant", "powerful", "strong"],
            "soft": ["gentle", "mild", "calm", "subtle"],
            "tall": ["lanky", "towering", "statuesque"],
            "short": ["petite", "compact", "small"],
            "brown": ["wheatish", "tan", "olive"],
            "light": ["fair", "pale", "light-skinned"],
            "dark": ["dusky", "deep", "rich"]
        }
    
    def normalize_complexion(self, complexion: str) -> str:
        """Normalize complexion values."""
        if not complexion:
            return "unknown"
        
        complexion_lower = complexion.lower().strip()
        return COMPLEXION_MAPPING.get(complexion_lower, complexion_lower)
    
    def normalize_height(self, height_str: Optional[str] = None, height_cm: Optional[float] = None) -> Dict[str, Any]:
        """Normalize height to cm and bucket."""
        if height_cm:
            # Already in cm
            cm_value = height_cm
        elif height_str:
            # Parse from string
            cm_value = self._parse_height_string(height_str)
        else:
            return {"height_cm": None, "height_bucket": "unknown"}
        
        if not cm_value:
            return {"height_cm": None, "height_bucket": "unknown"}
        
        return {
            "height_cm": cm_value,
            "height_bucket": get_height_bucket(cm_value)
        }
    
    def _parse_height_string(self, height_str: str) -> Optional[float]:
        """Parse height from various string formats."""
        height_str = height_str.lower().strip()
        
        for pattern in HEIGHT_PATTERNS:
            match = re.search(pattern, height_str)
            if match:
                groups = match.groups()
                
                if "cm" in height_str:
                    # Direct cm value
                    return float(groups[0])
                elif "'" in height_str or "ft" in height_str or "feet" in height_str:
                    # Feet and optional inches
                    feet = float(groups[0])
                    inches = float(groups[1]) if groups[1] else 0
                    return feet_to_cm(feet) + inches_to_cm(inches)
        
        return None
    
    def normalize_gender(self, gender: str) -> str:
        """Normalize gender values."""
        if not gender:
            return "unknown"
        
        gender_lower = gender.lower().strip()
        if gender_lower in ["male", "m", "man"]:
            return "male"
        elif gender_lower in ["female", "f", "woman"]:
            return "female"
        else:
            return "other"
    
    def normalize_text_field(self, text: str) -> str:
        """Normalize text fields."""
        if not text:
            return ""
        return text.lower().strip()
    
    def generate_semantic_profile(self, profile: Dict[str, Any]) -> str:
        """Generate rich semantic profile using LLM."""
        personal = profile.get("personal_info", {})
        professional = profile.get("professional_info", {})
        appearance_tags = profile.get("appearance_tags", [])
        rating = profile.get("rating", {})
        creator_metrics = profile.get("creator_metrics", {})
        
        # Extract key information
        gender = personal.get("gender", "unknown")
        height_info = self.normalize_height(
            personal.get("height"), 
            personal.get("height_cm")
        )
        complexion = self.normalize_complexion(personal.get("complexion"))
        craft = professional.get("craft", "unknown")
        skills = professional.get("skills", [])
        experience_years = professional.get("experience_years", 0)
        avg_rating = rating.get("average", 0)
        followers = creator_metrics.get("followers", 0)
        
        prompt = f"""
        Create a comprehensive casting profile description for this talent:

        Gender: {gender}
        Height: {height_info.get('height_bucket', 'unknown')}
        Complexion: {complexion}
        Craft: {craft}
        Skills: {', '.join(skills) if skills else 'none specified'}
        Experience: {experience_years} years
        Rating: {avg_rating}/5
        Social Following: {followers}
        Appearance Tags: {', '.join(appearance_tags) if appearance_tags else 'none specified'}

        Generate a rich paragraph that includes:
        - Role suitability (villain, hero, comic, supporting, etc.)
        - Personality traits (intense, dominant, soft, charming, etc.)
        - Appearance summary
        - Skills and experience level
        - Casting relevance and marketability

        Focus on what casting directors would find valuable. Be specific and descriptive.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating semantic profile: {e}")
            return f"Profile for {gender} {craft} with {experience_years} years experience."
    
    def extract_structured_semantics(self, profile: Dict[str, Any], semantic_profile: str) -> Dict[str, List[str]]:
        """Extract structured semantic fields using LLM."""
        prompt = f"""
        Based on this profile description, extract structured tags:

        Profile Description:
        {semantic_profile}

        Extract and return ONLY a JSON object with these exact keys:
        {{
            "casting_tags": ["list of suitable roles like villain, hero, comic, etc."],
            "personality_tags": ["list of personality traits like intense, soft, charming, etc."],
            "appearance_tags_enhanced": ["list of appearance descriptors"]
        }}

        Keep tags lowercase and specific. Maximum 5 tags per category.
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Try to parse JSON from response
            import json
            content = response.content.strip()
            
            # Extract JSON if it's wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            tags = json.loads(content)
            
            # Ensure all required keys exist
            return {
                "casting_tags": tags.get("casting_tags", []),
                "personality_tags": tags.get("personality_tags", []),
                "appearance_tags_enhanced": tags.get("appearance_tags_enhanced", [])
            }
            
        except Exception as e:
            print(f"Error extracting structured semantics: {e}")
            return {
                "casting_tags": [],
                "personality_tags": [],
                "appearance_tags_enhanced": []
            }
    
    def normalize_profile(self, raw_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single profile."""
        # Extract basic info
        personal = raw_profile.get("personal_info", {})
        professional = raw_profile.get("professional_info", {})
        
        # Extract crafts and projects
        crafts_data = professional.get("crafts", {})
        primary_subcraft = crafts_data.get("primary", {}).get("subcraft", "") if isinstance(crafts_data, dict) else ""
        secondary_subcrafts = [s.get("subcraft", "") for s in (crafts_data.get("secondary", []) if isinstance(crafts_data, dict) else [])]
        all_subcrafts = [primary_subcraft] + secondary_subcrafts
        all_subcrafts = [s for s in all_subcrafts if s]
        
        projects = professional.get("projects", [])
        past_roles = [p.get("role", "") for p in projects if isinstance(p, dict) and p.get("role")]
        
        # Normalize basic fields
        normalized = {
            "id": raw_profile.get("id"),
            "name": raw_profile.get("name", ""),
            "age": personal.get("age"),
            "gender": self.normalize_gender(personal.get("gender")),
            "height_info": self.normalize_height(
                personal.get("height"),
                personal.get("height_cm")
            ),
            "complexion_standard": self.normalize_complexion(personal.get("complexion")),
            "craft": self.normalize_text_field(professional.get("craft")),
            "skills": [self.normalize_text_field(skill) for skill in professional.get("skills", [])],
            "experience_years": professional.get("experience_years", 0),
            "rating": raw_profile.get("rating", {}),
            "creator_metrics": raw_profile.get("creator_metrics", {}),
            "appearance_tags": [self.normalize_text_field(tag) for tag in raw_profile.get("appearance_tags", [])],
            "past_roles": past_roles,
            "subcrafts": all_subcrafts
        }
        
        # Generate semantic enrichment
        semantic_profile = self.generate_semantic_profile(normalized)
        structured_semantics = self.extract_structured_semantics(normalized, semantic_profile)
        
        # Add semantic fields
        normalized.update({
            "semantic_profile": semantic_profile,
            "casting_tags": structured_semantics["casting_tags"],
            "personality_tags": structured_semantics["personality_tags"],
            "appearance_tags_enhanced": structured_semantics["appearance_tags_enhanced"]
        })
        
        return normalized
    
    def process_dataset(self, input_path: str, resume: bool = False) -> List[Dict[str, Any]]:
        """Process the entire dataset. Supports resuming from where it left off."""
        print(f"Loading dataset from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_profiles = json.load(f)
        
        # Resume support: load already-processed profiles
        processed_profiles = []
        start_idx = 0
        if resume and PROCESSED_PROFILES_PATH.exists():
            try:
                with open(PROCESSED_PROFILES_PATH, 'r', encoding='utf-8') as f:
                    processed_profiles = json.load(f)
                start_idx = len(processed_profiles)
                print(f"♻️  Resuming from profile {start_idx + 1} ({start_idx} already done)")
            except Exception:
                processed_profiles = []
                start_idx = 0
        
        total = len(raw_profiles)
        print(f"Processing {total - start_idx} remaining profiles (out of {total})...")
        
        for i in range(start_idx, total):
            profile = raw_profiles[i]
            if (i - start_idx) % 10 == 0:
                print(f"Processing profile {i+1}/{total}")
            
            try:
                normalized = self.normalize_profile(profile)
                processed_profiles.append(normalized)
            except Exception as e:
                print(f"Error processing profile {i}: {e}")
                continue
            
            # Incremental save every 10 profiles
            if len(processed_profiles) % 10 == 0:
                with open(PROCESSED_PROFILES_PATH, 'w', encoding='utf-8') as f:
                    json.dump(processed_profiles, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved checkpoint ({len(processed_profiles)} profiles)")
        
        # Final save
        print(f"Saving {len(processed_profiles)} processed profiles...")
        with open(PROCESSED_PROFILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_profiles, f, indent=2, ensure_ascii=False)
        
        print(f"Processed profiles saved to {PROCESSED_PROFILES_PATH}")
        return processed_profiles
    
    def expand_query_terms(self, query: str) -> List[str]:
        """Expand query terms using synonym dictionary."""
        expanded_terms = [query.lower()]
        
        for term, synonyms in self.synonym_dict.items():
            if term in query.lower():
                expanded_terms.extend(synonyms)
        
        return list(set(expanded_terms))  # Remove duplicates
