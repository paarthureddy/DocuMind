"""
Advanced ranking system for talent search results.
Implements multi-factor scoring with intelligent weighting.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import re
from difflib import SequenceMatcher

from config import SCORING_WEIGHTS

class RankingEngine:
    """Advanced ranking engine for talent search results."""
    
    def __init__(self):
        self.weights = SCORING_WEIGHTS.copy()
    
    def normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range."""
        if max_val <= min_val:
            return 0.5  # Default middle value
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def calculate_semantic_score(self, similarity_score: float) -> float:
        """
        Calculate semantic similarity score.
        FAISS returns L2 distance, so we convert to similarity.
        """
        # Convert distance to similarity (lower distance = higher similarity)
        # Typical FAISS distances range from 0-50 for sentence embeddings
        max_distance = 50.0
        similarity = max(0, 1 - (similarity_score / max_distance))
        return similarity
    
    def calculate_casting_tag_match(
        self, 
        casting_tags: List[str], 
        query_terms: List[str]
    ) -> float:
        """Calculate casting tag relevance score."""
        if not casting_tags or not query_terms:
            return 0.0
        
        # Convert to lowercase for matching
        casting_tags_lower = [tag.lower() for tag in casting_tags]
        query_terms_lower = [term.lower() for term in query_terms]
        
        # Count matches
        matches = 0
        for query_term in query_terms_lower:
            for tag in casting_tags_lower:
                if query_term in tag or tag in query_term:
                    matches += 1
                    break  # Count each query term only once
        
        # Normalize by number of query terms
        return matches / len(query_terms_lower)
    
    def calculate_personality_match(
        self, 
        personality_tags: List[str], 
        query_terms: List[str]
    ) -> float:
        """Calculate personality trait relevance score."""
        if not personality_tags or not query_terms:
            return 0.0
        
        # Convert to lowercase
        personality_tags_lower = [tag.lower() for tag in personality_tags]
        query_terms_lower = [term.lower() for term in query_terms]
        
        # Count matches
        matches = 0
        for query_term in query_terms_lower:
            for tag in personality_tags_lower:
                if query_term in tag or tag in query_term:
                    matches += 1
                    break
        
        return matches / len(query_terms_lower)
    
    def calculate_experience_score(self, experience_years: float) -> float:
        """Calculate experience score (0-1)."""
        if not experience_years:
            return 0.0
        
        # Experience scoring: 0-5 years = 0.2, 5-10 years = 0.5, 10+ years = 1.0
        if experience_years >= 10:
            return 1.0
        elif experience_years >= 5:
            return 0.5 + (experience_years - 5) * 0.1  # 0.5 to 1.0
        else:
            return experience_years * 0.04  # 0 to 0.2
    
    def calculate_rating_score(self, rating: float) -> float:
        """Calculate rating score (0-1)."""
        if not rating:
            return 0.0
        
        # Normalize rating (assuming 0-5 scale)
        return min(1.0, rating / 5.0)
    
    def calculate_engagement_score(self, engagement_rate: float, followers: int) -> float:
        """Calculate social media engagement score."""
        if not engagement_rate and not followers:
            return 0.0
        
        # Engagement rate is more important than follower count
        engagement_score = 0.0
        if engagement_rate:
            # Normalize engagement rate (assuming 0-20% is good range)
            engagement_score = min(1.0, engagement_rate / 20.0)
        
        follower_score = 0.0
        if followers:
            # Log scale for followers (1K = 0.2, 10K = 0.5, 100K = 0.8, 1M+ = 1.0)
            if followers >= 1000000:
                follower_score = 1.0
            elif followers >= 100000:
                follower_score = 0.8
            elif followers >= 10000:
                follower_score = 0.5
            elif followers >= 1000:
                follower_score = 0.2
            else:
                follower_score = followers / 5000.0  # 0 to 0.2
        
        # Weight engagement more heavily
        return engagement_score * 0.7 + follower_score * 0.3
    
    def extract_query_terms(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Extract relevant terms from parsed query for matching."""
        terms = []
        
        # Add semantic query terms
        semantic_query = parsed_query.get("semantic_query", "")
        if semantic_query:
            terms.extend(semantic_query.split())
        
        # Add expanded terms
        expanded_terms = parsed_query.get("expanded_terms", [])
        if expanded_terms:
            terms.extend(expanded_terms[1:])  # Skip the original query
        
        # Add filter-based terms
        filters = parsed_query.get("filters", {})
        if "craft" in filters:
            terms.append(filters["craft"])
        
        return list(set([term.lower() for term in terms if term.strip()]))
    
    def _fuzzy_role_match(self, query_role: str, profile_role: str) -> bool:
        """Check if two role strings match, handling typos via fuzzy matching."""
        q = query_role.lower().strip()
        p = profile_role.lower().strip()
        if not q or not p:
            return False
        # Exact substring match
        if q in p or p in q:
            return True
        # Fuzzy match (handles typos like 'villian' vs 'villain')
        ratio = SequenceMatcher(None, q, p).ratio()
        return ratio >= 0.75

    def calculate_final_score(
        self, 
        candidate: Dict[str, Any], 
        parsed_query: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate final ranking score for a candidate using dynamic scoring rules."""
        metadata = candidate.get("metadata", {})
        
        # Base Score: semantic_faiss_score directly
        semantic_score = self.calculate_semantic_score(candidate.get("similarity_score", 50.0))
        final_score = semantic_score
        
        target_role = parsed_query.get("target_role")
        target_gender = parsed_query.get("target_gender")
        target_age_min = parsed_query.get("target_age_min")
        target_age_max = parsed_query.get("target_age_max")
        target_min_rating = parsed_query.get("target_min_rating")
        
        # Strict Gender Filter: -100 penalty for mismatch
        gender_filtered = False
        if target_gender is not None:
            profile_gender = metadata.get("gender")
            if profile_gender:
                if str(profile_gender).lower() != str(target_gender).lower():
                    final_score -= 100
                    gender_filtered = True
                else:
                    final_score += 0.1
        
        # Role Boost with fuzzy matching
        if target_role is not None:
            past_roles = [str(r) for r in metadata.get("past_roles", [])]
            subcrafts = [str(s) for s in metadata.get("subcrafts", [])]
            casting_tags = [str(t) for t in metadata.get("casting_tags", [])]
            craft = str(metadata.get("craft", ""))
            all_profile_roles = past_roles + subcrafts + casting_tags + [craft]
            
            if any(self._fuzzy_role_match(target_role, pr) for pr in all_profile_roles):
                final_score += 0.2
        
        # Age Match
        if target_age_min is not None and target_age_max is not None:
            age = metadata.get("age")
            if age is not None:
                try:
                    age_val = int(age)
                    if target_age_min <= age_val <= target_age_max:
                        final_score += 0.1
                except (ValueError, TypeError):
                    pass
        
        # Rating Match
        if target_min_rating is not None:
            rating = metadata.get("rating_average")
            if rating is not None:
                try:
                    if float(rating) >= float(target_min_rating):
                        final_score += 0.1
                except (ValueError, TypeError):
                    pass
        
        # Normalize to percentage: min(int(final_score * 100), 100)
        score_pct = min(int(final_score * 100), 100) if not gender_filtered else 0
        
        return {
            "final_score": final_score,
            "final_score_pct": score_pct,
            "semantic_score": semantic_score,
            "gender_filtered": gender_filtered
        }
    
    def rank_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank candidates using multi-factor scoring."""
        ranked_candidates = []
        target_gender = parsed_query.get("target_gender")
        
        for candidate in candidates:
            scores = self.calculate_final_score(candidate, parsed_query)
            candidate.update(scores)
            # If gender was specified, completely remove mismatched profiles
            if target_gender is not None and scores.get("gender_filtered", False):
                continue  # Drop this profile entirely
            ranked_candidates.append(candidate)
        
        # Sort by final score (descending)
        ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        return ranked_candidates
    
    def generate_explanation(
        self, 
        result: Dict[str, Any], 
        parsed_query: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for why a profile matches."""
        metadata = result["metadata"]
        reasons = []
        
        # Check casting tag matches
        query_terms = self.extract_query_terms(parsed_query)
        casting_tags = metadata.get("casting_tags", [])
        for tag in casting_tags:
            for term in query_terms:
                if term.lower() in tag.lower():
                    reasons.append(f"matches {tag} role")
                    break
        
        # Check personality matches
        personality_tags = metadata.get("personality_tags", [])
        for tag in personality_tags:
            for term in query_terms:
                if term.lower() in tag.lower():
                    reasons.append(f"has {tag} personality")
                    break
        
        # Check experience
        experience = metadata.get("experience_years", 0)
        if experience >= 10:
            reasons.append("highly experienced")
        elif experience >= 5:
            reasons.append("good experience")
        
        # Check rating
        rating = metadata.get("rating_average", 0)
        if rating >= 4.5:
            reasons.append("excellent rating")
        elif rating >= 4.0:
            reasons.append("good rating")
        
        # Check physical attributes
        filters = parsed_query.get("filters", {})
        if "height_bucket" in filters:
            if metadata.get("height_bucket") == filters["height_bucket"]:
                reasons.append(f"{filters['height_bucket']} height")
        
        if "complexion" in filters:
            if metadata.get("complexion") == filters["complexion"]:
                reasons.append(f"{filters['complexion']} complexion")
        
        # Check skills
        skills = metadata.get("skills", [])
        if skills:
            reasons.append(f"skilled in {', '.join(skills[:3])}")
        
        # Generate final explanation
        if reasons:
            if len(reasons) <= 3:
                explanation = "Matches: " + ", ".join(reasons) + "."
            else:
                explanation = "Matches: " + ", ".join(reasons[:3]) + f" and {len(reasons)-3} more factors."
        else:
            explanation = "Good semantic match to your query."
        
        return explanation
    
    def update_weights(self, new_weights: Dict[str, float]) -> bool:
        """Update scoring weights."""
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            return False
        
        # Validate all required keys exist
        required_keys = set(self.weights.keys())
        provided_keys = set(new_weights.keys())
        
        if required_keys != provided_keys:
            return False
        
        self.weights = new_weights.copy()
        return True
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        return self.weights.copy()
