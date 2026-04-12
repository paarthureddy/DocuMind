"""
Robust Intent Classification Module for AI Talent Search System.

Intelligently classifies user queries into:
- calculator: Pure mathematical expressions
- profile_search: Talent/casting related queries  
- chat: General conversation and other queries

This ensures proper routing and avoids false classifications.
"""

import re
from typing import List, Dict, Any

class IntentClassifier:
    """Robust intent classifier for query routing."""
    
    def __init__(self):
        # Keywords that indicate talent search intent
        self.talent_keywords = [
            "actor", "actress", "model", "singer", "dancer", "performer", "artist",
            "talent", "casting", "cast", "role", "character", "villain", "hero",
            "protagonist", "antagonist", "lead", "supporting", "comic", "romantic",
            "male", "female", "gender", "height", "feet", "ft", "inch", "in", "cm",
            "tall", "short", "complexion", "brown", "fair", "dark", "light",
            "wheatish", "handsome", "beautiful", "appearance", "looks",
            "skills", "experience", "craft", "profession", "search", "find",
            "looking for", "need", "want", "intense", "charming", "aggressive",
            "soft", "dominant", "comic", "funny", "serious", "dramatic"
        ]
        
        # Math operators allowed in pure math expressions
        self.math_operators = ['+', '-', '*', '/', '^', '(', ')', '.', ' ']
        
        # Compile regex patterns for performance
        self.pure_math_pattern = re.compile(r'^[\d+\-*/().^ ]+$')
        self.math_with_words_pattern = re.compile(r'[a-zA-Z]')
    
    def is_math_expression(self, query: str) -> bool:
        """
        Determine if query is a PURE math expression.
        
        Rules:
        - Must contain ONLY numbers and math operators
        - No letters or words allowed
        - Can include spaces and parentheses
        
        Examples:
        ✓ "20+20" → True
        ✓ "100 - 50" → True  
        ✓ "5*6+2" → True
        ✗ "actor + singer" → False (contains words)
        ✗ "height 6+ feet" → False (contains words)
        ✗ "20+20=" → False (contains = which is not allowed)
        """
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        
        # Quick check: if it contains any letters, it's not pure math
        if self.math_with_words_pattern.search(query):
            return False
        
        # Check if it matches the pure math pattern (only digits and operators)
        if not self.pure_math_pattern.match(query):
            return False
        
        # Additional validation: ensure it has at least one number and one operator
        has_numbers = bool(re.search(r'\d', query))
        has_operators = bool(re.search(r'[+\-*/^]', query))
        
        # Allow single numbers as valid math expressions (e.g., "42")
        return has_numbers
    
    def contains_talent_keywords(self, query: str) -> bool:
        """
        Check if query contains talent-related keywords.
        
        Case-insensitive search for any talent keywords.
        """
        if not query or not isinstance(query, str):
            return False
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.talent_keywords)
    
    def classify_intent(self, query: str) -> str:
        """
        Classify query intent into one of three categories.
        
        Args:
            query: User input string
            
        Returns:
            str: One of "calculator", "profile_search", or "chat"
            
        Logic Flow:
        1. Check if it's a pure math expression → calculator
        2. Check if it contains talent keywords → profile_search  
        3. Default to chat for everything else
        """
        if not query or not isinstance(query, str):
            return "chat"
        
        query = query.strip()
        if not query:
            return "chat"
        
        # Priority 1: Pure math expressions
        if self.is_math_expression(query):
            return "calculator"
        
        # Priority 2: Talent search queries
        if self.contains_talent_keywords(query):
            return "profile_search"
        
        # Priority 3: Everything else is chat
        return "chat"
    
    def get_classification_details(self, query: str) -> Dict[str, Any]:
        """
        Get detailed classification information for debugging.
        
        Returns:
            Dict with classification details and reasoning
        """
        if not query or not isinstance(query, str):
            return {
                "intent": "chat",
                "confidence": 0.0,
                "reasoning": "Invalid or empty query",
                "is_math": False,
                "has_talent_keywords": False,
                "matched_keywords": []
            }
        
        query = query.strip()
        
        # Check math expression
        is_math = self.is_math_expression(query)
        if is_math:
            return {
                "intent": "calculator",
                "confidence": 0.95,
                "reasoning": "Pure mathematical expression detected",
                "is_math": True,
                "has_talent_keywords": False,
                "matched_keywords": []
            }
        
        # Check talent keywords
        query_lower = query.lower()
        matched_keywords = [kw for kw in self.talent_keywords if kw in query_lower]
        has_talent_keywords = len(matched_keywords) > 0
        
        if has_talent_keywords:
            confidence = min(0.5 + (len(matched_keywords) * 0.1), 0.9)
            return {
                "intent": "profile_search",
                "confidence": confidence,
                "reasoning": f"Talent keywords found: {', '.join(matched_keywords[:3])}",
                "is_math": False,
                "has_talent_keywords": True,
                "matched_keywords": matched_keywords
            }
        
        # Default to chat
        return {
            "intent": "chat",
            "confidence": 0.3,
            "reasoning": "No specific intent detected, defaulting to chat",
            "is_math": False,
            "has_talent_keywords": False,
            "matched_keywords": []
        }
    
    def test_classification(self, test_cases: List[str]) -> List[Dict[str, Any]]:
        """
        Test classification on provided test cases.
        
        Args:
            test_cases: List of query strings to test
            
        Returns:
            List of classification results
        """
        results = []
        for query in test_cases:
            details = self.get_classification_details(query)
            results.append({
                "query": query,
                "intent": details["intent"],
                "confidence": details["confidence"],
                "reasoning": details["reasoning"]
            })
        return results

# Global instance for easy import
intent_classifier = IntentClassifier()

# Convenience functions for backward compatibility
def is_math_expression(query: str) -> bool:
    """Check if query is a pure math expression."""
    return intent_classifier.is_math_expression(query)

def classify_intent(query: str) -> str:
    """Classify query intent."""
    return intent_classifier.classify_intent(query)

# Test cases (as requested in comments)
if __name__ == "__main__":
    test_cases = [
        "20+20",           # → calculator
        "actor + singer",  # → profile_search  
        "male villain",    # → profile_search
        "hi",              # → chat
        "100 - 50",        # → calculator
        "height 6+ feet",  # → profile_search
        "what's weather",  # → chat
        "5*6+2",           # → calculator
        "female model tall",# → profile_search
        "hello there",     # → chat
        "sqrt(16)",        # → calculator (will be chat since sqrt has letters)
        "42",              # → calculator (single number)
    ]
    
    print("Intent Classification Test Results:")
    print("=" * 60)
    
    for test_case in test_cases:
        result = intent_classifier.get_classification_details(test_case)
        print(f"Input: '{test_case}'")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Reason: {result['reasoning']}")
        print("-" * 40)
