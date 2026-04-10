"""
Test script for intent classification API integration.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_intent_classification():
    """Test the intent classification through the API."""
    
    test_cases = [
        ("20+20", "calculator"),
        ("actor + singer", "profile_search"),
        ("male villain 6 feet", "profile_search"),
        ("hi", "chat"),
        ("100 - 50", "calculator"),
        ("height 6+ feet", "profile_search"),
        ("what's weather", "chat"),
        ("5*6+2", "calculator"),
        ("female model tall", "profile_search"),
        ("hello there", "chat"),
        ("42", "calculator")
    ]
    
    print("Testing Intent Classification via API")
    print("=" * 50)
    
    for query, expected_intent in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"message": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                actual_intent = data.get("intent", "unknown")
                status = "✅" if actual_intent == expected_intent else "❌"
                
                print(f"{status} '{query}' → {actual_intent} (expected: {expected_intent})")
                
                if actual_intent != expected_intent:
                    print(f"   Response: {data.get('answer', 'No answer')[:100]}...")
            else:
                print(f"❌ '{query}' → API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ '{query}' → Connection Error: Server not running")
            break
        except Exception as e:
            print(f"❌ '{query}' → Error: {str(e)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_intent_classification()
