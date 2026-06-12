"""
Talent search agent using LangGraph's ReAct agent.
Integrates with the talent search engine for intelligent casting assistance.
"""

import sys
import os
import math
import re

sys.path.insert(0, os.path.dirname(__file__))

import json
import urllib.request
from langchain_core.tools import tool
from query_parser import QueryParser
from retrieval import RetrievalEngine
from query_parser import QueryParser

# Global cache for the agent and retrieval engine
_agent_instance = None
_retrieval_engine = None

def clear_agent_cache():
    """Resets the agent cache."""
    global _agent_instance, _retrieval_engine
    _agent_instance = None
    _retrieval_engine = None

def get_ready_agent():
    """Ensures the agent and all tools are loaded and ready."""
    global _agent_instance, _retrieval_engine
    if _agent_instance is None:
        try:
            print("DEBUG: AI talent agent is warming up...")
            _retrieval_engine = RetrievalEngine()
            _agent_instance = build_agent()
            print("DEBUG: AI talent agent is ready.")
        except Exception as e:
            print(f"DEBUG: Agent warmup Error: {str(e)}")
    return _agent_instance

@tool
def profile_search(query: str) -> str:
    """
    Search for talent profiles based on natural language queries.
    Use this tool for finding actors, models, or other talent for casting.
    Supports queries like: 'male villain brown 6 feet intense actor'
    Input: a search query describing the desired talent profile.
    """
    global _retrieval_engine
    if not _retrieval_engine:
        _retrieval_engine = RetrievalEngine()
    
    try:
        print(f"DEBUG: Processing talent search for '{query}'...")
        search_result = _retrieval_engine.search_profiles(query, max_results=10)
        
        if not search_result["success"]:
            return f"Search failed: {search_result.get('error', 'Unknown error')}"
        
        results = search_result["results"]
        if not results:
            return "No talent profiles found matching your criteria. Try adjusting your search terms."
        
        # Format results for the agent
        response_parts = [f"Found {len(results)} talent profiles matching '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            name = result["name"]
            score = result["score"]
            explanation = result["explanation"]
            
            metadata = result["metadata"]
            details = []
            
            if metadata.get("gender"):
                details.append(f"Gender: {metadata['gender']}")
            if metadata.get("height_cm"):
                details.append(f"Height: {metadata['height_cm']}cm")
            if metadata.get("complexion"):
                details.append(f"Complexion: {metadata['complexion']}")
            if metadata.get("craft"):
                details.append(f"Craft: {metadata['craft']}")
            if metadata.get("experience_years"):
                details.append(f"Experience: {metadata['experience_years']} years")
            if metadata.get("rating"):
                details.append(f"Rating: {metadata['rating']}/5")
            
            response_parts.append(f"{i}. **{name}** (Score: {score:.2f})")
            response_parts.append(f"   {explanation}")
            if details:
                response_parts.append(f"   Details: {', '.join(details)}")
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Error searching talent profiles: {str(e)}"

@tool
def similar_profiles(profile_id: str) -> str:
    """
    Find talent profiles similar to a given profile ID.
    Use this tool when you want to find alternatives to a specific talent.
    Input: the profile ID of the reference talent.
    """
    global _retrieval_engine
    if not _retrieval_engine:
        _retrieval_engine = RetrievalEngine()
    
    try:
        print(f"DEBUG: Finding profiles similar to '{profile_id}'...")
        result = _retrieval_engine.get_similar_profiles(profile_id, max_results=8)
        
        if not result["success"]:
            return f"Similarity search failed: {result.get('error', 'Unknown error')}"
        
        similar = result["results"]
        if not similar:
            return "No similar profiles found."
        
        response_parts = [f"Talents similar to {result.get('reference_profile_name', profile_id)}:\n"]
        
        for i, profile in enumerate(similar, 1):
            name = profile["name"]
            similarity = profile["similarity_score"]
            
            metadata = profile["metadata"]
            details = []
            
            if metadata.get("craft"):
                details.append(metadata["craft"])
            if metadata.get("gender"):
                details.append(metadata["gender"])
            if metadata.get("height_bucket"):
                details.append(metadata["height_bucket"])
            
            response_parts.append(f"{i}. **{name}** (Similarity: {similarity:.2f})")
            if details:
                response_parts.append(f"   {', '.join(details)}")
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Error finding similar profiles: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """
    Safe math expression evaluator.
    Use this tool for mathematical calculations.
    Input: a mathematical expression like '2 + 3 * 4' or 'sqrt(16)'.
    """
    try:
        # Safe evaluation - only allow math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("_")
        }
        allowed_names.update({
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'len': len
        })
        
        # Remove potentially dangerous builtins
        expression = expression.replace('^', '**')
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return f"The result of {expression} is {result}"
        
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def web_search(query: str) -> str:
    """
    Search the web for information.
    Use this tool when you need current information beyond the talent database.
    Input: a search query string.
    """
    try:
        from duckduckgo_search import DDGS
        
        print(f"DEBUG: Performing web search for '{query}'...")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return "No web search results found."
        
        response_parts = [f"Web search results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            # Truncate long descriptions
            if len(body) > 200:
                body = body[:200] + "..."
            
            response_parts.append(f"{i}. **{title}**")
            response_parts.append(f"   {body}")
            response_parts.append(f"   Source: {href}")
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Web search failed: {str(e)}"

def build_agent():
    """Stubbed agent builder as we've moved to native Azure REST endpoints over Langchain agents."""
    return True

def run_agent(message: str) -> dict:
    """Run the talent conversation loop directly with Azure OpenAI via REST to bypass Ollama constraints."""
    try:
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-5-mini")
        if not AZURE_ENDPOINT or not AZURE_API_KEY:
            return {
                "answer": "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.",
                "tools_used": [],
                "success": False
            }
        
        base_url = AZURE_ENDPOINT.rstrip('/')
        url = f"{base_url}/openai/deployments/{MODEL_NAME}/chat/completions?api-version={AZURE_API_VERSION}"
        
        headers = {
            "api-key": AZURE_API_KEY,
            "Content-Type": "application/json"
        }
        
        system_prompt = (
            "You are an AI-powered casting assistant for the entertainment industry. "
            "Your expertise is in finding the perfect talent for any role or project. "
            "Always provide helpful, professional casting advice and explain your recommendations. "
            "You represent the 'Tribli Talent Search Engine'."
        )
        
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7
        }

        print(f"DEBUG: Running Azure agent with message: {message}")
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            answer = res_data['choices'][0]['message']['content']
        
        return {
            "answer": answer,
            "tools_used": [],
            "success": True
        }
        
    except Exception as e:
        print(f"Agent execution error via Azure: {e}")
        return {
            "answer": f"Error communicating with Azure OpenAI: {str(e)}",
            "tools_used": [],
            "success": False
        }

def is_talent_query(message: str) -> bool:
    """Check if a message is related to talent search/casting."""
    message_lower = message.lower().strip()
    
    # First check for mathematical expressions
    if re.search(r'[\d+\-*/().^]+', message) and len(message.split()) <= 5:
        # Contains math operators and is short - likely a calculation
        return False
    
    # Check for calculation keywords
    calc_keywords = ["calculate", "compute", "solve", "what is", "equals", "sum", "add", "subtract", "multiply", "divide"]
    if any(keyword in message_lower for keyword in calc_keywords):
        return False
    
    # Check for pure numbers and operators
    if re.match(r'^[\d+\-*/().^ ]+$', message.strip()):
        return False
    
    # Now check for talent-related keywords
    talent_keywords = [
        "actor", "actress", "model", "talent", "casting", "cast", "role",
        "character", "villain", "hero", "lead", "supporting", "comic",
        "male", "female", "height", "complexion", "appearance", "skills",
        "experience", "profile", "search", "find", "looking for", "need"
    ]
    
    # Check for physical attributes
    physical_keywords = [
        "feet", "ft", "inch", "in", "cm", "tall", "short", "brown", "fair",
        "dark", "light", "wheatish", "complexion", "handsome", "beautiful"
    ]
    
    # Check for role/character types
    role_keywords = [
        "villain", "hero", "protagonist", "antagonist", "lead", "supporting",
        "comic", "romantic", "character", "negative", "positive", "main"
    ]
    
    # Check for personality traits
    personality_keywords = [
        "intense", "charming", "aggressive", "soft", "dominant", "comic",
        "funny", "serious", "dramatic", "bold", "subtle"
    ]
    
    # Check for craft/profession
    craft_keywords = [
        "actor", "actress", "model", "dancer", "singer", "performer", "artist"
    ]
    
    all_keywords = talent_keywords + physical_keywords + role_keywords + personality_keywords + craft_keywords
    
    return any(keyword in message_lower for keyword in all_keywords)
