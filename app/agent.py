"""
Agentic AI core using LangGraph's ReAct agent.
Compatible with llama3.1 which supports native tool-calling.

Tools:
  1. document_search  – searches uploaded documents (RAG/FAISS)
  2. calculator       – safe math expression evaluator
  3. web_search       – DuckDuckGo web search
"""

import sys
import os
import math
import re

sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from vector_store import load_vector_store, vector_store_exists

# Global cache for the agent and vector store
_agent_instance = None
_vector_db_cache = None

def clear_agent_cache():
    """Resets the vector store cache so deleted documents don't persist."""
    global _vector_db_cache
    _vector_db_cache = None

def get_ready_agent():
    """Ensures the agent and all tools are loaded and ready."""
    global _agent_instance, _vector_db_cache
    if _agent_instance is None:
        try:
            print("DEBUG: AI is warming up models and vector store...")
            if vector_store_exists():
                _vector_db_cache = load_vector_store()
            _agent_instance = build_agent()
            print("DEBUG: AI is ready.")
        except Exception as e:
            print(f"DEBUG: Warmup Error: {str(e)}")
    return _agent_instance

@tool
def document_search(query: str) -> str:
    """
    Search uploaded documents for information relevant to the query.
    Use this tool first when the question is about uploaded files or documents.
    Input: a search query string.
    """
    global _vector_db_cache
    if not vector_store_exists():
        return "No documents uploaded yet. Please upload a document first."
    try:
        # Fallback if cache missed
        if _vector_db_cache is None:
            _vector_db_cache = load_vector_store()
        
        # This is where Windows crashes might occur (FAISS/Torch)
        print(f"DEBUG: Processing document search for '{query}'...")
        docs = _vector_db_cache.similarity_search(query, k=50)
        
        query_lower = query.lower()
        strict_male = re.search(r'\b(male|man|boy)\b', query_lower) and not re.search(r'\b(female|woman|girl|actress)\b', query_lower)
        strict_female = re.search(r'\b(female|woman|girl|actress)\b', query_lower)

        filtered_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Explicit Social-Media-Level Hard Filtering
            if strict_male and "gender: female" in content_lower:
                continue
            if strict_female and "gender: male" in content_lower and "gender: female" not in content_lower:
                continue
                
            filtered_docs.append(doc)
            if len(filtered_docs) >= 15:
                break
        
        if not filtered_docs:
            print("DEBUG: No documents matched after strict physical filtering.")
            return "No relevant content found matching strict attributes."
        
        results = []
        for i, doc in enumerate(filtered_docs, 1):
            source = doc.metadata.get("source", "Document")
            page_val = str(doc.metadata.get("page", ""))
            
            # Simple string representation
            if page_val:
                # If page is numeric, it's a PDF page. Otherwise it's our Actor ID strings.
                if page_val.isdigit():
                    page_label = f" (page {int(page_val) + 1})"
                else:
                    page_label = f" (Actor: {page_val})"
            else:
                page_label = ""
                
            label = str(os.path.basename(source)) + page_label
            results.append(f"[{i}] From {label}:\n{str(doc.page_content).strip()}")
            
        print(f"DEBUG: Search returned {len(results)} snippets.")
        return "\n\n".join(results)
    except Exception as e:
        print(f"DEBUG: Search Tool Error: {str(e)}")
        return f"Document search failed: {str(e)}."


# ─────────────────────────────────────────────
# Tool 2: Calculator
# ─────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use for arithmetic, percentages, or any numeric computation.
    Input: a math expression like '25 * 4 + 10' or 'sqrt(144)'.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e
    """
    try:
        expression = expression.lower().replace(" x ", " * ").replace(" times ", " * ")

        # Handle "X% of Y" pattern (e.g. "15% of 200")
        pct_match = re.search(r'([\d.]+)\s*%\s*(?:of|times)\s*([\d.]+)', expression, re.I)
        if pct_match:
            pct, total = float(pct_match.group(1)), float(pct_match.group(2))
            return f"Result: {pct * total / 100}"

        # Keep %, keep a-z for sqrt/sin/cos/log etc.
        cleaned = re.sub(r'[^0-9+\-*/().,\s%^a-z_]', '', expression)
        cleaned = cleaned.replace('^', '**')

        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(cleaned, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculator error for '{expression}': {str(e)}"


# ─────────────────────────────────────────────
# Tool 3: Web Search (DuckDuckGo)
# ─────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the internet for current events, facts, or any information
    not available in uploaded documents.
    Input: a clear search query string.
    Returns top 4 results.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found. This might be because the internet is disconnected or restricted."
        out = []
        for i, r in enumerate(results, 1):
            title   = r.get('title', 'No title')
            snippet = r.get('body', 'No description')
            source  = r.get('href', '')
            out.append(f"[{i}] {title}\n{snippet}\nSource: {source}")
        return "\n\n".join(out)
    except Exception as e:
        return f"Web search tool failed. Please check your internet connection. Error: {str(e)}"


# ─────────────────────────────────────────────
# Build LangGraph ReAct Agent
# ─────────────────────────────────────────────
TOOLS = [document_search]

SYSTEM_PROMPT = """You are an expert Casting Director and AI Assistant for an Actor Database. Your job is to find the closest matching actors from the uploaded documents.

CRITICAL RULES:
1. You MUST ALWAYS use the `document_search` tool to fetch actors. If the user asks for multiple *separate* roles (e.g., "a female singer AND an actress"), you MUST invoke `document_search` multiple times in parallel, once for each role! DO NOT compress them into a single query because no one profile holds both.
2. The vector database may return partial matches. You MUST act as a STRICT FILTER.
3. If the user explicitly asks for a MAN/MALE, DO NOT present any FEMALE actors, and vice versa. Keep strict bounds on Ethnicity and Gender requests.
4. You MUST return your final answer EXCLUSIVELY as a valid JSON array of objects.
5. If no profiles match, return an empty array: []
6. If profiles do match, return them strictly ordered from BEST to LEAST match. EACH object MUST have these exact keys: "name", "age", "height", "skills", "rating", "reason".
7. NEVER wrap the JSON in markdown formatting (like ```json). Return ONLY the raw JSON string starting with [ and ending with ]. DO NOT say "Here are the top profiles...". ONLY JSON.
"""

import json

def parse_profile(content: str) -> dict:
    profile = {
        "name": "Unknown",
        "age": "N/A",
        "height": "N/A",
        "skills": "N/A",
        "rating": "N/A",
        "reason": "Direct Semantic Match"
    }
    for line in content.split("\n"):
        if line.startswith("Actor Name:"):
            profile["name"] = line.replace("Actor Name:", "").split("(ID:")[0].strip()
        elif line.startswith("Age:"):
            profile["age"] = line.replace("Age:", "").strip()
        elif line.startswith("Height:"):
            profile["height"] = line.replace("Height:", "").replace("cm", "").strip()
        elif line.startswith("Skills:"):
            profile["skills"] = line.replace("Skills:", "").strip()
        elif line.startswith("Rating:"):
            profile["rating"] = line.replace("Rating:", "").strip()
    return profile

def run_agent(query: str) -> dict:
    """Directly queries the Vector DB and returns a guaranteed strict JSON payload, entirely bypassing LLM hallucinations."""
    try:
        global _vector_db_cache
        if _vector_db_cache is None:
            if vector_store_exists():
                _vector_db_cache = load_vector_store()
            else:
                return {"answer": "[]", "tools_used": [], "success": False}
        
        # 1. Retrieve deep contextual matches
        docs = _vector_db_cache.similarity_search(query, k=150)
        
        # 2. Strict Social-Media Filtering Logic
        query_lower = query.lower()
        strict_male = re.search(r'\b(male|man|boy|actor)\b', query_lower) and not re.search(r'\b(female|woman|girl|actress)\b', query_lower)
        strict_female = re.search(r'\b(female|woman|girl|actress)\b', query_lower)

        profiles = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Explicit Social-Media-Level Hard Filtering
            if strict_male and "gender: female" in content_lower:
                continue
            if strict_female and "gender: male" in content_lower and "gender: female" not in content_lower:
                continue
                
            profiles.append(parse_profile(doc.page_content))
            
            if len(profiles) >= 30: # Return top 30 profiles instantly
                break
                
        return {
            "answer": json.dumps(profiles),
            "tools_used": [{"tool": "Semantic Vector Engine", "input": query}],
            "success": True,
        }
    except Exception as e:
        print(f"Database Query Error: {str(e)}")
        return {
            "answer": "[]",
            "tools_used": [],
            "success": False,
        }

# Pre-warm DB
if vector_store_exists():
    _vector_db_cache = load_vector_store()
