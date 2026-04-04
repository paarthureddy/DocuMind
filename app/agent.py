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
        docs = _vector_db_cache.similarity_search(query, k=4)
        
        if not docs:
            print("DEBUG: No documents matched.")
            return "No relevant content found in the uploaded documents."
        
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Document")
            page   = doc.metadata.get("page", "")
            label  = str(os.path.basename(source)) + (f" (page {int(page)+1})" if page != "" else "")
            results.append(f"[{i}] From {label}:\n{str(doc.page_content).strip()}")
            
        print(f"DEBUG: Search returned {len(results)} snippets.")
        return "\n\n".join(results)
    except Exception as e:
        print(f"DEBUG: Search Tool Error: {str(e)}")
        return f"Document search failed: {str(e)}. Try asking a general web question instead."


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
TOOLS = [document_search, calculator, web_search]

SYSTEM_PROMPT = """You are DocMind, an advanced AI Research Assistant. Your goal is to provide insightful, accurate, and professional responses.

Decision Rules for Tool Usage:
1. **document_search**: Use this FIRST whenever the question is about "my documents", "the report", "uploaded files", or specific content that might be in a user's private file.
2. **calculator**: Use this strictly for math, percentages, currency conversions (if values are known), or data processing tasks involving numbers.
3. **web_search**: Use this for general knowledge, current news (after 2024), or when document_search yields no results.

Response Quality Requirements:
- **Never be vague.** Provide specific data points found.
- **Structure your response:** Use bullet points, bold text for emphasis, and clear headings if the answer is long.
- **Citation:** Explicitly state the source of your information. Example: "[1] According to the uploaded report..." or "[2] Web results suggest..."
- **Synthetic Reasoning:** If multiple tools provide data, synthesize them into a cohesive narrative.
- **Honesty:** If you cannot find the answer, explain what you searched for and why it wasn't there. Do not make up facts.
"""

def build_agent():
    llm = get_llm()
    # 'prompt' is the correct parameter name in langgraph >= 1.1.x
    agent = create_react_agent(llm, TOOLS, prompt=SystemMessage(content=SYSTEM_PROMPT))
    return agent


def run_agent(query: str) -> dict:
    """Runs the LangGraph agent and processes the response."""
    try:
        agent = get_ready_agent()

        initial_state = {"messages": [HumanMessage(content=query)]}
        result = agent.invoke(initial_state)

        # The final answer is the last message string
        messages = result.get("messages", [])
        final_answer = str(messages[-1].content) if messages else "No response."

        # Track tool calls and ensure items are strings (for JSON stability)
        tools_used = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "tool":  str(tc.get("name", "unknown")),
                        "input": str(tc.get("args", {}))
                    })

        return {
            "answer":     final_answer,
            "tools_used": tools_used,
            "success":    True,
        }

    except Exception as e:
        print(f"Agent Execution Error: {str(e)}")
        return {
            "answer":     f"I encountered an error: {str(e)}",
            "tools_used": [],
            "success":    False,
        }

# Pre-warm up the AI on import
get_ready_agent()
