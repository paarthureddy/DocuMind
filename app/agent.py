"""
Agentic AI core using LangGraph's ReAct agent.
Tools:
  1. document_search  – searches uploaded documents (RAG/FAISS)
  2. calculator       – safe math expression evaluator
  3. web_search       – DuckDuckGo web search
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import re

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from llm import get_llm
from vector_store import load_vector_store, vector_store_exists


# ─────────────────────────────────────────────
# Tool 1: Document Search (RAG)
# ─────────────────────────────────────────────
@tool
def document_search(query: str) -> str:
    """
    Search uploaded documents for information relevant to the query.
    Use this tool first when the question is about uploaded files or documents.
    Input: a search query string.
    """
    if not vector_store_exists():
        return "No documents uploaded yet. Please upload a document first."
    try:
        vector_db = load_vector_store()
        docs = vector_db.similarity_search(query, k=4)
        if not docs:
            return "No relevant content found in the uploaded documents."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Document")
            page   = doc.metadata.get("page", "")
            label  = os.path.basename(source) + (f" (page {int(page)+1})" if page != "" else "")
            results.append(f"[{i}] From {label}:\n{doc.page_content.strip()}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching documents: {str(e)}"


# ─────────────────────────────────────────────
# Tool 2: Calculator
# ─────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use for arithmetic, percentages, or any numeric computation.
    Input: a math expression like '25 * 4 + 10' or 'sqrt(144)'.
    """
    try:
        cleaned = re.sub(r'[^0-9+\-*/().,\s%^a-zA-Z_]', '', expression)
        cleaned = cleaned.replace('^', '**')
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result  = eval(cleaned, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


# ─────────────────────────────────────────────
# Tool 3: Web Search (DuckDuckGo)
# ─────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the internet for current events, facts, or any information
    not available in uploaded documents.
    Input: a clear search query string.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        out = []
        for i, r in enumerate(results, 1):
            out.append(f"[{i}] {r.get('title','')}\n{r.get('body','')}\nSource: {r.get('href','')}")
        return "\n\n".join(out)
    except Exception as e:
        return f"Web search error: {str(e)}"


# ─────────────────────────────────────────────
# Build LangGraph ReAct Agent
# ─────────────────────────────────────────────
TOOLS = [document_search, calculator, web_search]

SYSTEM_PROMPT = """You are DocMind, an AI Research Assistant. You have access to three tools:

1. document_search – searches the user's uploaded documents
2. calculator      – evaluates math expressions  
3. web_search      – searches the internet for current information

Decision rules:
- If the question is about uploaded documents/files → use document_search first
- If the question involves numbers or math → use calculator
- If the question is about current events or general facts not in documents → use web_search
- You may use multiple tools if needed

Always provide a comprehensive, well-structured final answer."""


def build_agent():
    llm = get_llm()
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def run_agent(query: str) -> dict:
    """Run the LangGraph agent and return structured result with tool usage."""
    try:
        agent = build_agent()
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        messages = result.get("messages", [])

        # Extract final AI answer
        final_answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                if msg.content.strip():
                    final_answer = msg.content
                    break

        # Extract tool usage from ToolMessage steps
        tools_used = []
        for msg in messages:
            # AIMessage with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "tool": tc.get("name", "unknown"),
                        "input": str(tc.get("args", {})),
                        "output": ""  # filled in from ToolMessages below
                    })
            # ToolMessage (observation)
            if hasattr(msg, "name") and hasattr(msg, "content"):
                for tu in tools_used:
                    if tu["tool"] == msg.name and not tu["output"]:
                        content = msg.content
                        tu["output"] = content[:300] + "..." if len(content) > 300 else content
                        break

        if not final_answer:
            final_answer = "I wasn't able to generate an answer. Please try rephrasing."

        return {
            "answer": final_answer,
            "tools_used": tools_used,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "tools_used": [],
            "success": False
        }
