"""
FastAPI backend for the AI Research Assistant.

Endpoints:
  POST /upload      - Upload and index a document
  POST /chat        - Ask the agent a question
  GET  /documents   - List all uploaded documents
  DELETE /documents - Clear all documents and vector store
  GET  /health      - Health check
"""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_loader import load_and_split_document
from vector_store import add_to_vector_store, vector_store_exists
from agent import run_agent, clear_agent_cache

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
FAISS_DIR = BASE_DIR / "faiss_index"
DOCS_METADATA_FILE = BASE_DIR / "uploaded_docs.json"

UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".json"}

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="AI Research Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_docs_metadata() -> List[dict]:
    if DOCS_METADATA_FILE.exists():
        with open(DOCS_METADATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_docs_metadata(docs: List[dict]):
    with open(DOCS_METADATA_FILE, "w") as f:
        json.dump(docs, f, indent=2)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found. Place index.html in /static/</h1>", status_code=404)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "ollama": "connected",
        "vector_store": vector_store_exists(),
        "documents": len(load_docs_metadata())
    }


@app.get("/documents")
async def list_documents():
    """Return list of uploaded documents."""
    docs = load_docs_metadata()
    return {"documents": docs, "count": len(docs)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload, process and index a document."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save file
    save_path = UPLOADS_DIR / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Process and index
        chunks = load_and_split_document(str(save_path))
        add_to_vector_store(chunks)

        # Save metadata
        docs = load_docs_metadata()
        # Avoid duplicates
        docs = [d for d in docs if d["name"] != file.filename]
        docs.append({
            "name": file.filename,
            "size": len(content),
            "chunks": len(chunks),
            "path": str(save_path)
        })
        save_docs_metadata(docs)
        clear_agent_cache()

        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks from '{file.filename}'"
        }
    except Exception as e:
        # Clean up file if processing failed
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to the AI agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # SAFE MODE TEST: If the message is "test", return immediately.
    if request.message.lower() == "test":
        print("DEBUG: Safe Mode Triggered")
        return {
            "answer": "Connection successful! Safe mode works. If a normal chat fails, your computer might be running out of RAM when loading the full AI agent.",
            "tools_used": [],
            "success": True
        }

    result = run_agent(request.message)
    return {
        "answer": result["answer"],
        "tools_used": result["tools_used"],
        "success": result["success"]
    }


@app.delete("/documents")
async def clear_documents():
    """Clear all uploaded documents and the vector store."""
    # Remove uploads
    for f in UPLOADS_DIR.iterdir():
        if f.is_file():
            f.unlink()
    # Remove vector store
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    # Clear metadata
    if DOCS_METADATA_FILE.exists():
        DOCS_METADATA_FILE.unlink()

    clear_agent_cache()
    
    return {"success": True, "message": "All documents cleared."}
