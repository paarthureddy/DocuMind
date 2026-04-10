"""
FastAPI backend for the Talent Search Engine.

This is now a wrapper around the dedicated talent_api.py module.
Maintains backward compatibility with existing frontend.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from talent_api import app
from fastapi.staticfiles import StaticFiles

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# Export the app for uvicorn
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
