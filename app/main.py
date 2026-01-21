from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router

# Get the project root directory (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(title="LLM Router API", version="0.1.0")

app.include_router(router)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")


@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/styles.css")
async def serve_css():
    """Serve the CSS file."""
    return FileResponse(PROJECT_ROOT / "styles.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript file."""
    return FileResponse(PROJECT_ROOT / "app.js", media_type="application/javascript")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
