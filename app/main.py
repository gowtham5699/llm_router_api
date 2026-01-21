from fastapi import FastAPI

app = FastAPI(title="LLM Router API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
