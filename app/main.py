from fastapi import FastAPI

app = FastAPI(
    title="LLM Router API",
    description="API for routing requests to different LLM providers",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
