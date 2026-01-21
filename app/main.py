from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.executor import execute_plan
from app.meta_router import route_request
from app.models.schemas import Message, PlanType, RouteRequest, RouteResponse

app = FastAPI(title="LLM Router API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest) -> RouteResponse:
    """Route a prompt through the meta-router and execute with selected model(s).

    Returns the complete execution flow including:
    - Plan type (single_shot or multi_step)
    - Model selection with reasoning
    - Response content or step-by-step outputs
    - Latency and usage metadata
    """
    # Get routing decision from meta-router
    routing_result = await route_request(request)

    # Extract plan details
    plan_type = PlanType(routing_result["type"])
    selection_data = routing_result["selection"]

    from app.models.schemas import SelectionResult, Step

    selection = SelectionResult(**selection_data)

    steps = None
    if "steps" in routing_result:
        steps = [Step(**s) for s in routing_result["steps"]]

    # Execute the plan
    response = await execute_plan(
        messages=request.messages,
        plan_type=plan_type,
        selection=selection,
        steps=steps,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Add classification reasoning to metadata
    if response.metadata is None:
        response.metadata = {}
    response.metadata["classification"] = routing_result.get("classification", {})

    return response
