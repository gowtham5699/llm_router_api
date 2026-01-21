"""API endpoints for LLM routing.

Orchestrates: receive prompt -> meta-router selection -> executor -> return response.
"""

from fastapi import APIRouter, HTTPException

from app.executor import get_executor
from app.meta_router import get_meta_router
from app.models.schemas import (
    PlanType,
    RouteRequest,
    RouteResponse,
    SelectionResult,
    Step,
)

router = APIRouter()


@router.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest) -> RouteResponse:
    """Route a prompt through meta-router and execute with selected model.

    Orchestrates the full flow:
    1. Receive prompt and options
    2. Meta-router classifies query and selects model
    3. Executor runs the selected model
    4. Return aggregated response with metadata

    Args:
        request: RouteRequest with messages and optional parameters.

    Returns:
        RouteResponse with content/steps and metadata.
    """
    try:
        meta_router = get_meta_router()
        executor = get_executor()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")

    # Step 1: Route the request through meta-router
    routing_result = await meta_router.route(request)

    # Step 2: Extract routing decision
    plan_type = (
        PlanType.MULTI_STEP
        if routing_result["type"] == "multi_step"
        else PlanType.SINGLE_SHOT
    )

    selection = SelectionResult(**routing_result["selection"])

    # Extract steps if multi-step plan
    steps = None
    if "steps" in routing_result and routing_result["steps"]:
        steps = [Step(**step_data) for step_data in routing_result["steps"]]

    # Step 3: Execute the plan
    response = await executor.execute(
        messages=request.messages,
        plan_type=plan_type,
        selection=selection,
        steps=steps,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Step 4: Enrich metadata with classification info
    if response.metadata is None:
        response.metadata = {}
    response.metadata["classification"] = routing_result.get("classification", {})

    return response
