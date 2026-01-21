"""API endpoints for LLM routing.

Orchestrates: receive prompt -> meta-router selection -> executor -> return response.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.executor import get_executor
from app.meta_router import get_meta_router
from app.models.schemas import (
    Message,
    PlanType,
    RouteRequest,
    RouteResponse,
    SelectionResult,
    Step,
)

router = APIRouter()


class PromptRequest(BaseModel):
    """Simple prompt request for the frontend."""

    prompt: str


class PromptResponse(BaseModel):
    """Simplified response for the frontend."""

    response: str
    model: str
    plan_type: str
    steps: list[dict] | None = None
    usage: dict | None = None


@router.post("/api/prompt", response_model=PromptResponse)
async def prompt(request: PromptRequest) -> PromptResponse:
    """Simple prompt endpoint for the frontend.

    Converts a simple prompt to the full RouteRequest format and returns
    a simplified response suitable for display.

    Args:
        request: PromptRequest with a single prompt string.

    Returns:
        PromptResponse with the LLM response and metadata.
    """
    # Convert simple prompt to RouteRequest format
    route_request = RouteRequest(
        messages=[Message(role="user", content=request.prompt)]
    )

    # Use the route endpoint logic
    route_response = await route(route_request)

    # Format response for frontend
    if route_response.plan_type == PlanType.MULTI_STEP and route_response.steps:
        # For multi-step, concatenate step outputs
        step_outputs = []
        for step in route_response.steps:
            step_outputs.append({
                "step": step.step_number,
                "task": step.task,
                "output": step.output,
                "status": step.status,
            })
        response_text = "\n\n".join(
            f"**Step {s['step']}: {s['task']}**\n{s['output'] or 'No output'}"
            for s in step_outputs
        )
        steps = step_outputs
    else:
        response_text = route_response.content or "No response generated"
        steps = None

    return PromptResponse(
        response=response_text,
        model=route_response.selection.model,
        plan_type=route_response.plan_type.value,
        steps=steps,
        usage=route_response.usage,
    )


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
