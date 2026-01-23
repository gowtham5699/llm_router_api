"""API endpoints for LLM routing.

Orchestrates: receive prompt -> meta-router selection -> executor -> return response.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.executor import get_executor
from app.meta_router import get_meta_router, AVAILABLE_MODELS
from app.models.schemas import (
    AvailableModel,
    CandidateResponse,
    ClassifierInteraction,
    ClassifyRequest,
    ClassifyResponse,
    JudgeResult,
    Message,
    PlanType,
    RouteRequest,
    RouteResponse,
    RoutingDecision,
    SelectionResult,
    SelectModelRequest,
    SemanticSelectionDetails,
    Step,
    TournamentResult,
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


@router.post("/prompt", response_model=PromptResponse)
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


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Phase 1: Classify a prompt and return classifier interaction.

    Args:
        request: ClassifyRequest with the user prompt.

    Returns:
        ClassifyResponse with classifier interaction and available models.
    """
    try:
        meta_router = get_meta_router()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")

    # Classify the query
    messages = [Message(role="user", content=request.prompt)]
    classification, classifier_interaction = await meta_router.classify_query(messages)

    # Build available models list (names only for UI)
    available_models = [
        AvailableModel(
            name=name,
            provider=info["provider"],
            model=info["model"],
            description=info["description"],
            economy=info["economy"],
            responsiveness=info["responsiveness"],
        )
        for name, info in AVAILABLE_MODELS.items()
    ]

    return ClassifyResponse(
        classifier_interaction=classifier_interaction,
        classification=classification,
        available_models=available_models,
    )


@router.post("/select-model", response_model=RouteResponse)
async def select_model_and_execute(request: SelectModelRequest) -> RouteResponse:
    """Phase 2a: Select a model semantically and execute.

    Uses gpt-4o-mini to semantically assess which model is best for the task,
    then executes with that model.

    Args:
        request: SelectModelRequest with original prompt, selected model (optional),
                 and user preference.

    Returns:
        RouteResponse with the execution result.
    """
    try:
        meta_router = get_meta_router()
        executor = get_executor()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")

    # Semantic model selection
    selection, semantic_details = await meta_router.semantic_model_selection(
        task=request.original_prompt,
        user_preference=request.user_preference,
        selected_model_name=request.selected_model_name,
    )

    # Determine plan type from classification
    plan_type_str = request.classification.get("plan_type", "single_shot")
    plan_type = PlanType.MULTI_STEP if plan_type_str == "multi_step" else PlanType.SINGLE_SHOT

    messages = [Message(role="user", content=request.original_prompt)]

    # For multi-step, we need to build steps with semantic selection per step
    steps = None
    if plan_type == PlanType.MULTI_STEP:
        step_data = request.classification.get("steps", [])
        steps = []
        for s in step_data:
            # For each step, do semantic selection based on the step task
            step_selection, _ = await meta_router.semantic_model_selection(
                task=s.get("task", ""),
                user_preference=request.user_preference,
                selected_model_name=request.selected_model_name,
            )
            steps.append(
                Step(
                    step_number=s.get("step_number", len(steps) + 1),
                    task=s.get("task", "Unknown task"),
                    model=step_selection.model,
                    complexity=s.get("complexity", ""),
                    depends_on=s.get("depends_on", []),
                    status="pending",
                )
            )
        if not steps:
            steps = [
                Step(
                    step_number=1,
                    task="Execute the query",
                    model=selection.model,
                    depends_on=[],
                    status="pending",
                )
            ]

    # Execute
    response = await executor.execute(
        messages=messages,
        plan_type=plan_type,
        selection=selection,
        steps=steps,
        temperature=None,
        max_tokens=None,
    )

    # Add semantic selection details to metadata
    if response.metadata is None:
        response.metadata = {}
    response.metadata["semantic_selection"] = semantic_details
    response.metadata["classification"] = request.classification

    return response

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

    # Extract the full routing decision
    routing_decision = None
    if "routing_decision" in routing_result:
        routing_decision = RoutingDecision(**routing_result["routing_decision"])

    # Extract steps if multi-step plan
    steps = None
    if "steps" in routing_result and routing_result["steps"]:
        steps = [Step(**step_data) for step_data in routing_result["steps"]]

    # Step 3: Execute the plan
    # Candidate model pool for tournaments
    default_candidate_models: list[str] = []
    if routing_decision:
        default_candidate_models = [m.model for m in routing_decision.available_models]

    candidate_models = request.candidate_models or default_candidate_models

    # Single-shot tournament mode: run all candidates + judge winner, then return winner response
    if request.tournament and plan_type == PlanType.SINGLE_SHOT and candidate_models:
        msg_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        winner_model, tournament = await executor.run_tournament(
            messages=msg_dicts,
            candidate_models=candidate_models,
            judge_model=request.judge_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Pick the winning candidate response (fallback to first successful)
        winner_candidate = None
        for c in tournament.candidates:
            if c.model == winner_model and c.content is not None:
                winner_candidate = c
                break
        if winner_candidate is None:
            winner_candidate = next((c for c in tournament.candidates if c.content is not None), None)

        if winner_candidate is None:
            raise HTTPException(status_code=500, detail="All candidate models failed in tournament")

        selection.model = winner_candidate.model
        selection.provider = winner_candidate.provider

        # Keep routing_decision consistent with the final (tournament) winner.
        if routing_decision is not None:
            for opt in routing_decision.available_models:
                opt.selected = opt.model == winner_candidate.model
            routing_decision.selection_reasoning = (
                "Tournament mode enabled. The router suggested a model tier based on classification, "
                f"but the judge selected '{winner_candidate.model}' as the victor after comparing candidate outputs. "
                f"Judge reasoning is included in the tournament trace."
            )

        response = RouteResponse(
            plan_type=PlanType.SINGLE_SHOT,
            selection=selection,
            routing_decision=routing_decision,
            tournament=tournament,
            content=winner_candidate.content,
            steps=None,
            usage=winner_candidate.usage,
            metadata={
                "latency_ms": winner_candidate.latency_ms,
                "executed_model": winner_candidate.model,
            },
        )
    else:
        # Normal execution, optionally with per-step tournament
        response = await executor.execute(
            messages=request.messages,
            plan_type=plan_type,
            selection=selection,
            steps=steps,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tournament_per_step=bool(request.tournament_per_step),
            candidate_models=candidate_models if request.tournament_per_step else None,
            judge_model=request.judge_model,
        )

    # Step 4: Enrich response with routing decision and classification info
    response.routing_decision = routing_decision
    if response.metadata is None:
        response.metadata = {}
    response.metadata["classification"] = routing_result.get("classification", {})

    return response
