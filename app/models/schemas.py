from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlanType(str, Enum):
    """Type of execution plan."""

    SINGLE_SHOT = "single_shot"
    MULTI_STEP = "multi_step"


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class RouteRequest(BaseModel):
    """Request to route an LLM call."""

    messages: list[Message] = Field(..., description="Conversation messages")
    plan_type: PlanType = Field(
        default=PlanType.SINGLE_SHOT, description="Execution plan type"
    )
    model: str | None = Field(
        default=None, description="Preferred model (optional, router will select if not provided)"
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata for routing decisions"
    )

    tournament: bool = Field(
        default=False,
        description="If true, run a model tournament (collect candidate responses + judge a winner).",
    )
    tournament_per_step: bool = Field(
        default=False,
        description="If true and plan_type is multi_step, run a tournament for each step.",
    )
    candidate_models: list[str] | None = Field(
        default=None,
        description="Optional explicit candidate model list for tournament mode.",
    )
    judge_model: str = Field(
        default="gpt-4o-mini",
        description="Judge model used to select the best candidate response.",
    )


class Step(BaseModel):
    """A single step in a multi-step execution plan."""

    step_number: int = Field(..., ge=1, description="Step order (1-indexed)")
    task: str = Field(..., description="Task description for this step")
    model: str | None = Field(
        default=None, description="Model selected for this step"
    )
    complexity: str | None = Field(
        default=None, description="Complexity tier for this step (simple, standard, complex, code)"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this step depends on"
    )
    output: str | None = Field(default=None, description="Output from this step")
    status: str = Field(default="pending", description="Step status: pending, running, completed, failed")


class AvailableModel(BaseModel):
    """An available model in the registry."""

    name: str = Field(..., description="Model name (short identifier)")
    provider: str = Field(..., description="Provider name (e.g., openrouter)")
    model: str = Field(..., description="Full model identifier")
    description: str = Field(..., description="Model description/capability summary")
    economy: str = Field(..., description="Cost tier (cheap, free)")
    responsiveness: str = Field(..., description="Speed tier (fast, medium, slow)")


class ModelOption(BaseModel):
    """A single model option that could be selected (with selection status)."""

    provider: str = Field(..., description="Provider name (e.g., openrouter)")
    model: str = Field(..., description="Model identifier")
    tier: str = Field(default="", description="Complexity tier (legacy, may be empty)")
    name: str = Field(default="", description="Model short name")
    description: str = Field(..., description="Model description/capability summary")
    economy: str = Field(default="", description="Cost tier (cheap, free)")
    responsiveness: str = Field(default="", description="Speed tier (fast, medium, slow)")
    selected: bool = Field(default=False, description="Whether this model was selected")


class SelectionResult(BaseModel):
    """Result of model/provider selection."""

    provider: str = Field(..., description="Selected provider (e.g., openai, openrouter)")
    model: str = Field(..., description="Selected model identifier")
    reasoning: str | None = Field(
        default=None, description="Explanation for the selection"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score for selection"
    )


class ClassifierInteraction(BaseModel):
    """The full interaction with the classifier model."""

    model: str = Field(..., description="Classifier model used")
    prompt_sent: str = Field(..., description="The prompt sent to the classifier")
    raw_response: dict = Field(..., description="Raw JSON response from classifier")
    latency_ms: float | None = Field(default=None, description="Classification latency")


class RoutingDecision(BaseModel):
    """Full routing decision with all considered options."""

    classifier_model: str = Field(
        default="gpt-4o-mini", description="Model used for classification"
    )
    classifier_interaction: ClassifierInteraction | None = Field(
        default=None, description="Full classifier interaction details"
    )
    available_models: list[ModelOption] = Field(
        ..., description="All models that were considered"
    )
    detected_complexity: str = Field(
        ..., description="Complexity level detected by classifier"
    )
    classification_reasoning: str = Field(
        ..., description="Why this complexity was assigned"
    )
    selection_reasoning: str = Field(
        ..., description="Why the selected model was chosen over others"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this routing decision"
    )


class ClassifyRequest(BaseModel):
    """Request to classify a prompt (phase 1 of two-phase flow)."""

    prompt: str = Field(..., description="User prompt to classify")


class ClassifyResponse(BaseModel):
    """Response from classification (phase 1)."""

    classifier_interaction: ClassifierInteraction = Field(
        ..., description="Full classifier interaction details"
    )
    classification: dict = Field(..., description="Parsed classification result")
    available_models: list[AvailableModel] = Field(
        ..., description="List of available models (names only for UI)"
    )


class SelectModelRequest(BaseModel):
    """Request for the Select a Model flow (phase 2a)."""

    original_prompt: str = Field(..., description="Original user prompt")
    selected_model_name: str | None = Field(
        default=None, description="User's selected model name (if pre-selected)"
    )
    user_preference: str = Field(
        default="", description="User's second input/preference for model selection"
    )
    classification: dict = Field(..., description="Classification from phase 1")


class SemanticSelectionDetails(BaseModel):
    """Details of semantic model selection."""

    method: str = Field(..., description="Selection method (semantic_selection, user_selected, fallback)")
    prompt_sent: str | None = Field(default=None, description="Prompt sent to selector")
    raw_response: dict | None = Field(default=None, description="Raw selector response")
    latency_ms: float | None = Field(default=None, description="Selection latency")
    selected_model: str = Field(..., description="Selected model name")
    reasoning: str | None = Field(default=None, description="Why this model was selected")


class CandidateResponse(BaseModel):
    """A single candidate model's response in tournament mode."""

    provider: str = Field(..., description="Provider name (e.g., openrouter)")
    model: str = Field(..., description="Model identifier")
    content: str | None = Field(default=None, description="Candidate response content")
    latency_ms: float | None = Field(default=None, description="Latency for this candidate")
    usage: dict[str, int] | None = Field(default=None, description="Token usage for this candidate")
    error: str | None = Field(default=None, description="Error message if the candidate failed")


class JudgeResult(BaseModel):
    """Judge decision for a tournament."""

    model: str = Field(..., description="Judge model used")
    winner_model: str = Field(..., description="Winning candidate model")
    reasoning: str = Field(..., description="Why the winner was chosen")
    scores: dict[str, float] | None = Field(
        default=None, description="Optional numeric scores keyed by candidate model"
    )


class TournamentResult(BaseModel):
    """Full tournament trace for a single decision point."""

    candidates: list[CandidateResponse] = Field(
        default_factory=list, description="All candidate responses"
    )
    judge: JudgeResult | None = Field(default=None, description="Judge decision")


class RouteResponse(BaseModel):
    """Response from a routed LLM call."""

    plan_type: PlanType = Field(..., description="Type of plan executed")
    selection: SelectionResult = Field(..., description="Model selection details")
    routing_decision: RoutingDecision | None = Field(
        default=None, description="Full routing decision with all considered models"
    )
    tournament: TournamentResult | None = Field(
        default=None, description="Tournament results for single-shot routing (if enabled)"
    )
    step_tournaments: dict[int, TournamentResult] | None = Field(
        default=None,
        description="Tournament results per step (if tournament_per_step is enabled)",
    )
    content: str | None = Field(
        default=None, description="Response content (for single-shot)"
    )
    steps: list[Step] | None = Field(
        default=None, description="Execution steps (for multi-step)"
    )
    usage: dict[str, int] | None = Field(
        default=None, description="Token usage statistics"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional response metadata"
    )
