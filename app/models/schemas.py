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


class Step(BaseModel):
    """A single step in a multi-step execution plan."""

    step_number: int = Field(..., ge=1, description="Step order (1-indexed)")
    task: str = Field(..., description="Task description for this step")
    model: str | None = Field(
        default=None, description="Model selected for this step"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Step numbers this step depends on"
    )
    output: str | None = Field(default=None, description="Output from this step")
    status: str = Field(default="pending", description="Step status: pending, running, completed, failed")


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


class RouteResponse(BaseModel):
    """Response from a routed LLM call."""

    plan_type: PlanType = Field(..., description="Type of plan executed")
    selection: SelectionResult = Field(..., description="Model selection details")
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
