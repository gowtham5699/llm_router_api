"""Meta-router for query classification, planning, and model selection.

Uses LiteLLM to call OpenAI for routing decisions.
"""

import json
from typing import Any

import litellm

from app.config import settings
from app.models.schemas import (
    Message,
    PlanType,
    RouteRequest,
    SelectionResult,
    Step,
)


# Model tiers for selection
MODEL_TIERS = {
    "simple": {
        "provider": "openrouter",
        "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "description": "Fast, efficient for simple tasks",
    },
    "standard": {
        "provider": "openrouter",
        "model": "openrouter/anthropic/claude-3.5-sonnet",
        "description": "Balanced performance and capability",
    },
    "complex": {
        "provider": "openrouter",
        "model": "openrouter/anthropic/claude-3-opus",
        "description": "High capability for complex reasoning",
    },
    "code": {
        "provider": "openrouter",
        "model": "openrouter/deepseek/deepseek-coder",
        "description": "Specialized for code generation",
    },
}


CLASSIFICATION_PROMPT = """Analyze the following user query and determine:
1. Whether it requires a single response or multiple steps
2. The complexity level (simple, standard, complex, code)

Respond in JSON format:
{
    "plan_type": "single_shot" or "multi_step",
    "complexity": "simple" | "standard" | "complex" | "code",
    "reasoning": "brief explanation"
}

If multi_step, also include:
{
    "plan_type": "multi_step",
    "complexity": "complex",
    "reasoning": "explanation",
    "steps": [
        {"step_number": 1, "task": "description", "complexity": "simple|standard|complex|code", "depends_on": []},
        {"step_number": 2, "task": "description", "complexity": "simple|standard|complex|code", "depends_on": [1]}
    ]
}

User query:
"""


class MetaRouter:
    """Routes queries using LLM-based classification and planning."""

    def __init__(self, openai_api_key: str | None = None):
        """Initialize the meta router.

        Args:
            openai_api_key: OpenAI API key. If not provided, uses settings.
        """
        self.openai_api_key = openai_api_key or settings.openai_api_key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for meta routing")

    async def classify_query(self, messages: list[Message]) -> dict[str, Any]:
        """Classify a query to determine plan type and complexity.

        Args:
            messages: Conversation messages to analyze.

        Returns:
            Classification result with plan_type, complexity, and optionally steps.
        """
        # Extract the latest user message for classification
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return {
                "plan_type": "single_shot",
                "complexity": "simple",
                "reasoning": "No user message found",
            }

        query = user_messages[-1].content

        # Call OpenAI via LiteLLM for classification
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a query classifier. Respond only with valid JSON.",
                },
                {"role": "user", "content": CLASSIFICATION_PROMPT + query},
            ],
            api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {
                "plan_type": "single_shot",
                "complexity": "standard",
                "reasoning": "Failed to parse classification response",
            }

    def select_model(self, complexity: str) -> SelectionResult:
        """Select the appropriate model based on complexity.

        Args:
            complexity: Task complexity level.

        Returns:
            SelectionResult with provider and model details.
        """
        tier = MODEL_TIERS.get(complexity, MODEL_TIERS["standard"])
        return SelectionResult(
            provider=tier["provider"],
            model=tier["model"],
            reasoning=tier["description"],
            confidence=0.85,
        )

    def create_plan(
        self, classification: dict[str, Any]
    ) -> tuple[PlanType, list[Step] | None, SelectionResult]:
        """Create an execution plan based on classification.

        Args:
            classification: Query classification result.

        Returns:
            Tuple of (plan_type, steps, primary_selection).
        """
        plan_type_str = classification.get("plan_type", "single_shot")
        plan_type = (
            PlanType.MULTI_STEP
            if plan_type_str == "multi_step"
            else PlanType.SINGLE_SHOT
        )

        complexity = classification.get("complexity", "standard")
        primary_selection = self.select_model(complexity)

        if plan_type == PlanType.SINGLE_SHOT:
            return plan_type, None, primary_selection

        # Build steps for multi-step plan
        step_data = classification.get("steps", [])
        steps = []
        for s in step_data:
            step_complexity = s.get("complexity", complexity)
            step_selection = self.select_model(step_complexity)
            steps.append(
                Step(
                    step_number=s.get("step_number", len(steps) + 1),
                    task=s.get("task", "Unknown task"),
                    model=step_selection.model,
                    depends_on=s.get("depends_on", []),
                    status="pending",
                )
            )

        # If no steps were provided, create a default single step
        if not steps:
            steps = [
                Step(
                    step_number=1,
                    task="Execute the query",
                    model=primary_selection.model,
                    depends_on=[],
                    status="pending",
                )
            ]

        return plan_type, steps, primary_selection

    async def route(self, request: RouteRequest) -> dict[str, Any]:
        """Route a request by classifying and creating an execution plan.

        Args:
            request: The route request with messages and options.

        Returns:
            Routing result with type and steps/selection.
        """
        # If user specified a plan type, respect it
        if request.plan_type == PlanType.MULTI_STEP:
            classification = await self.classify_query(request.messages)
            # Force multi-step even if classifier says otherwise
            classification["plan_type"] = "multi_step"
        else:
            classification = await self.classify_query(request.messages)

        plan_type, steps, selection = self.create_plan(classification)

        # If user specified a model, override the selection
        if request.model:
            selection = SelectionResult(
                provider="user_specified",
                model=request.model,
                reasoning="User specified model",
                confidence=1.0,
            )
            if steps:
                for step in steps:
                    step.model = request.model

        result = {
            "type": plan_type.value,
            "selection": selection.model_dump(),
            "classification": {
                "complexity": classification.get("complexity", "standard"),
                "reasoning": classification.get("reasoning", ""),
            },
        }

        if steps:
            result["steps"] = [step.model_dump() for step in steps]

        return result


# Singleton instance for convenience
_router_instance: MetaRouter | None = None


def get_meta_router() -> MetaRouter:
    """Get or create the singleton MetaRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = MetaRouter()
    return _router_instance


async def route_request(request: RouteRequest) -> dict[str, Any]:
    """Convenience function to route a request.

    Args:
        request: The route request.

    Returns:
        Routing result.
    """
    router = get_meta_router()
    return await router.route(request)
