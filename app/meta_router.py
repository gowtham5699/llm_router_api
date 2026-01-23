"""Meta-router for query classification, planning, and model selection.

Uses LiteLLM to call OpenAI for routing decisions.
"""

import json
from typing import Any

import litellm

from app.config import settings
import time

from app.models.schemas import (
    ClassifierInteraction,
    Message,
    ModelOption,
    PlanType,
    RouteRequest,
    RoutingDecision,
    SelectionResult,
    Step,
)


# Available models - flat dictionary without static capability tiers
# Capability is assessed dynamically by gpt-4o-mini based on task requirements
AVAILABLE_MODELS = {
    "llama-3.1-8b": {
        "provider": "openrouter",
        "model": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "description": "Fast, efficient for simple tasks",
        "economy": "cheap",
        "responsiveness": "fast",
    },
    "nemotron-nano-9b": {
        "provider": "openrouter",
        "model": "openrouter/nvidia/nemotron-nano-9b-v2:free",
        "description": "Balanced performance and capability",
        "economy": "free",
        "responsiveness": "medium",
    },
    "gpt-oss-120b": {
        "provider": "openrouter",
        "model": "openrouter/openai/gpt-oss-120b:free",
        "description": "High capability for complex reasoning",
        "economy": "free",
        "responsiveness": "slow",
    },
    "nemotron-3-nano-30b": {
        "provider": "openrouter",
        "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
        "description": "Specialized for code generation",
        "economy": "free",
        "responsiveness": "medium",
    },
}

# Legacy mapping for backward compatibility
MODEL_TIERS = {
    "simple": AVAILABLE_MODELS["llama-3.1-8b"],
    "standard": AVAILABLE_MODELS["nemotron-nano-9b"],
    "complex": AVAILABLE_MODELS["gpt-oss-120b"],
    "code": AVAILABLE_MODELS["nemotron-3-nano-30b"],
}


SEMANTIC_MODEL_SELECTION_PROMPT = """You are a model selection expert. Given a task and available models, select the best model.

Available models:
{models_info}

User's task/prompt:
{task}

User's preference (if any):
{preference}

Analyze the task requirements and select the best model. Consider:
1. Task complexity and requirements
2. Model capabilities based on description
3. Economy preference (cheap/free)
4. Responsiveness needs (fast/medium/slow)

Respond in JSON format:
{{
    "selected_model": "model_name_from_list",
    "reasoning": "why this model is best for this task",
    "confidence": 0.0 to 1.0
}}
"""

CLASSIFICATION_PROMPT = """Analyze the following user query and determine:
1. Whether it requires a single response or multiple steps
2. The complexity level (simple, standard, complex, code)
3. Your confidence in this classification (0.0 to 1.0)

Confidence guidelines:
- 0.9-1.0: Very clear classification (e.g., "What is 2+2?" is clearly simple)
- 0.7-0.9: Fairly confident but some ambiguity
- 0.5-0.7: Uncertain, could reasonably be classified differently
- Below 0.5: Very uncertain, classification is a guess

Respond in JSON format:
{
    "plan_type": "single_shot" or "multi_step",
    "complexity": "simple" | "standard" | "complex" | "code",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}

If multi_step, also include:
{
    "plan_type": "multi_step",
    "complexity": "complex",
    "confidence": 0.0 to 1.0,
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

    async def classify_query(self, messages: list[Message]) -> tuple[dict[str, Any], ClassifierInteraction]:
        """Classify a query to determine plan type and complexity.

        Args:
            messages: Conversation messages to analyze.

        Returns:
            Tuple of (classification result, classifier interaction details).
        """
        # Extract the latest user message for classification
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return {
                "plan_type": "single_shot",
                "complexity": "simple",
                "reasoning": "No user message found",
            }, ClassifierInteraction(
                model="none",
                prompt_sent="No user message",
                raw_response={"error": "No user message found"},
                latency_ms=0,
            )

        query = user_messages[-1].content
        full_prompt = CLASSIFICATION_PROMPT + query

        # Call OpenAI via LiteLLM for classification
        start_time = time.perf_counter()
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a query classifier. Respond only with valid JSON.",
                },
                {"role": "user", "content": full_prompt},
            ],
            api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_text = response.choices[0].message.content
        try:
            parsed_result = json.loads(result_text)
            interaction = ClassifierInteraction(
                model="gpt-4o-mini",
                prompt_sent=full_prompt,
                raw_response=parsed_result,
                latency_ms=latency_ms,
            )
            return parsed_result, interaction
        except json.JSONDecodeError:
            interaction = ClassifierInteraction(
                model="gpt-4o-mini",
                prompt_sent=full_prompt,
                raw_response={"raw_text": result_text, "error": "Failed to parse JSON"},
                latency_ms=latency_ms,
            )
            return {
                "plan_type": "single_shot",
                "complexity": "standard",
                "reasoning": "Failed to parse classification response",
            }, interaction

    async def semantic_model_selection(
        self,
        task: str,
        user_preference: str = "",
        selected_model_name: str | None = None,
    ) -> tuple[SelectionResult, dict[str, Any]]:
        """Use gpt-4o-mini to semantically select the best model for a task.

        Args:
            task: The user's task/prompt to analyze.
            user_preference: Optional user preference for model selection.
            selected_model_name: If provided, use this model directly (user pre-selected).

        Returns:
            Tuple of (SelectionResult, semantic_selection_details).
        """
        # If user pre-selected a model, use it directly
        if selected_model_name and selected_model_name in AVAILABLE_MODELS:
            model_info = AVAILABLE_MODELS[selected_model_name]
            return SelectionResult(
                provider=model_info["provider"],
                model=model_info["model"],
                reasoning=f"User selected {selected_model_name}",
                confidence=1.0,
            ), {
                "method": "user_selected",
                "selected_model": selected_model_name,
                "reasoning": "User directly selected this model",
            }

        # Build models info string for the prompt
        models_info = "\n".join([
            f"- {name}: {info['description']} (economy: {info['economy']}, responsiveness: {info['responsiveness']})"
            for name, info in AVAILABLE_MODELS.items()
        ])

        prompt = SEMANTIC_MODEL_SELECTION_PROMPT.format(
            models_info=models_info,
            task=task,
            preference=user_preference or "None specified",
        )

        start_time = time.perf_counter()
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a model selection expert. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_text = response.choices[0].message.content
        try:
            selection_result = json.loads(result_text)
            selected_name = selection_result.get("selected_model", "nemotron-nano-9b")

            # Validate the selected model exists
            if selected_name not in AVAILABLE_MODELS:
                selected_name = "nemotron-nano-9b"  # fallback

            model_info = AVAILABLE_MODELS[selected_name]
            return SelectionResult(
                provider=model_info["provider"],
                model=model_info["model"],
                reasoning=selection_result.get("reasoning", "Semantically selected"),
                confidence=selection_result.get("confidence", 0.8),
            ), {
                "method": "semantic_selection",
                "prompt_sent": prompt,
                "raw_response": selection_result,
                "latency_ms": latency_ms,
                "selected_model": selected_name,
            }
        except json.JSONDecodeError:
            # Fallback to default model
            model_info = AVAILABLE_MODELS["nemotron-nano-9b"]
            return SelectionResult(
                provider=model_info["provider"],
                model=model_info["model"],
                reasoning="Fallback selection due to parsing error",
                confidence=0.5,
            ), {
                "method": "fallback",
                "error": "Failed to parse semantic selection response",
                "raw_text": result_text,
            }

    def select_model(self, complexity: str, confidence: float | None = None) -> SelectionResult:
        """Select the appropriate model based on complexity.

        Args:
            complexity: Task complexity level.
            confidence: Classification confidence from the classifier (0.0-1.0).

        Returns:
            SelectionResult with provider and model details.
        """
        tier = MODEL_TIERS.get(complexity, MODEL_TIERS["standard"])

        # Use classifier confidence, or default to 0.7 if not provided
        selection_confidence = confidence if confidence is not None else 0.7

        # Clamp confidence to valid range
        selection_confidence = max(0.0, min(1.0, selection_confidence))

        return SelectionResult(
            provider=tier["provider"],
            model=tier["model"],
            reasoning=tier["description"],
            confidence=selection_confidence,
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
        confidence = classification.get("confidence")
        primary_selection = self.select_model(complexity, confidence)

        if plan_type == PlanType.SINGLE_SHOT:
            return plan_type, None, primary_selection

        # Build steps for multi-step plan
        step_data = classification.get("steps", [])
        steps = []
        for s in step_data:
            step_complexity = s.get("complexity", complexity)
            # Use overall confidence for steps since individual step confidence isn't provided
            step_selection = self.select_model(step_complexity, confidence)
            steps.append(
                Step(
                    step_number=s.get("step_number", len(steps) + 1),
                    task=s.get("task", "Unknown task"),
                    model=step_selection.model,
                    complexity=step_complexity,
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
                    complexity=complexity,
                    depends_on=[],
                    status="pending",
                )
            ]

        return plan_type, steps, primary_selection

    def build_routing_decision(
        self,
        classification: dict[str, Any],
        selected_complexity: str,
        selection: SelectionResult,
        classifier_interaction: ClassifierInteraction | None = None,
    ) -> RoutingDecision:
        """Build a RoutingDecision showing all considered models and why victor was chosen.

        Args:
            classification: The classification result from the classifier.
            selected_complexity: The complexity tier that was selected.
            selection: The final model selection.
            classifier_interaction: The full interaction with the classifier.

        Returns:
            RoutingDecision with full details of the routing process.
        """
        # Build list of all available models, marking which was selected
        available_models = []
        for model_name, model_info in AVAILABLE_MODELS.items():
            # Check if this model is selected based on the selection result
            is_selected = selection.model == model_info["model"]
            available_models.append(
                ModelOption(
                    provider=model_info["provider"],
                    model=model_info["model"],
                    tier="",  # No static tiers anymore
                    name=model_name,
                    description=model_info["description"],
                    economy=model_info.get("economy", ""),
                    responsiveness=model_info.get("responsiveness", ""),
                    selected=is_selected,
                )
            )

        # Build selection reasoning explaining why this model was chosen
        tier_descriptions = {
            "simple": "fast responses with lower capability",
            "standard": "balanced performance and capability",
            "complex": "high capability for complex reasoning tasks",
            "code": "specialized code generation and analysis",
        }

        selection_reasoning = (
            f"Query classified as '{selected_complexity}' complexity. "
            f"Selected {selection.model.split('/')[-1]} because it provides "
            f"{tier_descriptions.get(selected_complexity, 'appropriate capability')}. "
        )

        # Add comparison reasoning
        if selected_complexity == "simple":
            selection_reasoning += (
                "More capable models (Claude Sonnet, Claude Opus) would be overkill "
                "for this straightforward query."
            )
        elif selected_complexity == "standard":
            selection_reasoning += (
                "Llama 8B would lack sufficient reasoning capability, "
                "while Claude Opus would be unnecessarily expensive for this task."
            )
        elif selected_complexity == "complex":
            selection_reasoning += (
                "Simpler models would struggle with the required reasoning depth. "
                "This task requires maximum capability."
            )
        elif selected_complexity == "code":
            selection_reasoning += (
                "DeepSeek specializes in code tasks with better performance "
                "than general-purpose models for this use case."
            )

        confidence = classification.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7

        return RoutingDecision(
            classifier_model="gpt-4o-mini",
            classifier_interaction=classifier_interaction,
            available_models=available_models,
            detected_complexity=selected_complexity,
            classification_reasoning=classification.get("reasoning", "No reasoning provided"),
            selection_reasoning=selection_reasoning,
            confidence=max(0.0, min(1.0, float(confidence))),
        )

    async def route(self, request: RouteRequest) -> dict[str, Any]:
        """Route a request by classifying and creating an execution plan.

        Args:
            request: The route request with messages and options.

        Returns:
            Routing result with type and steps/selection.
        """
        # Classify the query
        classification, classifier_interaction = await self.classify_query(request.messages)

        # If user specified a plan type, respect it
        if request.plan_type == PlanType.MULTI_STEP:
            # Force multi-step even if classifier says otherwise
            classification["plan_type"] = "multi_step"

        plan_type, steps, selection = self.create_plan(classification)
        selected_complexity = classification.get("complexity", "standard")

        # Build the full routing decision with classifier interaction details
        routing_decision = self.build_routing_decision(
            classification, selected_complexity, selection, classifier_interaction
        )

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
            # Update routing decision for user override but keep classifier interaction
            routing_decision = RoutingDecision(
                classifier_model="user_override",
                classifier_interaction=classifier_interaction,
                available_models=routing_decision.available_models,
                detected_complexity=selected_complexity,
                classification_reasoning=classification.get("reasoning", ""),
                selection_reasoning="User explicitly specified the model to use, overriding automatic selection.",
                confidence=1.0,
            )

        result = {
            "type": plan_type.value,
            "selection": selection.model_dump(),
            "routing_decision": routing_decision.model_dump(),
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
