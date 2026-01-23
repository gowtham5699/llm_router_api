"""Executor for LLM calls via OpenRouter API.

Handles single-shot and sequential multi-step execution with context passing.
Tracks latency and aggregates responses.
"""

import json
import time
from typing import Any

import litellm

from app.config import settings
from app.models.schemas import (
    CandidateResponse,
    JudgeResult,
    Message,
    PlanType,
    RouteResponse,
    SelectionResult,
    Step,
    TournamentResult,
)


class ExecutionResult:
    """Result of a single LLM execution."""

    def __init__(
        self,
        content: str,
        latency_ms: float,
        usage: dict[str, int] | None = None,
        model: str | None = None,
    ):
        self.content = content
        self.latency_ms = latency_ms
        self.usage = usage or {}
        self.model = model


class Executor:
    """Executes LLM calls via OpenRouter API."""

    def __init__(self, openrouter_api_key: str | None = None):
        """Initialize the executor.

        Args:
            openrouter_api_key: OpenRouter API key. If not provided, uses settings.
        """
        self.api_key = openrouter_api_key or settings.openrouter_api_key
        if not self.api_key:
            raise ValueError("OpenRouter API key is required for execution")

    async def _execute_call(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ExecutionResult:
        """Execute a single LLM call.

        Args:
            messages: Messages to send to the model.
            model: Model identifier (e.g., 'openrouter/anthropic/claude-3.5-sonnet').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            ExecutionResult with content, latency, and usage.
        """
        start_time = time.perf_counter()

        call_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "api_key": self.api_key,
        }

        if temperature is not None:
            call_params["temperature"] = temperature
        # Default to 1000 tokens to avoid credit issues
        call_params["max_tokens"] = max_tokens if max_tokens is not None else 1000

        response = await litellm.acompletion(**call_params)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        content = response.choices[0].message.content or ""

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ExecutionResult(
            content=content,
            latency_ms=latency_ms,
            usage=usage,
            model=model,
        )

    def _provider_for_model(self, model: str) -> str:
        if model.startswith("openrouter/"):
            return "openrouter"
        if model.startswith("gpt-") or model.startswith("o1-"):
            return "openai"
        return "unknown"

    def _api_key_for_model(self, model: str) -> str | None:
        provider = self._provider_for_model(model)
        if provider == "openrouter":
            return self.api_key
        if provider == "openai":
            return settings.openai_api_key
        return None

    async def run_tournament(
        self,
        messages: list[dict[str, str]],
        candidate_models: list[str],
        judge_model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str | None, TournamentResult]:
        candidates: list[CandidateResponse] = []

        for model in candidate_models:
            try:
                result = await self._execute_call(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                candidates.append(
                    CandidateResponse(
                        provider=self._provider_for_model(model),
                        model=model,
                        content=result.content,
                        latency_ms=result.latency_ms,
                        usage=result.usage,
                        error=None,
                    )
                )
            except Exception as e:
                candidates.append(
                    CandidateResponse(
                        provider=self._provider_for_model(model),
                        model=model,
                        content=None,
                        latency_ms=None,
                        usage=None,
                        error=str(e),
                    )
                )

        judge: JudgeResult | None = None
        winner_model: str | None = None

        # Judge only among successful candidates
        judge_candidates = [c for c in candidates if c.content is not None]
        if judge_candidates:
            judge_api_key = self._api_key_for_model(judge_model)
            if not judge_api_key:
                # If we can't call judge model, pick first successful candidate deterministically
                winner_model = judge_candidates[0].model
            else:
                prompt_parts = [
                    "You are judging responses from multiple LLMs.",
                    "Pick the single best response for the USER based on: correctness, completeness, clarity, following instructions, and formatting.",
                    "",
                    "IMPORTANT COST-AWARE RULE:",
                    "- Models are listed in order from CHEAPEST to MOST EXPENSIVE.",
                    "- If multiple responses are correct and similarly clear, PREFER the cheaper model (earlier in the list).",
                    "- Do NOT reward unnecessary verbosity, emojis, or extra friendliness unless the user explicitly asked for it.",
                    "- A concise correct answer from a cheap model should beat a verbose correct answer from an expensive model.",
                    "",
                    "Respond ONLY as valid JSON with keys: winner_model (string), reasoning (string), scores (object mapping model->0-10).",
                    "\nCANDIDATE RESPONSES (ordered cheapest to most expensive):",
                ]
                for idx, c in enumerate(judge_candidates, start=1):
                    prompt_parts.append(
                        f"\n[{idx}] MODEL: {c.model}\nRESPONSE:\n{c.content}\n"
                    )
                judge_prompt = "\n".join(prompt_parts)

                try:
                    judge_resp = await litellm.acompletion(
                        model=judge_model,
                        api_key=judge_api_key,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a strict JSON-only evaluator.",
                            },
                            {"role": "user", "content": judge_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=700,
                        response_format={"type": "json_object"},
                    )
                    raw = judge_resp.choices[0].message.content or "{}"
                    parsed = json.loads(raw)
                    winner_model = parsed.get("winner_model")
                    if winner_model not in {c.model for c in judge_candidates}:
                        winner_model = judge_candidates[0].model
                    judge = JudgeResult(
                        model=judge_model,
                        winner_model=winner_model,
                        reasoning=str(parsed.get("reasoning") or ""),
                        scores=parsed.get("scores") if isinstance(parsed.get("scores"), dict) else None,
                    )
                except Exception:
                    winner_model = judge_candidates[0].model

        tournament = TournamentResult(candidates=candidates, judge=judge)
        return winner_model, tournament

    async def execute_single_shot(
        self,
        messages: list[Message],
        selection: SelectionResult,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RouteResponse:
        """Execute a single-shot request.

        Args:
            messages: Conversation messages.
            selection: Model selection from meta-router.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            RouteResponse with content and metadata.
        """
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        result = await self._execute_call(
            messages=msg_dicts,
            model=selection.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return RouteResponse(
            plan_type=PlanType.SINGLE_SHOT,
            selection=selection,
            content=result.content,
            steps=None,
            usage=result.usage,
            metadata={
                "latency_ms": result.latency_ms,
                "executed_model": result.model,
            },
        )

    async def execute_multi_step(
        self,
        messages: list[Message],
        steps: list[Step],
        selection: SelectionResult,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tournament_per_step: bool = False,
        candidate_models: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
    ) -> RouteResponse:
        """Execute a multi-step plan sequentially with context passing.

        Steps are executed in order based on dependencies. Each step receives
        the outputs of its dependent steps as context.

        Args:
            messages: Original conversation messages.
            steps: Steps to execute.
            selection: Primary model selection.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per step.

        Returns:
            RouteResponse with completed steps and aggregated metadata.
        """
        executed_steps: list[Step] = []
        step_outputs: dict[int, str] = {}
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        step_latencies: dict[int, float] = {}
        step_tournaments: dict[int, TournamentResult] = {}

        # Sort steps by step_number to ensure correct order
        sorted_steps = sorted(steps, key=lambda s: s.step_number)

        for step in sorted_steps:
            # Build context from dependent steps
            context_parts = []
            for dep_num in step.depends_on:
                if dep_num in step_outputs:
                    dep_step = next(
                        (s for s in sorted_steps if s.step_number == dep_num), None
                    )
                    if dep_step:
                        context_parts.append(
                            f"[Result from step {dep_num} - {dep_step.task}]:\n{step_outputs[dep_num]}"
                        )

            # Build messages for this step
            step_messages = [{"role": m.role, "content": m.content} for m in messages]

            # Add context from previous steps if any
            if context_parts:
                context_message = {
                    "role": "system",
                    "content": "Previous step results:\n\n" + "\n\n".join(context_parts),
                }
                step_messages.insert(1, context_message)  # After initial system message

            # Add step-specific instruction
            step_messages.append(
                {
                    "role": "user",
                    "content": f"[Current task - Step {step.step_number}]: {step.task}",
                }
            )

            # Use step's model if specified, otherwise use primary selection
            model = step.model or selection.model

            # Mark step as running
            step.status = "running"

            try:
                if tournament_per_step and candidate_models:
                    # Put the "recommended" model first for deterministic tie-breaking
                    step_candidates = [model] + [m for m in candidate_models if m != model]
                    winner, tournament = await self.run_tournament(
                        messages=step_messages,
                        candidate_models=step_candidates,
                        judge_model=judge_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    step_tournaments[step.step_number] = tournament
                    model = winner or model
                    step.model = model

                    # Use the winner's content from tournament instead of making duplicate call
                    winner_candidate = next(
                        (c for c in tournament.candidates if c.model == model and c.content),
                        None
                    )
                    if winner_candidate:
                        step.output = winner_candidate.content
                        step.status = "completed"
                        step_outputs[step.step_number] = winner_candidate.content
                        step_latencies[step.step_number] = winner_candidate.latency_ms or 0.0
                        if winner_candidate.usage:
                            for key in total_usage:
                                total_usage[key] += winner_candidate.usage.get(key, 0)
                    else:
                        # Fallback: make a fresh call if winner content not found
                        result = await self._execute_call(
                            messages=step_messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        step.output = result.content
                        step.status = "completed"
                        step_outputs[step.step_number] = result.content
                        step_latencies[step.step_number] = result.latency_ms
                        if result.usage:
                            for key in total_usage:
                                total_usage[key] += result.usage.get(key, 0)
                else:
                    # No tournament - make normal call
                    result = await self._execute_call(
                        messages=step_messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # Handle empty responses
                    output_content = result.content
                    if not output_content or output_content.strip() == "":
                        output_content = f"[Model returned empty response for step {step.step_number}]"
                    step.output = output_content
                    step.status = "completed"
                    step_outputs[step.step_number] = output_content
                    step_latencies[step.step_number] = result.latency_ms
                    if result.usage:
                        for key in total_usage:
                            total_usage[key] += result.usage.get(key, 0)

            except Exception as e:
                step.status = "failed"
                step.output = f"Error: {str(e)}"
                step_latencies[step.step_number] = 0.0

            executed_steps.append(step)

        # Calculate total latency
        total_latency = sum(step_latencies.values())

        return RouteResponse(
            plan_type=PlanType.MULTI_STEP,
            selection=selection,
            content=None,
            steps=executed_steps,
            usage=total_usage if any(total_usage.values()) else None,
            step_tournaments=step_tournaments if step_tournaments else None,
            metadata={
                "total_latency_ms": total_latency,
                "step_latencies_ms": step_latencies,
                "steps_completed": sum(1 for s in executed_steps if s.status == "completed"),
                "steps_failed": sum(1 for s in executed_steps if s.status == "failed"),
            },
        )

    async def execute(
        self,
        messages: list[Message],
        plan_type: PlanType,
        selection: SelectionResult,
        steps: list[Step] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tournament_per_step: bool = False,
        candidate_models: list[str] | None = None,
        judge_model: str = "gpt-4o-mini",
    ) -> RouteResponse:
        """Execute a routing plan.

        Dispatches to single_shot or multi_step based on plan type.

        Args:
            messages: Conversation messages.
            plan_type: Type of execution plan.
            selection: Model selection.
            steps: Steps for multi-step execution.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            RouteResponse with execution results.
        """
        if plan_type == PlanType.SINGLE_SHOT:
            return await self.execute_single_shot(
                messages=messages,
                selection=selection,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            if not steps:
                raise ValueError("Steps are required for multi-step execution")
            return await self.execute_multi_step(
                messages=messages,
                steps=steps,
                selection=selection,
                temperature=temperature,
                max_tokens=max_tokens,
                tournament_per_step=tournament_per_step,
                candidate_models=candidate_models,
                judge_model=judge_model,
            )


# Singleton instance for convenience
_executor_instance: Executor | None = None


def get_executor() -> Executor:
    """Singleton Executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = Executor()
    return _executor_instance


async def execute_plan(
    messages: list[Message],
    plan_type: PlanType,
    selection: SelectionResult,
    steps: list[Step] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> RouteResponse:
    """Convenience function to execute a plan.

    Args:
        messages: Conversation messages.
        plan_type: Type of execution plan.
        selection: Model selection.
        steps: Steps for multi-step execution.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.

    Returns:
        RouteResponse with execution results.
    """
    executor = get_executor()
    return await executor.execute(
        messages=messages,
        plan_type=plan_type,
        selection=selection,
        steps=steps,
        temperature=temperature,
        max_tokens=max_tokens,
    )
