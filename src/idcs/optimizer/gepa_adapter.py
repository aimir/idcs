"""GEPA adapter for optimizing IDCS prompt components.

This module is intentionally isolated from the hand-rolled coevolution loop.
It gives GEPA the two things it needs:

1. an ``evaluate`` method that runs a candidate generator/distinguisher prompt
   pair through the normal IDCS pipeline, and
2. a reflective dataset with concrete but non-lookup-style feedback.

The import of GEPA itself is lazy so this file remains testable without adding
``gepa`` to the core project dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from idcs.benchmark.scoring import ScoreResult, score_detailed
from idcs.coder import Coder
from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.llm import BudgetExceededError, LLMClient
from idcs.optimizer.coevolve import _format_failure_summaries, _hash_prompt
from idcs.optimizer.mutate import Mutator
from idcs.orchestrator import run_episode
from idcs.rewards import RewardWeights, compute_reward_breakdown
from idcs.schemas import RewardBreakdown, Task, Trace
from idcs.user_proxy import NullUserProxy, UserProxy

GENERATOR_COMPONENT = "generator_prompt"
DISTINGUISHER_COMPONENT = "distinguisher_prompt"
CODER_COMPONENT = "coder_prompt"

OPTIMIZED_COMPONENTS = (GENERATOR_COMPONENT, DISTINGUISHER_COMPONENT)


@dataclass(frozen=True)
class IDCSGepaOutput:
    """Compact per-task result returned to GEPA."""

    task_id: str
    entry_point: str | None
    score: float
    benchmark_score: float
    benchmark_delta: float
    regression_penalty: float
    pass_count: int
    total_count: int
    base_pass_rate: float | None
    turn_count: int
    issue_count: int
    error: str | None = None


@dataclass(frozen=True)
class IDCSGepaTrajectory:
    """Opaque trajectory object consumed by ``make_reflective_dataset``."""

    task: Task
    output: IDCSGepaOutput
    trace: Trace | None
    reward: RewardBreakdown
    score_result: ScoreResult | None
    failure_summaries: list[str] = field(default_factory=list)
    code_excerpt: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LocalEvaluationBatch:
    """Fallback with GEPA's EvaluationBatch shape for unit tests."""

    outputs: list[IDCSGepaOutput]
    scores: list[float]
    trajectories: list[IDCSGepaTrajectory] | None = None
    objective_scores: list[dict[str, float]] | None = None


def default_user_factory(task: Task) -> UserProxy:
    """Default for benchmark optimization: no oracle answers."""

    del task
    return NullUserProxy()


def compute_direct_baselines(
    tasks: Sequence[Task],
    llm: LLMClient,
    *,
    coder_prompt: str | None = None,
) -> dict[str, float]:
    """Score direct task-prompt-to-code once per task for anti-regression terms."""

    coder = Coder(llm, prompt=coder_prompt) if coder_prompt is not None else Coder(llm)
    return {
        task.id: score_detailed(task, coder.from_prompt(task.prompt)).pass_rate
        for task in tasks
    }


@dataclass
class IDCSGepaAdapter:
    """Adapter implementing GEPA's custom-system protocol for IDCS."""

    llm: LLMClient
    generator_prompt: str
    distinguisher_prompt: str
    coder_prompt: str | None = None
    generator_llm: LLMClient | None = None
    distinguisher_llm: LLMClient | None = None
    coder_llm: LLMClient | None = None
    mutator_llm: LLMClient | None = None
    user_factory: Callable[[Task], UserProxy] = default_user_factory
    weights: RewardWeights = field(default_factory=RewardWeights)
    baseline_scores: Mapping[str, float] = field(default_factory=dict)
    max_turns: int = 3

    def evaluate(
        self,
        batch: list[Task],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Run a GEPA candidate on a batch of IDCS tasks."""

        outputs: list[IDCSGepaOutput] = []
        scores: list[float] = []
        trajectories: list[IDCSGepaTrajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []

        for task in batch:
            output, trajectory = self._evaluate_task(task, candidate)
            outputs.append(output)
            scores.append(output.score)
            objective_scores.append(
                {
                    "benchmark": output.benchmark_score,
                    "no_regression": 1.0 - min(1.0, output.regression_penalty),
                    "combined_reward": output.score,
                }
            )
            if trajectories is not None:
                trajectories.append(trajectory)

        return _make_evaluation_batch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build GEPA reflection records from captured IDCS trajectories."""

        raw_trajectories = getattr(eval_batch, "trajectories", None)
        if raw_trajectories is None:
            raise ValueError("GEPA requested reflection without captured trajectories.")
        trajectories = cast(list[IDCSGepaTrajectory], raw_trajectories)
        if not trajectories:
            raise ValueError("No trajectories available for GEPA reflection.")

        dataset: dict[str, list[Mapping[str, Any]]] = {}
        for component in components_to_update:
            dataset[component] = [
                _reflective_record(component, candidate, trajectory)
                for trajectory in trajectories
            ]
        return dataset

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Use IDCS' existing mutator as GEPA's proposal hook.

        This lets ``scripts/train_gepa.py`` run with the local Codex backend
        alone, without requiring a second provider key for GEPA's default
        reflection LM. GEPA still owns selection and frontier tracking; this
        method only proposes replacement text for the requested components.
        """

        mutator = Mutator(self.mutator_llm or self.llm)
        proposed: dict[str, str] = {}
        for component in components_to_update:
            current_text = candidate.get(component)
            if current_text is None:
                continue
            feedback = _proposal_feedback(component, reflective_dataset.get(component, []))
            mutations = mutator.mutate(
                current_text,
                feedback,
                role=_component_role(component),
                count=1,
            )
            proposed[component] = mutations[0] if mutations else current_text
        return proposed

    def _evaluate_task(
        self,
        task: Task,
        candidate: dict[str, str],
    ) -> tuple[IDCSGepaOutput, IDCSGepaTrajectory]:
        generator_prompt = cast(
            str,
            _candidate_text(
                candidate,
                GENERATOR_COMPONENT,
                self.generator_prompt,
            ),
        )
        distinguisher_prompt = cast(
            str,
            _candidate_text(
                candidate,
                DISTINGUISHER_COMPONENT,
                self.distinguisher_prompt,
            ),
        )
        coder_prompt = _candidate_text(
            candidate,
            CODER_COMPONENT,
            self.coder_prompt,
            allow_missing=True,
        )

        generator = Generator(self.generator_llm or self.llm, prompt=generator_prompt)
        distinguisher = Distinguisher(
            self.distinguisher_llm or self.llm,
            prompt=distinguisher_prompt,
        )
        active_coder_llm = self.coder_llm or self.llm
        coder = (
            Coder(active_coder_llm, prompt=coder_prompt)
            if coder_prompt is not None
            else Coder(active_coder_llm)
        )

        try:
            trace = run_episode(
                task,
                generator,
                distinguisher,
                self.user_factory(task),
                max_turns=self.max_turns,
            )
            trace.generator_prompt_hash = _hash_prompt(generator_prompt)
            trace.distinguisher_prompt_hash = _hash_prompt(distinguisher_prompt)
            if trace.final_spec is None:
                raise RuntimeError("no final spec produced")
            code = coder.from_spec(trace.final_spec, task.prompt)
            score_result = score_detailed(task, code)
            benchmark = score_result.pass_rate
            breakdown = compute_reward_breakdown(
                trace,
                task.prompt,
                benchmark_score=benchmark,
                weights=self.weights,
                baseline_score=self.baseline_scores.get(task.id),
            )
            trace.benchmark_score = benchmark
            trace.rewards = breakdown
            failure_summaries = _format_failure_summaries(task, score_result)
            output = IDCSGepaOutput(
                task_id=task.id,
                entry_point=task.entry_point,
                score=_optimization_score(breakdown),
                benchmark_score=benchmark,
                benchmark_delta=breakdown.benchmark_delta,
                regression_penalty=breakdown.regression_penalty,
                pass_count=score_result.pass_count,
                total_count=score_result.total_count,
                base_pass_rate=score_result.base_pass_rate,
                turn_count=len(trace.turns),
                issue_count=sum(len(turn.issues) for turn in trace.turns),
            )
            trajectory = IDCSGepaTrajectory(
                task=task,
                output=output,
                trace=trace,
                reward=breakdown,
                score_result=score_result,
                failure_summaries=failure_summaries,
                code_excerpt=_truncate(code, 2000),
            )
            return output, trajectory
        except BudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - per-example failures are scores, not crashes.
            error = f"{type(exc).__name__}: {exc}"
            breakdown = RewardBreakdown(
                benchmark_score=0.0,
                r_generator=-1.0,
                r_distinguisher=-1.0,
            )
            output = IDCSGepaOutput(
                task_id=task.id,
                entry_point=task.entry_point,
                score=0.0,
                benchmark_score=0.0,
                benchmark_delta=0.0,
                regression_penalty=0.0,
                pass_count=0,
                total_count=0,
                base_pass_rate=None,
                turn_count=0,
                issue_count=0,
                error=error,
            )
            trajectory = IDCSGepaTrajectory(
                task=task,
                output=output,
                trace=None,
                reward=breakdown,
                score_result=None,
                failure_summaries=[f"{task.id}: evaluation error: {error}"],
                error=error,
            )
            return output, trajectory


def seed_candidate(
    *,
    generator_prompt: str,
    distinguisher_prompt: str,
    coder_prompt: str | None = None,
) -> dict[str, str]:
    """Return the initial GEPA candidate for IDCS prompt optimization."""

    candidate = {
        GENERATOR_COMPONENT: generator_prompt,
        DISTINGUISHER_COMPONENT: distinguisher_prompt,
    }
    if coder_prompt is not None:
        candidate[CODER_COMPONENT] = coder_prompt
    return candidate


def _optimization_score(breakdown: RewardBreakdown) -> float:
    combined = (breakdown.r_generator + breakdown.r_distinguisher) / 2.0
    return max(0.0, min(1.0, combined))


def _candidate_text(
    candidate: Mapping[str, str],
    component: str,
    fallback: str | None,
    *,
    allow_missing: bool = False,
) -> str | None:
    text = candidate.get(component, fallback)
    if text is None:
        if allow_missing:
            return None
        raise ValueError(f"GEPA candidate missing {component!r}.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"GEPA candidate component {component!r} is empty.")
    return cleaned


def _reflective_record(
    component: str,
    candidate: Mapping[str, str],
    trajectory: IDCSGepaTrajectory,
) -> Mapping[str, Any]:
    trace = trajectory.trace
    output = trajectory.output
    return {
        "Task": {
            "prompt": trajectory.task.prompt,
            "entry_point": trajectory.task.entry_point,
        },
        "Component under update": component,
        "Current component text": _truncate(candidate.get(component, ""), 4000),
        "Component-specific guidance": _component_guidance(component),
        "Score": {
            "gepa_score": output.score,
            "benchmark_score": output.benchmark_score,
            "benchmark_delta_vs_direct": output.benchmark_delta,
            "regression_penalty": output.regression_penalty,
            "plus_tests": f"{output.pass_count}/{output.total_count}",
            "base_pass_rate": output.base_pass_rate,
            "turn_count": output.turn_count,
            "issue_count": output.issue_count,
            "error": output.error,
        },
        "Reward breakdown": trajectory.reward.model_dump(),
        "Final spec": trace.final_spec.model_dump() if trace and trace.final_spec else None,
        "Issues by turn": _issues_by_turn(trace),
        "Failed hidden-test feedback": trajectory.failure_summaries,
        "Generated code excerpt": trajectory.code_excerpt,
        "Instruction": (
            "Infer reusable semantic rules from the task, trace, and failures. "
            "Do not copy task ids, exact hidden inputs, or lookup tables into the prompt."
        ),
    }


def _component_guidance(component: str) -> str:
    if component == GENERATOR_COMPONENT:
        return (
            "Improve the spec generator so it writes concrete edge semantics, "
            "acceptance criteria, and postconditions before code generation."
        )
    if component == DISTINGUISHER_COMPONENT:
        return (
            "Improve the distinguisher so it flags missing semantics that the "
            "generator can fix, while avoiding noisy or unhelpful user questions."
        )
    if component == CODER_COMPONENT:
        return "Improve the coder prompt without hiding task-specific lookup rules in it."
    return "Improve this text component while preserving its role."


def _component_role(component: str) -> str:
    if component == GENERATOR_COMPONENT:
        return "generator"
    if component == DISTINGUISHER_COMPONENT:
        return "distinguisher"
    if component == CODER_COMPONENT:
        return "coder"
    return component


def _proposal_feedback(
    component: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    if not records:
        return (
            f"GEPA selected {component} for mutation but no reflective records "
            "were available. Make a conservative, role-preserving improvement."
        )
    concise = "\n\n".join(
        _record_feedback_summary(index, record)
        for index, record in enumerate(records, 1)
    )
    serialized = json.dumps(list(records), indent=2, ensure_ascii=False, default=str)
    return (
        f"GEPA selected {component} for mutation based on these reflective "
        "records. Improve the prompt to address recurring failures and reward "
        "signals. Preserve the role and schema contract. Do not encode task ids, "
        "exact hidden inputs, or lookup tables.\n\n"
        "High-signal summary:\n"
        f"{concise}\n\n"
        "Full reflective context, truncated if needed:\n"
        f"{_truncate(serialized, 6000)}"
    )


def _record_feedback_summary(index: int, record: Mapping[str, Any]) -> str:
    score = record.get("Score", {})
    failures = record.get("Failed hidden-test feedback", [])
    task = record.get("Task", {})
    guidance = record.get("Component-specific guidance", "")
    return (
        f"Record {index}\n"
        f"- Task: {_safe_json(task)}\n"
        f"- Score: {_safe_json(score)}\n"
        f"- Guidance: {guidance}\n"
        f"- Hidden-test failures: {_safe_json(failures)}"
    )


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _issues_by_turn(trace: Trace | None) -> list[dict[str, Any]]:
    if trace is None:
        return []
    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(trace.turns, 1):
        rows.append(
            {
                "turn": index,
                "issues": [
                    {
                        "kind": issue.kind,
                        "route": issue.route,
                        "location": issue.location,
                        "description": issue.description,
                        "suggested_question": issue.suggested_question,
                    }
                    for issue in turn.issues
                ],
                "user_answers": turn.user_answers,
            }
        )
    return rows


def _make_evaluation_batch(
    *,
    outputs: list[IDCSGepaOutput],
    scores: list[float],
    trajectories: list[IDCSGepaTrajectory] | None,
    objective_scores: list[dict[str, float]],
) -> Any:
    try:
        from gepa.core.adapter import EvaluationBatch  # type: ignore[import-not-found]
    except ImportError:
        return LocalEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=objective_scores,
    )


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
