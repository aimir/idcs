"""Reward computation for generator and distinguisher episodes."""

from __future__ import annotations

from dataclasses import dataclass

from idcs.schemas import Issue, RewardBreakdown, Spec, Trace


@dataclass(frozen=True)
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.05
    delta: float = 0.1
    epsilon: float = 0.05
    min_spec_ratio: float = 0.8
    # Cap on type-2 (user-routed) questions per episode. Above the cap D is
    # penalized by ``excess_type2_penalty`` per extra question — large enough
    # that asking a 6th question can't be made profitable by being slightly
    # useful. design.md: "penalize on the cap, not the average".
    max_type2_per_episode: int = 5
    excess_type2_penalty: float = 1.0


def compute_reward_breakdown(
    trace: Trace,
    task_prompt: str,
    *,
    benchmark_score: float,
    weights: RewardWeights | None = None,
    baseline_score: float | None = None,
) -> RewardBreakdown:
    """Compute per-term reward components for one trace."""
    weights = weights or RewardWeights()

    type1_count = 0
    type2_count = 0
    type2_dismissed_count = 0
    for turn in trace.turns:
        for issue in turn.issues:
            if issue.route == "generator":
                type1_count += 1
            else:
                type2_count += 1
                if issue.location not in turn.user_answers:
                    type2_dismissed_count += 1

    type1_fixed_count = _count_type1_fixed(trace)
    clarification_count = sum(len(turn.user_answers) for turn in trace.turns)
    useful_clarification_rate = 0.0
    if baseline_score is not None and clarification_count > 0:
        useful_clarification_rate = (benchmark_score - baseline_score) / clarification_count

    spec_complexity_penalty = compute_spec_complexity_penalty(
        trace.final_spec, task_prompt, weights.min_spec_ratio
    )

    r_generator = (
        weights.alpha * benchmark_score
        - weights.beta * type1_count
        - weights.gamma * spec_complexity_penalty
    )
    excess_type2 = max(0, type2_count - weights.max_type2_per_episode)
    r_distinguisher = (
        weights.alpha * benchmark_score
        + weights.beta * type1_fixed_count
        + weights.delta * useful_clarification_rate
        - weights.epsilon * type2_dismissed_count
        - weights.excess_type2_penalty * excess_type2
    )

    return RewardBreakdown(
        benchmark_score=benchmark_score,
        type1_count=type1_count,
        type1_fixed_count=type1_fixed_count,
        type2_count=type2_count,
        type2_dismissed_count=type2_dismissed_count,
        useful_clarification_rate=useful_clarification_rate,
        spec_complexity_penalty=spec_complexity_penalty,
        r_generator=r_generator,
        r_distinguisher=r_distinguisher,
    )


def compute_spec_complexity_penalty(
    spec: Spec | None,
    task_prompt: str,
    min_ratio: float,
) -> float:
    """Penalize specs that are much shorter than the task prompt."""
    if spec is None:
        return 1.0
    spec_len = len(spec.model_dump_json())
    prompt_len = max(1, len(task_prompt))
    ratio = spec_len / prompt_len
    if ratio >= min_ratio:
        return 0.0
    return min_ratio - ratio


def _count_type1_fixed(trace: Trace) -> int:
    if len(trace.turns) < 2:
        return 0
    fixed = 0
    for current, nxt in zip(trace.turns, trace.turns[1:], strict=False):
        current_keys = {_issue_key(issue) for issue in current.issues if issue.route == "generator"}
        if not current_keys:
            continue
        next_keys = {_issue_key(issue) for issue in nxt.issues}
        fixed += sum(1 for key in current_keys if key not in next_keys)
    return fixed


def _issue_key(issue: Issue) -> tuple[str, str]:
    """Key for matching the *same underlying issue* across consecutive turns.

    Uses only ``(kind, location)``. ``description`` is LLM-generated free
    text and may legitimately vary when D re-flags the same gap — including
    that in the key would let G score a "fix" just by D rewording. ``route``
    is excluded too: if D moves a type-1 to a type-2 next turn (same gap,
    different routing decision), the gap is still present and shouldn't
    count as fixed.
    """
    return (issue.kind, issue.location)
