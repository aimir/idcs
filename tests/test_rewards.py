from __future__ import annotations

from idcs.rewards import RewardWeights, compute_reward_breakdown, compute_spec_complexity_penalty
from idcs.schemas import Issue, Spec, Trace, Turn


def _spec() -> Spec:
    return Spec(goal="goal")


def test_reward_counts_and_fixed() -> None:
    type1 = Issue(kind="gap", route="generator", location="goal", description="missing")
    type2 = Issue(
        kind="ambiguity",
        route="user",
        location="inputs[0].type",
        description="ambiguous",
        suggested_question="?",
    )
    turn1 = Turn(spec=_spec(), issues=[type1, type2], user_answers={"inputs[0].type": "int"})
    turn2 = Turn(spec=_spec(), issues=[], user_answers={})
    trace = Trace(task_id="t1", turns=[turn1, turn2], final_spec=_spec())

    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "prompt", benchmark_score=0.5, weights=weights)

    assert breakdown.type1_count == 1
    assert breakdown.type1_fixed_count == 1
    assert breakdown.type2_count == 1
    assert breakdown.type2_dismissed_count == 0


def test_type2_dismissed_counts() -> None:
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="goal",
        description="needs clarification",
        suggested_question="?",
    )
    trace = Trace(task_id="t2", turns=[Turn(spec=_spec(), issues=[issue], user_answers={})])
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "prompt", benchmark_score=0.0, weights=weights)
    assert breakdown.type2_dismissed_count == 1


def test_spec_complexity_penalty_handles_missing_spec() -> None:
    penalty = compute_spec_complexity_penalty(None, "prompt", min_ratio=0.9)
    assert penalty == 1.0
