from __future__ import annotations

from idcs.rewards import RewardWeights, compute_reward_breakdown, compute_spec_complexity_penalty
from idcs.schemas import Issue, Spec, Trace, Turn


def _minimal_spec() -> Spec:
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
    turn1 = Turn(
        spec=_minimal_spec(),
        issues=[type1, type2],
        user_answers={"inputs[0].type": "int"},
    )
    turn2 = Turn(spec=_minimal_spec(), issues=[], user_answers={})
    trace = Trace(task_id="t1", turns=[turn1, turn2], final_spec=_minimal_spec())

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
    trace = Trace(task_id="t2", turns=[Turn(spec=_minimal_spec(), issues=[issue], user_answers={})])
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "prompt", benchmark_score=0.0, weights=weights)
    assert breakdown.type2_dismissed_count == 1


def test_type2_dismissed_skips_unaskable_issues() -> None:
    """A route=user issue without a suggested_question can't be asked, so it
    must not count as 'dismissed' — that would penalize D for D's own
    schema-allowed omission rather than for the user-proxy's silence."""
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="goal",
        description="needs clarification",
        suggested_question=None,
    )
    trace = Trace(task_id="t", turns=[Turn(spec=_minimal_spec(), issues=[issue], user_answers={})])
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "prompt", benchmark_score=0.0, weights=weights)
    assert breakdown.type2_count == 1
    assert breakdown.type2_dismissed_count == 0


def test_spec_complexity_penalty_handles_missing_spec() -> None:
    penalty = compute_spec_complexity_penalty(None, "prompt", min_ratio=0.9)
    assert penalty == 1.0


def test_type1_fixed_ignores_description_rewording() -> None:
    """D rewording the same (kind, location) doesn't count as 'fixed'."""
    type1_orig = Issue(
        kind="gap",
        route="generator",
        location="goal",
        description="missing return type",
    )
    type1_reword = Issue(
        kind="gap",
        route="generator",
        location="goal",
        description="goal lacks a return-type annotation",
    )
    trace = Trace(
        task_id="t",
        turns=[
            Turn(spec=_minimal_spec(), issues=[type1_orig]),
            Turn(spec=_minimal_spec(), issues=[type1_reword]),
        ],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "p", benchmark_score=0.0, weights=weights)
    # Same (kind, location) reappears with different description — still present.
    assert breakdown.type1_fixed_count == 0


def test_type1_fixed_credits_actual_fix() -> None:
    """When the (kind, location) does not reappear, the issue is fixed."""
    type1 = Issue(
        kind="gap",
        route="generator",
        location="goal",
        description="missing return type",
    )
    trace = Trace(
        task_id="t",
        turns=[
            Turn(spec=_minimal_spec(), issues=[type1]),
            Turn(spec=_minimal_spec(), issues=[]),
        ],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "p", benchmark_score=0.0, weights=weights)
    assert breakdown.type1_fixed_count == 1


def test_excess_type2_penalizes_distinguisher_not_generator() -> None:
    """Asking more than max_type2 questions penalizes D, leaves G unaffected."""
    issues = [
        Issue(
            kind="ambiguity",
            route="user",
            location=f"inputs[{i}].type",
            description="?",
            suggested_question="?",
        )
        for i in range(7)
    ]
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=issues, user_answers={})],
        final_spec=_minimal_spec(),
    )
    capped = RewardWeights(min_spec_ratio=0.0, max_type2_per_episode=5)
    uncapped = RewardWeights(min_spec_ratio=0.0, max_type2_per_episode=999)

    b_cap = compute_reward_breakdown(trace, "p", benchmark_score=0.5, weights=capped)
    b_no_cap = compute_reward_breakdown(trace, "p", benchmark_score=0.5, weights=uncapped)

    # G's reward doesn't reference type2_count — unaffected.
    assert b_cap.r_generator == b_no_cap.r_generator
    # D pays for the 2 excess questions.
    assert b_cap.r_distinguisher < b_no_cap.r_distinguisher
    excess = 7 - 5
    expected_delta = capped.excess_type2_penalty * excess
    assert abs((b_no_cap.r_distinguisher - b_cap.r_distinguisher) - expected_delta) < 1e-9


def test_useful_clarification_rate_is_zero_without_baseline() -> None:
    """When baseline_score is None, the rate term contributes 0 (current proxy)."""
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="goal",
        description="?",
        suggested_question="?",
    )
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=[issue], user_answers={"goal": "yes"})],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "p", benchmark_score=0.8, weights=weights)
    assert breakdown.useful_clarification_rate == 0.0


def test_useful_clarification_rate_uses_baseline_when_provided() -> None:
    """With baseline_score set, the rate is (benchmark - baseline) / clarifications."""
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="goal",
        description="?",
        suggested_question="?",
    )
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=[issue], user_answers={"goal": "yes"})],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(
        trace, "p", benchmark_score=0.8, weights=weights, baseline_score=0.5
    )
    # 1 clarification answered, Δ = 0.3 → rate = 0.3
    assert abs(breakdown.useful_clarification_rate - 0.3) < 1e-9
    assert abs(breakdown.benchmark_delta - 0.3) < 1e-9
    assert breakdown.regression_penalty == 0.0


def test_baseline_regression_penalizes_both_roles() -> None:
    """Candidates that lose hidden tests versus direct baseline are down-ranked."""
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_spec(), issues=[])],
        final_spec=_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0, regression_penalty_weight=2.0)

    breakdown = compute_reward_breakdown(
        trace,
        "p",
        benchmark_score=0.25,
        weights=weights,
        baseline_score=0.5,
    )

    assert breakdown.benchmark_delta == -0.25
    assert breakdown.regression_penalty == 0.25
    assert breakdown.r_generator == -0.25
    assert breakdown.r_distinguisher == -0.25


def test_useful_clarification_rate_zero_when_no_clarifications() -> None:
    """No user answers → division by zero avoided; rate stays 0 even with baseline."""
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=[], user_answers={})],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(
        trace, "p", benchmark_score=0.9, weights=weights, baseline_score=0.5
    )
    assert breakdown.useful_clarification_rate == 0.0


def test_at_cap_no_excess_penalty() -> None:
    """Exactly at the cap, no excess penalty fires."""
    issues = [
        Issue(
            kind="ambiguity",
            route="user",
            location=f"x{i}",
            description="?",
            suggested_question="?",
        )
        for i in range(5)
    ]
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=issues, user_answers={})],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0, max_type2_per_episode=5)
    b = compute_reward_breakdown(trace, "p", benchmark_score=0.5, weights=weights)
    # All 5 dismissed → epsilon penalty fires, but no excess penalty
    assert b.type2_dismissed_count == 5
    # excess_type2 = 0 so the only D penalty is epsilon * 5
    expected = (
        weights.alpha * 0.5
        + weights.beta * 0  # no type1_fixed
        + weights.delta * 0  # no baseline → useful_rate = 0
        - weights.epsilon * 5
    )
    assert abs(b.r_distinguisher - expected) < 1e-9


def test_route_change_does_not_count_as_fixed() -> None:
    """D moving a gap from generator-routed to user-routed isn't a fix."""
    t1 = Issue(kind="gap", route="generator", location="goal", description="a")
    t2 = Issue(
        kind="gap",
        route="user",
        location="goal",
        description="b",
        suggested_question="?",
    )
    trace = Trace(
        task_id="t",
        turns=[Turn(spec=_minimal_spec(), issues=[t1]), Turn(spec=_minimal_spec(), issues=[t2])],
        final_spec=_minimal_spec(),
    )
    weights = RewardWeights(min_spec_ratio=0.0)
    breakdown = compute_reward_breakdown(trace, "p", benchmark_score=0.0, weights=weights)
    assert breakdown.type1_fixed_count == 0
