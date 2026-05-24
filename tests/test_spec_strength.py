"""Tests for the composite spec-strength score."""

from __future__ import annotations

from idcs.schemas import Field, Issue, Spec, Trace, Turn
from idcs.spec_strength import spec_strength


def _rich_spec() -> Spec:
    return Spec(
        goal="Sort a list",
        inputs=[Field(name="lst", type="list[int]", description="input list")],
        outputs=[Field(name="result", type="list[int]", description="sorted list")],
        preconditions=["lst is a list of ints"],
        postconditions=["result is sorted ascending", "result is a permutation of lst"],
        edge_cases=["empty list", "single element", "already sorted"],
        acceptance_criteria=["passes all test cases"],
    )


def _bare_spec() -> Spec:
    return Spec(goal="do something")


def test_perfect_episode_scores_high() -> None:
    """An episode that converges immediately with a rich spec → high score."""
    trace = Trace(
        task_id="t1",
        turns=[Turn(spec=_rich_spec(), issues=[], user_answers={})],
        final_spec=_rich_spec(),
    )
    score = spec_strength(trace, max_turns=5)
    # attack_survival=30, resolution=25, clarification=20, richness≈12.9, speed=8
    assert score > 80


def test_no_turns_scores_low() -> None:
    trace = Trace(task_id="t2", turns=[], final_spec=None)
    score = spec_strength(trace, max_turns=5)
    assert score == 0.0


def test_bare_spec_lower_than_rich() -> None:
    """A bare spec (goal only) should score lower on richness."""
    rich_trace = Trace(
        task_id="t3",
        turns=[Turn(spec=_rich_spec(), issues=[], user_answers={})],
        final_spec=_rich_spec(),
    )
    bare_trace = Trace(
        task_id="t4",
        turns=[Turn(spec=_bare_spec(), issues=[], user_answers={})],
        final_spec=_bare_spec(),
    )
    assert spec_strength(rich_trace) > spec_strength(bare_trace)


def test_unresolved_issues_lower_score() -> None:
    """Issues that persist across turns reduce the score."""
    gap = Issue(kind="gap", route="generator", location="postconditions", description="missing")
    trace = Trace(
        task_id="t5",
        turns=[
            Turn(spec=_bare_spec(), issues=[gap], user_answers={}),
            Turn(spec=_bare_spec(), issues=[gap], user_answers={}),  # same issue persists
        ],
        final_spec=_bare_spec(),
    )
    score = spec_strength(trace, max_turns=5)
    # resolution = 0/1 = 0, survival = 0/2 = 0, no convergence
    assert score < 40


def test_dismissed_clarifications_reduce_score() -> None:
    """Type-2 questions the user dismissed (no answer) hurt the score."""
    q = Issue(
        kind="ambiguity", route="user", location="goal",
        description="unclear", suggested_question="?",
    )
    answered_trace = Trace(
        task_id="t6",
        turns=[Turn(spec=_bare_spec(), issues=[q], user_answers={"goal": "yes"})],
        final_spec=_bare_spec(),
    )
    dismissed_trace = Trace(
        task_id="t7",
        turns=[Turn(spec=_bare_spec(), issues=[q], user_answers={})],
        final_spec=_bare_spec(),
    )
    assert spec_strength(answered_trace) > spec_strength(dismissed_trace)


def test_score_in_range() -> None:
    """Score should always be in [0, 100]."""
    for spec_fn in [_rich_spec, _bare_spec]:
        for issues in [[], [Issue(kind="gap", route="generator", location="goal", description="x")]]:
            trace = Trace(
                task_id="range",
                turns=[Turn(spec=spec_fn(), issues=issues, user_answers={})],
                final_spec=spec_fn(),
            )
            s = spec_strength(trace, max_turns=5)
            assert 0.0 <= s <= 100.0, f"score {s} out of range"
