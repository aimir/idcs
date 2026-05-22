"""Round-trip tests for the core data types."""

from __future__ import annotations

from idcs.schemas import (
    Field,
    Issue,
    RewardBreakdown,
    Spec,
    Task,
    Test,
    Trace,
    Turn,
)


def test_task_round_trip() -> None:
    task = Task(
        id="mbpp/1",
        prompt="Write a function that returns the sum of two integers.",
        tests=[Test(id="t1", code="assert add(1, 2) == 3")],
    )
    restored = Task.model_validate_json(task.model_dump_json())
    assert restored == task


def test_spec_round_trip() -> None:
    spec = Spec(
        goal="Add two ints",
        inputs=[Field(name="a", type="int", description="first addend")],
        outputs=[Field(name="result", type="int", description="sum")],
        preconditions=["a, b are finite ints"],
        postconditions=["result == a + b"],
        acceptance_criteria=["passes example tests"],
    )
    assert Spec.model_validate_json(spec.model_dump_json()) == spec


def test_spec_defaults_empty() -> None:
    spec = Spec(goal="x")
    assert spec.inputs == []
    assert spec.preconditions == []
    assert spec.acceptance_criteria == []


def test_issue_type_2_with_question() -> None:
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="preconditions",
        description="Behavior on integer overflow is unspecified.",
        suggested_question="Should overflow wrap, saturate, or raise?",
    )
    restored = Issue.model_validate_json(issue.model_dump_json())
    assert restored.route == "user"
    assert restored.suggested_question == "Should overflow wrap, saturate, or raise?"


def test_issue_type_1_no_question() -> None:
    issue = Issue(
        kind="gap",
        route="generator",
        location="postconditions[0]",
        description="Missing return value contract.",
    )
    assert issue.suggested_question is None


def test_trace_round_trip() -> None:
    spec = Spec(goal="x")
    trace = Trace(
        task_id="t1",
        turns=[
            Turn(
                spec=spec,
                issues=[
                    Issue(
                        kind="gap",
                        route="generator",
                        location="postconditions",
                        description="missing",
                    )
                ],
                user_answers={"preconditions[2]": "wrap on overflow"},
            )
        ],
        final_spec=spec,
        benchmark_score=0.75,
        rewards=RewardBreakdown(r_generator=0.5, r_distinguisher=0.6),
    )
    restored = Trace.model_validate_json(trace.model_dump_json())
    assert restored == trace
