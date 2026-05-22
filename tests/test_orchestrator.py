"""Loop-logic tests for the orchestrator. No LLM calls."""

from __future__ import annotations

from idcs.distinguisher import Distinguisher
from idcs.generator import Generator
from idcs.orchestrator import run_episode
from idcs.schemas import Issue, IssueList, Spec, Task
from tests.fakes import FakeLLM, FakeUserProxy


def _task() -> Task:
    return Task(id="t/1", prompt="Sum two ints.", tests=[])


def _spec(goal: str) -> Spec:
    return Spec(goal=goal)


def test_loop_exits_when_distinguisher_returns_empty() -> None:
    llm = FakeLLM(
        typed_responses=[
            _spec("draft"),  # generator.draft
            IssueList(issues=[]),  # distinguisher.critique → empty
        ]
    )
    trace = run_episode(_task(), Generator(llm), Distinguisher(llm), FakeUserProxy())

    assert len(trace.turns) == 1
    assert trace.turns[0].issues == []
    assert trace.final_spec == _spec("draft")


def test_type1_issue_triggers_revise_without_user_call() -> None:
    issues = IssueList(
        issues=[
            Issue(
                kind="gap",
                route="generator",
                location="postconditions",
                description="missing",
            )
        ]
    )
    llm = FakeLLM(
        typed_responses=[
            _spec("draft"),
            issues,
            _spec("revised"),
            IssueList(issues=[]),
        ]
    )
    user = FakeUserProxy()

    trace = run_episode(_task(), Generator(llm), Distinguisher(llm), user)

    assert len(trace.turns) == 2
    assert trace.turns[0].spec == _spec("draft")
    assert trace.turns[0].issues == issues.issues
    assert trace.turns[0].user_answers == {}  # no user call
    assert trace.turns[1].spec == _spec("revised")
    assert trace.turns[1].issues == []
    assert trace.final_spec == _spec("revised")
    assert user.calls == []  # type-1 must not consult the user


def test_type2_issue_consults_user_and_records_answer() -> None:
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="preconditions[0]",
        description="overflow unspecified",
        suggested_question="Wrap or raise on overflow?",
    )
    llm = FakeLLM(
        typed_responses=[
            _spec("draft"),
            IssueList(issues=[issue]),
            _spec("revised"),
            IssueList(issues=[]),
        ]
    )
    user = FakeUserProxy(answers={"preconditions[0]": "raise"})

    trace = run_episode(_task(), Generator(llm), Distinguisher(llm), user)

    assert user.calls == [("preconditions[0]", "Wrap or raise on overflow?")]
    assert trace.turns[0].user_answers == {"preconditions[0]": "raise"}


def test_type2_dismissal_records_no_answer() -> None:
    issue = Issue(
        kind="ambiguity",
        route="user",
        location="goal",
        description="unclear",
        suggested_question="What do you mean?",
    )
    llm = FakeLLM(
        typed_responses=[
            _spec("draft"),
            IssueList(issues=[issue]),
            _spec("revised"),
            IssueList(issues=[]),
        ]
    )
    user = FakeUserProxy(answers={"goal": None})

    trace = run_episode(_task(), Generator(llm), Distinguisher(llm), user)

    assert user.calls == [("goal", "What do you mean?")]
    assert trace.turns[0].user_answers == {}  # None is treated as dismissal


def test_max_turns_truncates() -> None:
    issue = Issue(
        kind="gap",
        route="generator",
        location="x",
        description="d",
    )
    # Always returns one issue → loop runs to max_turns
    typed: list[Spec | IssueList] = [_spec("v0")]
    for i in range(1, 10):
        typed.append(IssueList(issues=[issue]))
        typed.append(_spec(f"v{i}"))
    llm = FakeLLM(typed_responses=typed)  # type: ignore[arg-type]

    trace = run_episode(
        _task(), Generator(llm), Distinguisher(llm), FakeUserProxy(), max_turns=3
    )

    assert len(trace.turns) == 3
    # last turn still records the issues from D's last critique
    assert trace.turns[-1].issues == [issue]
    # final_spec is the result of the most recent revise
    assert trace.final_spec == _spec("v3")
